import argparse
import json
import os
import sys
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict, concatenate_datasets
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from tqdm import tqdm


MODELS = ["qwen2.5-coder:1.5b", "codesensei"]
MODEL_LABELS = {
    "qwen2.5-coder:1.5b": "base",
    "codesensei": "codesensei",
}

SEED = 3407
MAX_LOG_CHARS = 6000
# Always save results in: CodeSensei/evaluation/benchmark_results (relative to this file)
SCRIPT_DIR = Path(__file__).resolve().parent                 # .../CodeSensei/evaluation
PROJECT_DIR = SCRIPT_DIR.parent                              # .../CodeSensei
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "benchmark_results"       # .../CodeSensei/evaluation/benchmark_results
RESULTS_DIR = DEFAULT_RESULTS_DIR

STOP_TOKENS = ["<|im_end|>", "</s>"]

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful coding assistant that fixes Python code from stack traces.",
    ),
    (
        "user",
        "Fix the following Python program using the error trace when available.\n"
        "Return ONLY the full fixed program (no markdown, no explanation, no extra tests).\n\n"
        "### BUGGY CODE:\n{code}\n\n"
        "### ERROR TRACE:\n{log}\n",
    ),
])


def normalize_output(text: str) -> str:
    if text is None:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def truncate_log(log: str, max_chars: int = MAX_LOG_CHARS) -> str:
    if log is None:
        return ""
    log = str(log)
    if len(log) <= max_chars:
        return log
    return "...(truncated)...\n" + log[-max_chars:]


def build_error_trace(example: dict) -> str:
    stderr = (example.get("stderr") or "").strip()
    err = (example.get("error") or "").strip()
    status = (example.get("original_status") or "").strip()
    stdout = (example.get("stdout") or "").strip()

    trace = stderr if stderr else err
    if not trace:
        trace = f"No stack trace captured. original_status={status}. stdout={stdout}"
    return truncate_log(trace, MAX_LOG_CHARS)


import re

FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
HEADER_CODE_RE = re.compile(
    r"(?:^|\n)\s*(?:CORRECTED CODE|FULL FIXED CODE|FIXED CODE)\s*:?\s*\n(.*)",
    re.DOTALL | re.IGNORECASE
)
STOP_SECTIONS_RE = re.compile(r"\n(?:EXPLANATION|ROOT CAUSE|NOTES|ANALYSIS)\s*:.*", re.IGNORECASE)

def extract_assistant_code(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()

    # 0) Strip chat special tokens if present
    if "<|im_start|>assistant" in cleaned:
        cleaned = cleaned.split("<|im_start|>assistant", 1)[-1]
    if "<|im_end|>" in cleaned:
        cleaned = cleaned.split("<|im_end|>", 1)[0]
    cleaned = cleaned.strip()

    # 1) Prefer fenced code blocks
    m = FENCE_RE.search(cleaned)
    if m:
        return m.group(1).strip()

    # 2) If they wrote "CORRECTED CODE:" etc.
    m = HEADER_CODE_RE.search(cleaned)
    if m:
        code = m.group(1).strip()
        code = STOP_SECTIONS_RE.split(code, 1)[0].strip()
        return code

    # 3) Fallback: try to drop leading explanation before first code-looking line
    starters = ("import ", "from ", "def ", "class ", "if __name__")
    lines = cleaned.splitlines()
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith(starters) or re.match(r"^[A-Za-z_]\w*\s*=", s):
            candidate = "\n".join(lines[i:]).strip()
            candidate = STOP_SECTIONS_RE.split(candidate, 1)[0].strip()
            return candidate

    # 4) Last resort: return as-is
    return cleaned



def _limit_resources():
    try:
        import resource

        soft_cpu = 2
        resource.setrlimit(resource.RLIMIT_CPU, (soft_cpu, soft_cpu))

        soft_mem = 512 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (soft_mem, soft_mem))
    except Exception:
        pass


def run_python_code(code: str, stdin_data: str, timeout_s: float):
    if not code.strip():
        return False, "", "EMPTY_CODE", 1, False

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        path = f.name
        f.write(code)

    try:
        proc = subprocess.run(
            [sys.executable, path],
            input=(stdin_data or ""),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            preexec_fn=_limit_resources if os.name != "nt" else None,
        )
        return (
            proc.returncode == 0,
            proc.stdout,
            proc.stderr,
            proc.returncode,
            False,
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        err = e.stderr or "TIMEOUT"
        return False, out, err, -1, True
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def is_consistent(example: dict) -> bool:
    fail = example.get("fail") or ""
    stderr = (example.get("stderr") or "")
    code_lines = [ln.strip() for ln in stderr.splitlines() if ln.startswith("    ")]
    if not code_lines:
        return True
    failing_line = code_lines[-1].strip()
    if len(failing_line) < 6:
        return True
    return failing_line in fail


def split_by_problem_id(dataset, seed: int) -> DatasetDict:
    pids = dataset.unique("problem_id")
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)

    n = len(pids)
    n_test = max(20, int(0.05 * n))
    n_val = max(20, int(0.05 * n))

    test_pids = set(pids[:n_test])
    val_pids = set(pids[n_test:n_test + n_val])
    train_pids = set(pids[n_test + n_val:])

    def pid_in(pidset):
        return lambda x: x["problem_id"] in pidset

    return DatasetDict({
        "train": dataset.filter(pid_in(train_pids)),
        "validation": dataset.filter(pid_in(val_pids)),
        "test": dataset.filter(pid_in(test_pids)),
    })


def load_bugnet_splits(seed: int) -> DatasetDict:
    raw = load_dataset("alexjercan/bugnet")
    dataset = concatenate_datasets([raw["train"], raw["validation"], raw["test"]])

    dataset = dataset.filter(lambda x: x.get("language") == "Python")
    dataset = dataset.filter(lambda x: isinstance(x.get("fail"), str) and isinstance(x.get("pass"), str))
    dataset = dataset.filter(lambda x: len(x["fail"].strip()) > 0 and len(x["pass"].strip()) > 0)
    dataset = dataset.filter(lambda x: (x.get("original_status") is None) or (x["original_status"] != "Accepted"))
    dataset = dataset.filter(is_consistent)

    return split_by_problem_id(dataset, seed=seed)


def sample_test_split(test_ds, n: int, seed: int):
    if n is None or n <= 0 or n >= len(test_ds):
        return test_ds
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(test_ds), size=n, replace=False)
    return test_ds.select(indices.tolist())


def results_paths(n: int, seed: int, timeout_s: float, out_dir: str):
    key = f"n{n}_seed{seed}_timeout{timeout_s}"
    out_path = Path(out_dir)
    return (
        out_path / f"benchmark_{key}.csv",
        out_path / f"benchmark_{key}.summary.json",
    )


def compute_metrics(df: pd.DataFrame, label: str) -> dict:
    mask = df["has_expected_output"] == True
    if mask.any():
        pass_at_1 = df.loc[mask, f"{label}_ok"].mean()
    else:
        pass_at_1 = float("nan")

    runtime_error_rate = df[f"{label}_runtime_error"].mean()
    timeout_rate = df[f"{label}_timed_out"].mean()
    stderr_rate = df[f"{label}_stderr_present"].mean()
    empty_output_rate = df[f"{label}_empty_output"].mean()
    avg_output_len = df[f"{label}_output_len"].mean()

    return {
        "pass@1": pass_at_1,
        "runtime_error_rate": runtime_error_rate,
        "timeout_rate": timeout_rate,
        "stderr_rate": stderr_rate,
        "empty_output_rate": empty_output_rate,
        "avg_output_len": avg_output_len,
    }


def build_failure_examples(df: pd.DataFrame, limit: int = 5):
    failing = df[(~df["base_ok"]) | (~df["codesensei_ok"])].copy()
    failing = failing.head(limit)
    examples = []
    for _, row in failing.iterrows():
        examples.append({
            "id": row.get("id"),
            "problem_id": row.get("problem_id"),
            "buggy_code": row.get("buggy_code"),
            "error_trace": row.get("error_trace"),
            "input": row.get("input"),
            "expected_output": row.get("expected_output"),
            "base": {
                "ok": row.get("base_ok"),
                "stdout": row.get("base_stdout"),
                "stderr": row.get("base_stderr"),
                "timed_out": row.get("base_timed_out"),
                "exit_code": row.get("base_exit_code"),
                "generated_code": row.get("base_generated_code"),
            },
            "codesensei": {
                "ok": row.get("codesensei_ok"),
                "stdout": row.get("codesensei_stdout"),
                "stderr": row.get("codesensei_stderr"),
                "timed_out": row.get("codesensei_timed_out"),
                "exit_code": row.get("codesensei_exit_code"),
                "generated_code": row.get("codesensei_generated_code"),
            },
        })
    return examples


def run_benchmark(n: int, seed: int, timeout_s: float, out_dir: str, force: bool = False, progress_callback=None):
    os.makedirs(out_dir, exist_ok=True)
    csv_path, summary_path = results_paths(n, seed, timeout_s, out_dir)

    if not force and csv_path.exists() and summary_path.exists():
        df = pd.read_csv(csv_path)
        summary = json.loads(summary_path.read_text())
        return df, summary, True

    ds = load_bugnet_splits(seed=seed)
    test_ds = sample_test_split(ds["test"], n=n, seed=seed)

    llms = {
        model: ChatOllama(model=model, temperature=0, stop=STOP_TOKENS)
        for model in MODELS
    }
    
    # --- Preflight: verify models respond ---
    for model_name, llm in llms.items():
        try:
            _ = llm.invoke("ping").content
        except Exception as e:
            raise RuntimeError(f"Model preflight failed for {model_name}: {e}")


    rows = []
    iterator = enumerate(test_ds)
    if progress_callback is None:
        iterator = tqdm(iterator, total=len(test_ds))

    for idx, ex in iterator:
        buggy = ex["fail"]
        trace = build_error_trace(ex)
        stdin_data = ex.get("input", "") or ""
        expected_out = ex.get("output", None)
        pid = ex.get("problem_id")

        messages = PROMPT.format_messages(code=buggy, log=trace)

        row = {
            "id": idx,
            "problem_id": pid,
            "buggy_code": buggy,
            "error_trace": trace,
            "input": stdin_data,
            "expected_output": expected_out,
            "has_expected_output": expected_out is not None,
        }

        for model_name, llm in llms.items():
            label = MODEL_LABELS.get(model_name, model_name.replace(":", "_"))
            try:
                resp = llm.invoke(messages)
                text = resp.content
            except Exception as e:
                row.update({
                    f"{label}_ok": False,
                    f"{label}_stdout": "",
                    f"{label}_stderr": f"INVOKE_ERROR: {e}",
                    f"{label}_timed_out": False,
                    f"{label}_runtime_error": True,
                    f"{label}_empty_output": True,
                    f"{label}_stderr_present": True,
                    f"{label}_output_len": 0,
                    f"{label}_exit_code": -1,
                    f"{label}_generated_code": "",
                })
                continue

            gen_code = extract_assistant_code(text)
            ok_run, stdout, stderr, rc, timed_out = run_python_code(
                gen_code, stdin_data, timeout_s=timeout_s
            )
            normalized_stdout = normalize_output(stdout)
            normalized_expected = normalize_output(expected_out) if expected_out is not None else None
            output_match = normalized_expected is not None and normalized_stdout == normalized_expected

            runtime_error = (not timed_out) and (rc != 0)
            stderr_present = bool((stderr or "").strip())
            empty_output = normalized_stdout == ""

            row.update({
                f"{label}_ok": output_match,
                f"{label}_stdout": stdout,
                f"{label}_stderr": stderr,
                f"{label}_timed_out": timed_out,
                f"{label}_runtime_error": runtime_error,
                f"{label}_empty_output": empty_output,
                f"{label}_stderr_present": stderr_present,
                f"{label}_output_len": len(normalized_stdout),
                f"{label}_exit_code": rc,
                f"{label}_generated_code": gen_code,
            })

        rows.append(row)
        if progress_callback is not None:
            progress_callback(idx + 1, len(test_ds))

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    summary = {
        "config": {
            "models": MODELS,
            "n": n,
            "seed": seed,
            "timeout_s": timeout_s,
        },
        "metrics": {
            "base": compute_metrics(df, "base"),
            "codesensei": compute_metrics(df, "codesensei"),
        },
        "failure_examples": build_failure_examples(df, limit=5),
    }

    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n--- Benchmark Summary ---")
    for label in ["base", "codesensei"]:
        metrics = summary["metrics"][label]
        print(f"{label}: pass@1={metrics['pass@1']:.2%} runtime_error={metrics['runtime_error_rate']:.2%} timeout={metrics['timeout_rate']:.2%}")

    return df, summary, False

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--timeout", type=float, default=2.0)

    # Default to CodeSensei/evaluation/benchmark_results
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_RESULTS_DIR))

    parser.add_argument("--force", action="store_true", help="Force re-run even if cached")
    args = parser.parse_args()

    run_benchmark(
        n=args.n,
        seed=args.seed,
        timeout_s=args.timeout,
        out_dir=args.out_dir,
        force=args.force,
    )



if __name__ == "__main__":
    main()
