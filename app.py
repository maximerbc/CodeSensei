import subprocess
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag.get_embedding_function import get_embedding_function
from code_quality import extract_code_from_response, evaluate_code_quality

# Load environment
load_dotenv()

CHROMA_PATH = str(Path(__file__).resolve().parent / "chroma")

# --- CODESENSEI PROMPT TEMPLATE ---
# Supply retrieved reference material alongside code and error information.
PROMPT_TEMPLATE = """
You are a helpful coding assistant, specialized in Python debugging.
Use the following Python documentation to fix the Python program using the error trace when available.

### RETRIEVED PYTHON REFERENCES (official documentation and third-party guides):
{context}

---

### BUGGY CODE:
{code}

### ERROR TRACE:
{log}

### INSTRUCTIONS:
1.Return ONLY the full fixed program (no markdown, no explanation, no extra tests).
"""


@st.cache_resource
def get_db():
    """Load the Chroma DB once and reuse it."""
    embedding = get_embedding_function()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding,
    )


@st.cache_resource
def get_llm(model_name: str):
    """Load the Ollama model once and reuse it."""
    # Use temperature=0 to reduce sampling variability.
    return ChatOllama(model=model_name, temperature=0)


def get_ollama_models():
    """Return available Ollama model names."""
    try:
        output = subprocess.check_output(["ollama", "list"], text=True)
    except Exception:
        return []

    models = []
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    # Skip header
    for line in lines[1:]:
        model = line.split()[0]
        if model:
            models.append(model)
    return models


def build_prompt_and_context(results, code_input: str, log_input: str):
    # Combine retrieved docs into a single string
    context_text = "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')} "
        f"(page index {doc.metadata.get('page', 'unknown')})\n{doc.page_content}"
        for doc, _ in results
    )
    
    # Format the strict template
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        code=code_input,
        log=log_input
    )
    return prompt, context_text


def main():
    st.set_page_config(page_title="CodeSensei Debugger", layout="wide")
    
    # Header
    st.title("CodeSensei: Python Debugging Assistant")
    st.markdown("""
    Explore suggested fixes from installed Ollama models using code, error traces,
    and retrieved Python references. No repair-quality improvement has been established.
    """)

    tab_debugger, tab_benchmark = st.tabs(["Debugger", "Benchmark"])

    with tab_debugger:
        # --- SIDEBAR CONFIG ---
        available_models = get_ollama_models()
        
        # Prefer the base and fine-tuned models when installed.
        default_models = ["qwen2.5-coder:1.5b", "codesensei"]
        
        # Filter defaults to only what is actually installed
        valid_defaults = [m for m in default_models if m in available_models]
        if not valid_defaults and available_models:
            valid_defaults = [available_models[0]]

        st.sidebar.header("Configuration")
        models_to_compare = st.sidebar.multiselect(
            "Select Models",
            available_models,
            default=valid_defaults,
        )
        
        k = st.sidebar.slider("RAG Retrieval Depth (k)", min_value=1, max_value=5, value=3)

        if not available_models:
            st.info("No Ollama models found. Start Ollama and install a model; see the README.")

        # --- MAIN INPUT AREA (Split View) ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Buggy Code")
            code_input = st.text_area(
                "Paste Python Code here:", 
                height=250, 
                placeholder="def my_func(x):\n    return x / 0",
                key="code_in"
            )

        with col2:
            st.subheader("2. Error Log")
            log_input = st.text_area(
                "Paste Stack Trace here:", 
                height=250, 
                placeholder="ZeroDivisionError: division by zero...",
                key="log_in"
            )

        # --- EXECUTION BUTTON ---
        if st.button("🔍 Suggest a Fix", type="primary"):
            if not models_to_compare:
                st.error("Select an installed Ollama model first.")
            elif not code_input.strip() or not log_input.strip():
                st.error("Please provide BOTH the buggy code and the error log.")
            else:
                with st.spinner("🧠 Retrieving Python References & Generating Suggestions..."):
                    
                    # 1. RAG Search
                    # We combine code + log for the search query to find relevant docs
                    search_query = f"{code_input}\n\n{log_input}"
                    results = get_db().similarity_search_with_score(search_query, k=k)
                    
                    # 2. Build Prompt
                    prompt, context_text = build_prompt_and_context(results, code_input, log_input)

                    # 3. Run Models
                    model_outputs = []
                    for model_name in models_to_compare:
                        try:
                            llm = get_llm(model_name)
                            response = llm.invoke(prompt)
                            answer_text = response.content
                            
                            # This heuristic does not test whether the bug was fixed.
                            final_score, metrics = evaluate_code_quality(answer_text)
                            
                            model_outputs.append((model_name, answer_text, final_score, metrics))
                        except Exception as e:
                            st.error(f"Error running {model_name}: {e}")

                # --- RESULTS DISPLAY ---
                st.divider()
                
                # Display RAG Context (Hidden by default to keep UI clean)
                with st.expander(f"📚 RAG Context Used ({len(results)} chunks retrieved)"):
                    for i, (doc, score) in enumerate(results, start=1):
                        st.markdown(f"**Source:** `{doc.metadata.get('id', 'unknown')}` (Score: {score:.3f})")
                        st.code(doc.page_content, language="text")

                # Display Model Comparison
                st.subheader("Model Comparison")
                st.caption("Syntax and formatting heuristic: 60% syntax validity + 40% code density. "
                           "No code is executed here; this does not measure correctness or efficiency.")
                
                if not model_outputs:
                    st.info("No models selected.")
                else:
                    cols = st.columns(len(model_outputs))
                    for col, (model_name, answer_text, score, metrics) in zip(cols, model_outputs):
                        with col:
                            # Report the heuristic separately from the generated suggestion.
                            st.markdown(f"### 🤖 {model_name}")
                            st.markdown(f"**Syntax and formatting heuristic: {score}/1.0**")
                            st.caption(f"Syntax: {metrics['syntax']} | Code Density: {metrics['density']}")
                            
                            st.divider()
                            
                            # Smart Formatting:
                            # If the model is 'High Density' (mostly code) and didn't use markdown backticks,
                            # we force it into a code block to look pretty.
                            if metrics['density'] > 0.8 and "```" not in answer_text:
                                st.code(answer_text, language='python')
                            else:
                                st.markdown(answer_text)

    with tab_benchmark:
        st.subheader("Benchmark")
        st.markdown("Experimental execution benchmark without RAG. The split differs from the training notebook; "
                    "training/test independence is not established. No validated comparative results are included.")

        st.warning("This benchmark runs generated Python locally with time/resource limits, "
                   "not in an isolated sandbox. It requires both configured Ollama models.")

        col1, col2, col3 = st.columns(3)
        with col1:
            n_samples = st.number_input("Number of test samples", min_value=1, max_value=1000, value=100, step=1)
        with col2:
            seed = st.number_input("Random seed", min_value=1, max_value=999999, value=3407, step=1)
        with col3:
            timeout_s = st.number_input("Timeout (seconds)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

        use_cache = st.checkbox("Use cached results when available", value=True)

        run_clicked = st.button("Run benchmark")
        if "benchmark_df" not in st.session_state:
            st.session_state["benchmark_df"] = None
            st.session_state["benchmark_summary"] = None
            st.session_state["benchmark_from_cache"] = False

        if run_clicked:
            from evaluation.benchmark2 import run_benchmark, RESULTS_DIR, results_paths

            progress = st.progress(0, text="Running benchmark...")
            status = st.empty()

            def _progress_callback(done, total):
                progress.progress(done / total)
                status.write(f"Processed {done}/{total} samples")

            csv_path, summary_path = results_paths(int(n_samples), int(seed), float(timeout_s), RESULTS_DIR)
            force = not use_cache

            if use_cache and Path(csv_path).exists() and Path(summary_path).exists() and not force:
                df = pd.read_csv(csv_path)
                summary = json.loads(Path(summary_path).read_text())
                from_cache = True
            else:
                df, summary, from_cache = run_benchmark(
                    n=int(n_samples),
                    seed=int(seed),
                    timeout_s=float(timeout_s),
                    out_dir=RESULTS_DIR,
                    force=force,
                    progress_callback=_progress_callback,
                )

            progress.empty()
            status.empty()

            st.session_state["benchmark_df"] = df
            st.session_state["benchmark_summary"] = summary
            st.session_state["benchmark_from_cache"] = from_cache

        df = st.session_state.get("benchmark_df")
        summary = st.session_state.get("benchmark_summary")
        from_cache = st.session_state.get("benchmark_from_cache")

        if df is not None and summary is not None:
            if from_cache:
                st.info("Loaded cached results from disk.")

            metrics = summary.get("metrics", {})
            base = metrics.get("base", {})
            codesensei = metrics.get("codesensei", {})

            st.subheader("Summary Metrics")
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric("Base pass@1", f"{base.get('pass@1', float('nan')):.2%}")
                st.metric("Base runtime error", f"{base.get('runtime_error_rate', float('nan')):.2%}")
                st.metric("Base timeout", f"{base.get('timeout_rate', float('nan')):.2%}")
            with mcol2:
                st.metric("CodeSensei pass@1", f"{codesensei.get('pass@1', float('nan')):.2%}")
                st.metric("CodeSensei runtime error", f"{codesensei.get('runtime_error_rate', float('nan')):.2%}")
                st.metric("CodeSensei timeout", f"{codesensei.get('timeout_rate', float('nan')):.2%}")

            chart_df = pd.DataFrame({
                "metric": ["pass@1", "runtime_error", "timeout"],
                "base": [
                    base.get("pass@1", 0.0),
                    base.get("runtime_error_rate", 0.0),
                    base.get("timeout_rate", 0.0),
                ],
                "codesensei": [
                    codesensei.get("pass@1", 0.0),
                    codesensei.get("runtime_error_rate", 0.0),
                    codesensei.get("timeout_rate", 0.0),
                ],
            })
            st.subheader("Comparison Chart")
            st.bar_chart(chart_df.set_index("metric"))



if __name__ == "__main__":
    main()
