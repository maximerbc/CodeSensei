# Experimental evaluation

There are **no validated comparative benchmark results** in this repository. Earlier performance claims were unsupported; their summary and placeholder prediction CSV have been removed.

## Implemented procedure

`benchmark2.py` loads BugNet, combines its splits, filters Python examples, creates its own problem-ID split, and selects test rows. For each row it asks `qwen2.5-coder:1.5b` and `codesensei` for one full program using code and error information. It performs **no retrieval**.

Generated Python is extracted and run locally with the row's input. The script records stdout, stderr, process status, timeouts, generated code, and normalized output matches. Normalization converts line endings and strips surrounding whitespace; it does not ignore all whitespace differences.

## Metrics and limitations

- `pass@1` is currently the fraction of rows with an expected output whose normalized stdout matches it. A match is recorded independently of process success; this is a limitation, not a validated correctness metric.
- Runtime-error rate includes model invocation failures as well as non-timeout nonzero process exits. Timeout rate records generated-program execution timeouts.
- The debugger's syntax/formatting heuristic is separate from this benchmark and does not measure repair correctness.
- The notebook and script use different filtering rules and split sizes: the notebook uses approximately 10% each for validation/test with a 100-problem minimum where possible; the script uses 5% with a 20-problem minimum. They also use different traceback consistency thresholds. Training/test independence is not established by the shared seed.
- The evaluation uses one supplied input/output pair per row, without a full correctness suite or a RAG ablation.
- Dataset revisions and model digests are not pinned. Cached filenames contain sample count, seed, and timeout only. Use `--force` after changing models, data, or code; pre-existing local caches are not validated results.
- The historical dataset-loading workflow depended on custom dataset code. Compatibility with current `datasets` releases has not been validated.

These limitations need resolution before drawing comparative performance conclusions. No new benchmark has been run as part of repository cleanup.

## Running

From the repository root, with dependencies installed and both Ollama models available:

```bash
python -m evaluation.benchmark2 --n 100 --seed 3407 --timeout 2.0 --force
```

The Streamlit Benchmark tab uses the same implementation. Results are written to the ignored local `evaluation/benchmark_results/` directory. `--n` is a requested sample count; fewer rows may be used if the split is smaller.

**Execution environment:** generated programs run in local subprocesses, not in an isolated sandbox. The script sets a wall-clock timeout and attempts Unix CPU/memory limits; these limits can be unavailable or fail silently. The subprocess retains host filesystem/network access. Use a disposable isolated environment for untrusted generated code.
