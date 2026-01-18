# Evaluation

This module evaluates code-generation models with a focus on **first-try correctness** rather than conversational quality.

Models are compared under identical conditions:
- Base model
- Hybrid RAG + fine-tuned model

All generated programs are **executed automatically** to measure real-world reliability.

---

## Metrics

### 1. pass@1 (Primary Metric)
Proportion of tasks solved **correctly on the first generated attempt**.

A solution counts as correct if:
- The code compiles
- The program runs without crashing
- The output matches the expected behavior

This metric is prioritized because it reflects practical usability in automated coding and debugging scenarios.

---

### 2. Syntax / Compilation Success
Fraction of generated programs that compile without syntax errors.

Used to distinguish:
- Language-level failures
- Logical or runtime failures

---

### 3. Runtime Execution Success
Fraction of compiled programs that execute without runtime exceptions.

Typical failures include:
- Unhandled exceptions
- Infinite loops
- Invalid memory or type operations

---

## Evaluation Procedure

For each task:
1. Generate code using the selected model
2. Extract executable code
3. Run the program in a sandboxed environment
4. Record syntax, runtime, and correctness outcomes

Results are aggregated and stored in `benchmark_results/`.

---

## Design Choice

The evaluation intentionally **penalizes verbosity and retries**.  
Models are evaluated in a **single-shot setting** to reflect realistic developer workflows and to avoid masking failure modes with regeneration.

