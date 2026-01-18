# 🥋 CodeSensei: Domain-Adapted LLM for Multi-Modal Debugging

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: Qwen2.5-Coder](https://img.shields.io/badge/Base%20Model-Qwen2.5--Coder--1.5B-blueviolet)](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
[![Fine-Tuned With: Unsloth](https://img.shields.io/badge/Fine--Tuned%20With-Unsloth-green)](https://github.com/unslothai/unsloth)

> **"Code assistants often fail because they read code, but don't read the error logs. CodeSensei reads both."**

## 🚀 Project Overview
CodeSensei is a specialized Low-Rank Adaptation (LoRA) of **Qwen 2.5 Coder**, fine-tuned to act as an automated reliability engineer. Unlike standard coding assistants that blindly guess fixes based on syntax, CodeSensei consumes **multi-modal input (Source Code + Runtime Stack Traces)** to perform root cause analysis and precise patch generation.

Achieved a **+6% absolute improvement in Pass@1 (execution)** compared to the base model on the BugNet (`alexjercan/bugnet`) holdout split.

## 🏗️ System Architecture
1.  **Data Pipeline:** Fine-tuned on BugNet (`alexjercan/bugnet`) with Python-only samples, using program-level `{buggy_code, error_trace, fixed_code}` pairs and a leakage-safe split by `problem_id`.
2.  **Fine-Tuning:** Trained using **Unsloth** (QLoRA) on a Tesla T4 GPU to minimize memory footprint while maximizing instruction adherence.
3.  **Inference Engine:** Quantized to GGUF for low-latency CPU inference via **Ollama**.

## 📊 Performance Benchmarks
| Metric | Base Qwen 2.5 Coder | CodeSensei (Fine-Tuned) | Lift |
| :--- | :---: | :---: | :---: |
| **Pass@1 (Execution)** | 67% | **73%** | 🔺 +6% |
| **Runtime Error Rate** | 23% | **20%** | 🔻 -3% |
| **Timeout Rate** | 0% | **1%** | 🔺 +1% |

Pass@1 (execution) means the percentage of test samples where the model's first generated program produces the exact expected stdout for the provided input (after whitespace normalization).

## ⚡ Quick Start
### 1. Installation
Clone the repo and install dependencies:
```bash
git clone [https://github.com/yourname/CodeSensei.git](https://github.com/yourname/CodeSensei.git)
pip install -r requirements.txt
```

### 2. Model Setup (Required)
Before using the app, you need the fine-tuned model in Ollama:
1. Install and run **Ollama** on your machine.
2. Open the fine-tuning notebook in **Google Colab**: `finetuning/CodeSensei_fineTuned (4).ipynb`.
3. Run all cells to produce the fine-tuned model artifacts.
4. Save the model into Ollama and name it **`codesensei`**.

Once done, you can run the Streamlit app or benchmark with the fine-tuned model.
