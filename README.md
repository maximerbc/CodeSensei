# CodeSensei

An experimental Python debugging assistant combining LoRA fine-tuning of **Qwen2.5-Coder-1.5B-Instruct**, retrieval of Python references, and an **Ollama / Streamlit** interface.

Provide Python source code and an error trace to request a complete corrected program. The interface lets you inspect retrieved passages and compare suggestions from installed models. Generated suggestions are not automatically verified in the debugger.

**Status:** The repository contains implementation code and historical training/export outputs. It contains **no validated comparative benchmark results**, and no improvement in repair correctness or latency has been established. Previously published performance figures were unsupported and have been removed, along with placeholder result files.

## How it works

1. **Fine-tuning:** The [Colab notebook](finetuning/CodeSensei_finetuning.ipynb) prepares Python examples from [BugNet](https://huggingface.co/datasets/alexjercan/bugnet), groups them by problem ID, and trains LoRA adapters using Unsloth. Inputs contain buggy code and error information; targets are the dataset's reference corrections.
2. **Retrieval:** PDFs in `data/` are split into overlapping chunks, embedded with OpenAI's `text-embedding-3-small`, and stored locally in Chroma. They include official documentation and third-party guides; see [source attribution](data/Readme.md).
3. **Inference:** The Streamlit debugger and CLI retrieve relevant chunks and send them with code/error information to a local Ollama model. Generation is local; embedding document chunks and retrieval queries calls OpenAI's API.
4. **Experimental evaluation:** A separate script compares the base model and `codesensei` by executing generated Python against a supplied input/output pair. It does **not** use retrieval. See [evaluation limitations](evaluation/Readme.md).

Code and logs are both text inputs. This project does not implement an image/audio debugging model.

## What the saved notebook records

These are historical outputs, not newly reproduced results:

- NVIDIA A100-SXM4-40GB; LoRA training with four-bit base-model loading disabled.
- 2,211 training rows, 490 validation rows, and 423 test rows after filtering and grouping by problem ID.
- 400 recorded training steps (approximately 1.44 epochs), adapter saving, and one example generation.
- A reported successful GGUF conversion, followed by a download lookup `IndexError`. A recovery cell located an approximately 0.92 GB GGUF file and initiated its download.

The repository does not include the adapters or GGUF, an independently verified downloaded model, or predictions supporting a comparison with the base model. Training loss and an example response do not establish repair correctness. The notebook retains the original code and saved outputs, including warnings and the export/download failure, with added explanatory Markdown.

## Run the debugger

### 1. Install the Python environment

Use Python 3.10 or newer and run commands from the repository root. Dependencies are not pinned; this is a prototype rather than a reproducible training release.

```bash
git clone https://github.com/maximerbc/CodeSensei.git
cd CodeSensei
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Start Ollama and install the base model

Install [Ollama](https://ollama.com), start its desktop app or run `ollama serve` in another terminal, then:

```bash
ollama pull qwen2.5-coder:1.5b
```

The base model is enough to try the debugger. The fine-tuned `codesensei` model is optional and must be created separately.

### 3. Configure embeddings and build the document index

Set `OPENAI_API_KEY` in your environment or in a local, ignored `.env` file. Use your own key; do not commit it. Index creation and retrieval queries send text to OpenAI's embedding service and may incur API charges.

```bash
python -m rag.create_database
```

This indexes PDFs in `data/` into `chroma/`. To rebuild the index after changing documents, use `python -m rag.create_database --reset`. The database is generated locally and is not committed.

### 4. Launch the interface or CLI

```bash
python -m streamlit run app.py
```

Select an installed Ollama model, paste code and an error log, and request a suggested fix. The debugger's **Syntax and formatting heuristic** weights syntax validity (60%) and extracted-code character density (40%). It executes no code and measures neither functional correctness nor efficiency.

For file inputs, replace the example paths with your own files:

```bash
python -m rag.query --file path/to/buggy.py --log path/to/error.log --model qwen2.5-coder:1.5b
```

Alternatively, pass the debugging request with `--query`. Always use `--model qwen2.5-coder:1.5b` when only the base model is installed; the CLI's default is `codesensei`.

## Optional: import your own fine-tuned model

The [notebook](finetuning/CodeSensei_finetuning.ipynb) contains the historical GPU training/export workflow. Review its dependency warnings, runtime restart instructions, and recovery cell before attempting a new run. A successful uninterrupted Run All workflow has not been verified.

If you produce a fine-tuned Q4_K_M GGUF, place it in `MyModel/` under the exact filename referenced by the `FROM` line in `MyModel/Modelfile`, or edit that line to match your artifact. Then run:

```bash
cd MyModel
ollama create codesensei -f Modelfile
cd ..
```

The filename alone does not establish that an artifact contains the fine-tuned weights. No trained model is downloaded by these instructions.

## Evaluation and limitations

The benchmark requires both Ollama model names, access to BugNet, and a compatible dataset-loading environment. The historical notebook installs `datasets<4.0.0`; current dataset compatibility has not been validated. The Streamlit Benchmark tab invokes the same script as:

```bash
python -m evaluation.benchmark2 --n 100 --seed 3407 --timeout 2.0 --force
```

This command **executes generated Python locally** in subprocesses with time limits and attempted CPU/memory limits. It is not an isolated sandbox. Use a disposable isolated environment for untrusted generated programs.

Important limits of the existing evaluation:

- Training and evaluation use different filtering rules and problem-ID split proportions. Their shared seed does not establish that the benchmark excludes training examples.
- There is no RAG condition or ablation, so the script cannot establish a retrieval benefit.
- Output matching covers only the supplied input/output pair and does not prove general program correctness. The implementation also records output matches independently of successful process termination.
- Model/API invocation errors are counted in the runtime-error column. Cached filenames do not encode dataset revisions or model identities.
- There are no validated comparative results, measured latency claims, or established repair-quality gains.

The evaluation pipeline is retained as experimental code; redesigning it and collecting new results are future work.

## Development checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py code_quality.py rag evaluation tests
```

These checks do not train models, call embedding APIs, or establish debugging performance.

## Attribution and licensing

The project uses Qwen2.5-Coder, BugNet, Unsloth, Ollama, Streamlit, LangChain, Chroma, and OpenAI embeddings. Retrieval documents are attributed separately in `data/Readme.md`. No project license has been selected or included; third-party materials retain their respective terms.
