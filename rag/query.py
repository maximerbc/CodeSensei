import argparse
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag.get_embedding_function import get_embedding_function

BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_PATH = str(BASE_DIR / "chroma")

# --- CODESENSEI PROMPT ---
# We frame the RAG context as "Official Python Documentation" to ground the model.
PROMPT_TEMPLATE = """
You are a helpful coding assistant that fixes Python code from stack traces. 
Use the following Python documentation to analyze the error and provide a fix.

### OFFICIAL DOCUMENTATION (Reference):
{context}

---

### USER DEBUGGING REQUEST:
{question}

### INSTRUCTIONS:
1. Return ONLY the full fixed program (no markdown, no explanation, no extra tests).
"""

def main():
    # Create CLI with support for Files (Real Engineering Workflow)
    parser = argparse.ArgumentParser()
    
    # Option 1: Direct text input (like your old project)
    parser.add_argument("--query", type=str, help="Direct query text.", default=None)
    
    # Option 2: File input (The "Internship Killer" feature)
    parser.add_argument("--file", type=str, help="Path to the buggy python file.", default=None)
    parser.add_argument("--log", type=str, help="Path to the error log file.", default=None)
    
    # Model selection (Default to your fine-tuned model)
    parser.add_argument("--model", default="codesensei", help="Ollama model name (e.g., codesensei or qwen2.5-coder:1.5b)")
    
    args = parser.parse_args()

    # Logic to build the query string
    final_query = ""
    
    if args.file and args.log:
        # Load from files
        try:
            with open(args.file, 'r') as f: code_content = f.read()
            with open(args.log, 'r') as f: log_content = f.read()
            final_query = f"### BUGGY CODE:\n{code_content}\n\n### ERROR TRACE:\n{log_content}"
            print(f"📂 Loaded code from {args.file} and logs from {args.log}")
        except FileNotFoundError as e:
            print(f"❌ Error loading files: {e}")
            return
    elif args.query:
        # Fallback to simple text
        final_query = args.query
    else:
        print("❌ Error: You must provide either --query OR (--file and --log)")
        return

    query_rag(final_query, args.model)


def query_rag(query_text: str, model_name: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    # Note: For debugging, the "Error Trace" usually contains the keywords we need (e.g., "IndexError")
    results = db.similarity_search_with_score(query_text, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print(f"🤖 CodeSensei is thinking using model: {model_name}...")
    
    # Initialize Model
    model = ChatOllama(model=model_name)
    
    # Invoke
    response_message = model.invoke(prompt)
    response_text = response_message.content

    # Output Parsing
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    print("\n" + "="*50)
    print("🔎 RELEVANT DOCS FOUND:")
    for src in sources:
        print(f" - {src}")
    print("="*50 + "\n")
    
    print(f"💡 DIAGNOSIS & FIX:\n{response_text}")
    
    return response_text


if __name__ == "__main__":
    main()
