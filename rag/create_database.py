import argparse
import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Assuming you have this file in the same directory
from rag.get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = str(BASE_DIR / "data")
CHROMA_PATH = str(BASE_DIR / "chroma")

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    """
    CodeSensei Adaptation:
    We modify the separators to prioritize 'class ' and 'exception '.
    This forces the chunk to start exactly where a new Python error is defined.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        # ---------------------------------------------------------
        # THE "CODESENSEI" CHANGE IS HERE:
        # We put "class " first. The splitter checks these in order.
        # It means: "Split here if you see 'class ', otherwise try 'exception ', etc."
        separators=["class ", "exception ", "\n\n", "\n", " ", ""],
        # ---------------------------------------------------------
        
        chunk_size=1000, # Reduced size to keep chunks focused on one error type
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = building_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")
        

def building_chunk_ids(chunks):
    
    last_page_index = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata["source"]
        page = chunk.metadata["page"]
        current_page_id = f"{source}_page_{page}"
        
        if last_page_index != current_page_id:
            current_chunk_index = 0
        else:
            current_chunk_index += 1
            
        chunk_id = f"{current_page_id}_chunk_{current_chunk_index}"
        last_page_index = current_page_id
        
        chunk.metadata["id"] = chunk_id
        
    return chunks


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
