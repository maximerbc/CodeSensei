from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

def get_embedding_function():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file or environment variables.")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
