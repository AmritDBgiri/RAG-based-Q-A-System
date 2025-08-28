import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def load_documents(data_dir: str):
    docs = []
    data_path = Path(data_dir)
    for p in data_path.rglob("*"):
        if p.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
        elif p.suffix.lower() == ".txt":
            loader = TextLoader(str(p), autodetect_encoding=True)
            docs.extend(loader.load())
    return docs

def main():
    load_dotenv()  # Loads OPENAI_API_KEY from .env
    data_dir = "data"
    index_dir = "index"

    print(f"[INGEST] Scanning documents under: {data_dir!r}")
    documents = load_documents(data_dir)
    if not documents:
        print("[INGEST] No documents found. Please add PDFs/TXT to the 'data/' folder.")
        return

    print(f"[INGEST] Loaded {len(documents)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    print(f"[INGEST] Created {len(chunks)} chunks. Building FAISS index...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)

    # Persist the index
    os.makedirs(index_dir, exist_ok=True)
    vectordb.save_local(index_dir)
    print(f"[INGEST] Index saved to: {index_dir}/")

if __name__ == "__main__":
    main()
