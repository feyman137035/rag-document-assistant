from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

def ingest_documents(file_path: str):
    print(f"Loading document: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    print("Creating embeddings (this calls OpenAI)...")
    # embeddings = OpenAIEmbeddings()
    # This downloads a tiny, powerful model that runs completely free on your computer
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("faiss_index")
    print("Done. Vector store saved to faiss_index/")

if __name__ == "__main__":
    ingest_documents("data/sample.pdf")