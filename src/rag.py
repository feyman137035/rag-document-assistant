"""
RAG retrieval pipeline: load FAISS index, retrieve relevant chunks, and answer
questions with conversational memory via ConversationalRetrievalChain.
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# Project root is one level above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_index"


def load_vectorstore(index_path: Path = FAISS_INDEX_PATH) -> FAISS:
    """Load the persisted FAISS vector store using HuggingFace embeddings."""
    print(f"Loading FAISS index from: {index_path}")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("Vector store loaded successfully.")
    return vectorstore


def build_rag_chain(
    index_path: Path = FAISS_INDEX_PATH,
    k: int = 4,
) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain with buffer memory for multi-turn Q&A.

    Args:
        index_path: Path to the saved FAISS index directory.
        k: Number of top chunks to retrieve per query.
    """
    print("Building RAG chain...")

    vectorstore = load_vectorstore(index_path)

    print(f"Creating retriever (top {k} chunks)...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    print("Initializing ChatOllama (mistral, temperature=0)...")
    llm = ChatOllama(model="mistral", temperature=0)

    print("Setting up conversation memory (chat_history)...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    print("Assembling ConversationalRetrievalChain...")
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True,
    )
    print("RAG chain ready.\n")
    return chain


def _format_page_numbers(source_documents) -> str:
    """Extract unique 1-based page numbers from retrieved source chunks."""
    pages = []
    for doc in source_documents:
        page = doc.metadata.get("page")
        if page is not None:
            # PyPDFLoader stores 0-based page indices
            pages.append(int(page) + 1)
    if not pages:
        return "unknown"
    return ", ".join(str(p) for p in sorted(set(pages)))


def ask(chain: ConversationalRetrievalChain, question: str) -> dict:
    """Run a single question through the chain and print the answer with sources."""
    print(f"\nQuestion: {question}")
    print("-" * 60)

    result = chain.invoke({"question": question})
    answer = result["answer"]
    source_docs = result.get("source_documents", [])

    print(f"Answer: {answer}")
    print(f"Source pages: {_format_page_numbers(source_docs)}")
    return result


if __name__ == "__main__":
    load_dotenv()

    print("=" * 60)
    print("RAG Document Q&A — Phase 2 test")
    print("=" * 60)

    chain = build_rag_chain()

    ask(chain, "What is this document about?")
    ask(chain, "Can you elaborate on the first point?")

    print("\n" + "=" * 60)
    print("Done. Memory test complete — the follow-up used chat history.")
    print("=" * 60)
