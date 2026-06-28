"""
Streamlit chat app for RAG-powered Document Q&A using Google Gemini.
"""

import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide",
)

# Initialize session state
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False


def process_document(uploaded_file):
    """
    Process uploaded PDF: load, split, embed, and build RAG chain.
    Uses Google Generative AI embeddings and Gemini LLM.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    with st.spinner("Processing document..."):
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        st.info(f"Loaded {len(docs)} pages")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)
        st.info(f"Split into {len(chunks)} chunks")

        # Check if any chunks were created
        if not chunks:
            st.error("No text could be extracted from this PDF. It may be a scanned document or contain only images.")
            st.info("Try a different PDF with selectable text.")
            return

        # Create embeddings using HuggingFace (free, reliable)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Initialize Ollama LLM (local, free)
        llm = ChatOllama(model="mistral", temperature=0)

        # Setup conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        # Build RAG chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
        )

        # Store in session state
        st.session_state.chain = chain
        st.session_state.doc_processed = True
        st.session_state.chat_history = []

        # Clean up temp file
        Path(tmp_path).unlink()

    st.success("Document ready! Ask me anything.")


def format_sources(source_documents):
    """Format source documents for display."""
    sources = []
    for i, doc in enumerate(source_documents, 1):
        page = doc.metadata.get("page", 0) + 1  # Convert to 1-based
        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        sources.append(f"**Source {i} (Page {page}):**\n{content}")
    return "\n\n".join(sources)


# Sidebar
with st.sidebar:
    st.title("📄 RAG Document Assistant")
    st.markdown("Upload a PDF and ask questions about its content using AI.")

    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("1. Upload a PDF")
    st.markdown("2. Wait for processing")
    st.markdown("3. Ask questions")

    st.markdown("---")

    # Summarize button
    if st.button("Summarize Document", disabled=not st.session_state.doc_processed):
        if st.session_state.chain:
            with st.spinner("Generating summary..."):
                result = st.session_state.chain.invoke(
                    {"question": "Give me a structured summary of all key points in this document"}
                )
                st.markdown("### Summary")
                st.markdown(result["answer"])


# Main content
st.title("Chat with Your Document")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    process_document(uploaded_file)

# Chat interface
if st.session_state.doc_processed:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Ask a question about your document"):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke({"question": user_input})
                answer = result["answer"]
                source_docs = result.get("source_documents", [])

                st.markdown(answer)

                # Show sources
                if source_docs:
                    with st.expander("View Sources"):
                        st.markdown(format_sources(source_docs))

        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    # Guard rail - show warning if trying to chat without document
    if st.session_state.chat_history or st.session_state.get("chat_attempted", False):
        st.warning("Please upload a PDF first.")
