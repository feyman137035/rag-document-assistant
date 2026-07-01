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
if "loaded_documents" not in st.session_state:
    st.session_state.loaded_documents = []


def process_documents(uploaded_files):
    """
    Process multiple uploaded PDFs: load, split, embed, and build RAG chain.
    Merges chunks from all documents into a single FAISS vector store.
    """
    all_chunks = []
    loaded_filenames = []
    failed_files = []

    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            try:
                # Load PDF
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                st.info(f"Loaded {len(docs)} pages from {uploaded_file.name}")

                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = splitter.split_documents(docs)
                st.info(f"Split into {len(chunks)} chunks from {uploaded_file.name}")

                # Check if any chunks were created
                if not chunks:
                    st.warning(f"No text extracted from {uploaded_file.name} (may be scanned). Skipping.")
                    failed_files.append(uploaded_file.name)
                    Path(tmp_path).unlink()
                    continue

                # Add source filename to metadata
                for chunk in chunks:
                    chunk.metadata["source"] = uploaded_file.name

                all_chunks.extend(chunks)
                loaded_filenames.append(uploaded_file.name)

                # Clean up temp file
                Path(tmp_path).unlink()

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                failed_files.append(uploaded_file.name)
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()
                continue

        # Check if any chunks were created
        if not all_chunks:
            st.error("No text could be extracted from any PDF. They may be scanned documents or contain only images.")
            st.info("Try PDFs with selectable text.")
            return

        # Show warning for failed files
        if failed_files:
            st.warning(f"Failed to process: {', '.join(failed_files)}")

        # Create embeddings using HuggingFace (free, reliable)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(all_chunks, embeddings)

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
        st.session_state.loaded_documents = loaded_filenames

    st.success(f"{len(loaded_filenames)} document(s) ready! Ask me anything.")


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
    st.markdown("Upload PDFs and ask questions about their content using AI.")

    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("1. Upload PDF(s)")
    st.markdown("2. Wait for processing")
    st.markdown("3. Ask questions")

    st.markdown("---")

    # Show loaded documents
    if st.session_state.loaded_documents:
        st.markdown(f"### 📂 {len(st.session_state.loaded_documents)} document(s) loaded")
        for filename in st.session_state.loaded_documents:
            st.markdown(f"- {filename}")

    st.markdown("---")

    # Summarize button
    if st.button("Summarize Document", disabled=not st.session_state.doc_processed):
        if st.session_state.chain:
            with st.spinner("Generating summary..."):
                result = st.session_state.chain.invoke(
                    {"question": "Give me a structured summary of all key points in these documents"}
                )
                st.markdown("### Summary")
                st.markdown(result["answer"])

    st.markdown("---")

    # Clear conversation button
    if st.button("🗑️ Clear Conversation", disabled=not st.session_state.doc_processed):
        st.session_state.chat_history = []
        if st.session_state.chain:
            st.session_state.chain.memory.clear()
        st.info("Conversation cleared.")
        st.rerun()

    # Export chat button
    if st.session_state.chat_history:
        chat_text = ""
        for msg in st.session_state.chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_text += f"[{role}]: {msg['content']}\n\n"
        st.download_button(
            label="📥 Download Chat",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )


# Main content
st.title("Chat with Your Document")

# File upload
uploaded_files = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    process_documents(uploaded_files)

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

                # Confidence indicator
                unique_pages = len(set(doc.metadata.get("page", 0) for doc in source_docs))
                if unique_pages < 2:
                    st.warning("⚠️ Low confidence — limited relevant content found")
                else:
                    st.success("✅ High confidence — answer grounded in multiple sources")

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
