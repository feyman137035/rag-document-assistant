# RAG Document Assistant

An intelligent document Q&A system powered by Retrieval-Augmented Generation (RAG), enabling users to upload PDF documents and interact with them through natural language conversations.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Features

- **Multi-Document Support**: Upload and query across multiple PDFs simultaneously
- **Conversational Memory**: Context-aware conversations that remember previous exchanges
- **Source Citations**: Every answer includes page references and source content for verification
- **Confidence Scoring**: Automatic confidence indicators based on source document diversity
- **Chat Export**: Download complete conversation history as a text file
- **Local & Free**: Uses HuggingFace embeddings and Ollama LLM - no API costs
- **Smart Error Handling**: Graceful handling of scanned/image-only PDFs

## 🛠️ Tech Stack

### Core Technologies
- **LangChain**: Orchestration framework for RAG pipeline
- **FAISS**: Efficient vector similarity search
- **Streamlit**: Interactive web UI
- **Ollama**: Local LLM inference (Mistral model)
- **HuggingFace**: Sentence embeddings (all-MiniLM-L6-v2)

### Key Libraries
- `langchain-classic`: Conversational chains and memory
- `langchain-community`: Document loaders and vector stores
- `langchain-ollama`: Ollama LLM integration
- `langchain-huggingface`: HuggingFace embeddings
- `PyPDFLoader`: PDF text extraction
- `RecursiveCharacterTextSplitter`: Intelligent text chunking

## 📋 Prerequisites

- Python 3.8 or higher
- Ollama installed ([Download](https://ollama.com))
- Virtual environment (recommended)

## 🔧 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-document-assistant
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Ollama and pull the model**
```bash
# Install Ollama from https://ollama.com
# Then pull the model
ollama pull mistral
```

## 🚀 Usage

### Running the Application

```bash
streamlit run src/app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the App

1. **Upload Documents**: Click "Upload PDF file(s)" and select one or more PDFs
2. **Wait for Processing**: The system will extract text, create embeddings, and build the vector store
3. **Ask Questions**: Type questions in the chat input to query your documents
4. **View Sources**: Expand "View Sources" to see page references and content
5. **Export Chat**: Click "📥 Download Chat" to save your conversation
6. **Clear Conversation**: Click "🗑️ Clear Conversation" to start fresh

## 📁 Project Structure

```
rag-document-assistant/
├── data/                    # Sample PDF documents
├── faiss_index/             # Persisted FAISS vector store
├── src/
│   ├── ingest.py           # Phase 1: Document ingestion pipeline
│   ├── rag.py              # Phase 2: RAG retrieval chain with memory
│   └── app.py              # Phase 3: Streamlit UI with agentic features
├── .env                    # Environment variables (GOOGLE_API_KEY - optional)
├── .gitignore
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔍 How It Works

### Phase 1: Document Ingestion (`ingest.py`)
1. Loads PDF using PyPDFLoader
2. Splits text into chunks (500 characters, 50 overlap)
3. Creates embeddings using HuggingFace (all-MiniLM-L6-v2)
4. Builds FAISS vector store and saves to disk

### Phase 2: RAG Pipeline (`rag.py`)
1. Loads FAISS index with matching embeddings
2. Creates retriever fetching top-4 relevant chunks
3. Initializes ChatOllama (Mistral model)
4. Builds ConversationalRetrievalChain with memory
5. Supports multi-turn conversations with context

### Phase 3: Streamlit UI (`app.py`)
1. Accepts multiple PDF uploads
2. Processes documents inline (no pre-processing needed)
3. Merges chunks from all documents into single vector store
4. Provides chat interface with message history
5. Shows confidence scores and source citations
6. Enables chat export and conversation clearing

## 🎯 Key Design Decisions

### Why HuggingFace Embeddings?
- **Free**: No API costs or rate limits
- **Reliable**: Works offline, no dependency on external services
- **Quality**: all-MiniLM-L6-v2 provides excellent semantic understanding

### Why Ollama?
- **Local**: Complete privacy, no data leaves your machine
- **Free**: No API keys or billing required
- **Flexible**: Easy to switch between different models

### Why FAISS?
- **Fast**: Optimized for similarity search
- **Scalable**: Handles large document collections
- **Local**: No external vector database needed

### Why ConversationalRetrievalChain?
- **Memory**: Maintains conversation context across turns
- **Flexible**: Easy to customize prompts and retrieval strategies
- **Production-ready**: Battle-tested in real-world applications

## 🧪 Testing

### Test Multi-Document Support
```bash
# Upload 2-3 PDFs and ask questions spanning all documents
# Verify sidebar shows correct document count
# Check that answers reference multiple sources
```

### Test Confidence Scoring
```bash
# Ask specific questions → should show "High confidence"
# Ask obscure questions → should show "Low confidence"
```

### Test Chat Export
```bash
# Have a conversation, then download chat history
# Verify format: [User]: question, [Assistant]: answer
```

## 🐛 Known Limitations

- **Scanned PDFs**: Image-only PDFs cannot be processed (requires OCR integration)
- **Large Documents**: Processing time scales with document size
- **Ollama Dependency**: Requires local installation and model download
- **Memory Usage**: Large document collections may require significant RAM

## 🔮 Future Enhancements

- [ ] OCR integration for scanned PDFs (Tesseract)
- [ ] Hybrid search (semantic + keyword)
- [ ] Document reranking for improved relevance
- [ ] Support for additional file formats (DOCX, TXT, Markdown)
- [ ] Advanced filtering and search options
- [ ] User authentication and session management
- [ ] Deployment options (Docker, cloud)

## 📝 Environment Variables

Optional: Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here  # Currently unused, kept for future Gemini integration
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Built as a demonstration of RAG architecture and modern AI application development.

## 🙏 Acknowledgments

- LangChain team for the excellent framework
- HuggingFace for open-source embeddings
- Ollama for making local LLMs accessible
- Streamlit for simplifying Python web apps
