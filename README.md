# 📚 Paper Memory

<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-009688.svg)
![Next.js](https://img.shields.io/badge/Next.js-15.5.3-black.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3.27-blue.svg)
![RAGAS](https://img.shields.io/badge/RAGAS-0.4.2-orange.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

**An advanced Retrieval-Augmented Generation (RAG) system for intelligent document interaction**

_Chat with your PDFs using state-of-the-art AI models with multi-collection support, intelligent query classification, and conversation memory_

[Features](#-features) • [Usage](#-usage) • [Architecture](#️-architecture) • [Installation](#-installation) • [API Documentation](#-api-reference) • [Evaluation](#-evaluation-framework)

</div>

---

## 🌟 Overview

Paper Memory is a production-ready RAG application that transforms static PDF documents into an intelligent, conversational knowledge base. Built with FastAPI and Next.js, it combines vector search capabilities with LLM reasoning to provide accurate, context-aware responses from your document collections.

### Why Paper Memory?

- **Multi-Model Support**: Seamlessly switch between Gemini, Groq, and Ollama
- **Intelligent Query Routing**: LLM-powered query classification for optimal results
- **Selective PDF Querying**: Target specific documents across collections
- **Conversation Memory**: Persistent chat history with context retention
- **Evaluation Framework**: Built-in RAGAS metrics for RAG performance

---

## ✨ Features

### 🎯 Core RAG Capabilities

**Multi-Collection Management**

- Organize documents into logical collections
- Full CRUD operations (create, read, update, delete)
- Bulk operations and batch processing

**Advanced Document Processing**

- PyMuPDF-based text extraction with layout preservation
- Intelligent chunking with configurable overlap (1000 chars, 200 overlap)
- Rich metadata: titles, headings, page numbers, document structure
- Automatic title extraction from document headings

**Hybrid Search Modes**

- **Single Collection**: Query within a specific document collection
- **ChatALL Mode**: Search across all collections simultaneously
- **Selective PDF Mode**: Cherry-pick specific PDFs across collections

**Smart Query Classification**

- LLM-based intent detection with few-shot prompting
- Automatic routing between metadata and content queries
- File-specific query handling
- Types: `list_pdfs`, `count_pdfs`, `file_specific_search`, `content_search`

### 🤖 LLM Integration

**Supported Providers**

- **Google Gemini**: gemini-2.5-flash (best quality)
- **Groq**: llama-3.1-8b-instant (fastest inference)
- **Ollama**: Local models (llama3.1, llama3.2, etc.)

**Features**

- Environment-based configuration with priority fallback
- Streaming responses for real-time interaction
- Optimized prompt construction with source attribution
- Temperature control (default: 0.1 for consistency)

### 💬 Conversation Management

- **Persistent Memory**: JSON-based storage with chat sessions
- **History Tracking**: Full conversation history with timestamps
- **Context Retrieval**: Resume previous conversations

### 🎨 Modern Frontend

- **Next.js 15 with React 19**: Latest features and optimal performance
- **Tailwind CSS**: Beautiful, responsive UI with custom components
- **Real-time Updates**: Server-Sent Events (SSE) for streaming responses
- **React Markdown**: Rich markdown rendering with syntax highlighting

### 📊 Evaluation Framework

- **RAGAS Integration**: Comprehensive RAG evaluation metrics
  - Context Precision & Recall
  - Faithfulness
  - Answer Relevancy
  - Answer Correctness

- **Multiple Question Types**:
  - Open-ended questions
  - Multiple-choice questions (MCQ)

---

## 💻 Usage

### Basic Workflow

1. **Create a Collection**:
   - Click "New Collection" in the sidebar
   - Enter a descriptive name (e.g., "Research Papers", "Technical Docs")

2. **Upload PDFs**:
   - Select your collection
   - Click "Upload PDFs"
   - Choose one or multiple PDF files
   - Wait for processing (chunking, embedding, indexing)

3. **Start Chatting**:
   - **Single Collection Mode**: Select a collection and ask questions
   - **ChatALL Mode**: Click "Chat All" to query across all collections
   - **Selective PDF Mode**: Click "Select PDFs" to cherry-pick specific documents

4. **View Sources**:
   - Each response includes source citations
   - Click on sources to see the exact content used
   - View PDF button opens the original document

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Sidebar     │  │  Chat Area   │  │  Collection Manager  │  │
│  │  - Collections│  │  - Messages  │  │  - Upload/Delete     │  │
│  │  - PDF Select│  │  - Streaming │  │  - Rename/View       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/SSE
┌────────────────────────────┴────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    API Layer (app.py)                     │  │
│  │  - 30+ RESTful Endpoints                                  │  │
│  │  - CORS Middleware                                        │  │
│  │  - Request/Response Validation (Pydantic)                │  │
│  └───────────────────────┬──────────────────────────────────┘  │
│                          │                                       │
│  ┌──────────────────────┴───────────────────────────────────┐  │
│  │              Service Layer (Orchestration)                │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │ ChatService          │ Generate responses, manage LLM    │  │
│  │ DocumentProcessor    │ PDF extraction, chunking, embed   │  │
│  │ CollectionManager    │ Vector DB operations, CRUD        │  │
│  │ QueryClassifier      │ LLM-based intent classification   │  │
│  │ MemoryService        │ Conversation persistence          │  │
│  │ PDFSelectionService  │ Multi-PDF selection tracking      │  │
│  │ MetadataService      │ Document metadata operations      │  │
│  │ FileSearchService    │ Content search within files       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│  ┌──────────────────────┴───────────────────────────────────┐  │
│  │                  Data Layer                               │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │ ChromaDB             │ Vector storage & similarity search│  │
│  │ HuggingFace Embed    │ all-MiniLM-L6-v2 embeddings      │  │
│  │ JSON Storage         │ Conversation memory               │  │
│  │ File System          │ Original PDF storage              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│  ┌──────────────────────┴───────────────────────────────────┐  │
│  │                  LLM Factory                              │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │  Priority Order:                                          │  │
│  │  1. Ollama (if USE_LOCAL_LLM=true)                       │  │
│  │  2. Gemini (if GOOGLE_API_KEY set)                       │  │
│  │  3. Groq (if GROQ_API_KEY set)                           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Document Upload**:

   ```
   PDF Upload → PyMuPDF Extraction → Text Chunking → Embedding Generation
   → ChromaDB Storage → Metadata Indexing → Original PDF Storage
   ```

2. **Query Processing**:

   ```
   User Query → Query Classification (LLM) → Route Decision
   → Vector Search (ChromaDB) → Context Assembly → LLM Generation
   → Streaming Response → Memory Storage
   ```

3. **PDF Selection Mode**:
   ```
   User Selects PDFs → Session Management → Filtered Search
   → Selected Collections Only → Context from Selected PDFs
   → Targeted Response
   ```

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.11 or higher
- **Node.js**: 18.x or higher
- **npm** : Latest version
- **API Keys**: At least one of:
  - Google Gemini API Key (recommended)
  - Groq API Key
  - Local Ollama installation

### Backend Setup

```bash
# 1. Clone the repository
git clone https://github.com/KAMRANKHANALWI/papermemory.git
cd papermemory

# 2. Navigate to backend
cd backend

# 3. Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements-lock.txt

# 5. Create environment variables
cat > .env << EOF
# LLM Configuration (choose one)
# Option 1: Google Gemini (recommended)
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
DEFAULT_MODEL_PROVIDER=gemini

# Option 2: Groq
# GROQ_API_KEY=your_groq_api_key_here
# GROQ_MODEL=llama-3.1-8b-instant
# DEFAULT_MODEL_PROVIDER=groq

# Option 3: Local Ollama
# USE_LOCAL_LLM=true
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.1:latest

# Optional: Evaluation
# GROQ_API_KEY=your_groq_api_key_for_ragas
EOF

# 6. Create required directories
mkdir -p data/chroma_db data/pdfs data/memory

# 7. Start the backend server
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup

```bash
# 1. Navigate to frontend directory
cd ../frontend

# 2. Install dependencies
npm install

# 3. Create environment file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# 4. Start development server
npm run dev

# 5. Access the application
# Open http://localhost:3000 in your browser
```

### Docker Setup (Optional)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access:
# - Frontend: http://localhost:3000
# - Backend: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

---

## 📡 API Reference

### Interactive Documentation

**🔗 Full API Docs**: http://localhost:8000/docs (Swagger UI)

Paper Memory provides 30+ REST endpoints. Here's a quick reference:

### Collections

```http
GET    /api/collections                           # List all collections
POST   /api/collections/{name}/upload             # Upload PDFs
DELETE /api/collections/{name}                    # Delete collection
PUT    /api/collections/rename                    # Rename collection
GET    /api/collections/{name}/pdfs/{file}/view   # View PDF
DELETE /api/collections/{name}/pdfs/{file}        # Delete PDF
PUT    /api/collections/pdfs/rename               # Rename PDF
```

### Chat (Server-Sent Events)

```http
GET /api/chat/single/{collection}?query=...&chat_id=...
# Chat with single collection

GET /api/chat/all?query=...&chat_id=...
# Chat across all collections

GET /api/selection/{session_id}/chat?query=...&num_results=25
# Chat with selected PDFs
```

### PDF Selection

```http
POST   /api/selection/{session_id}/select         # Select PDF
POST   /api/selection/{session_id}/deselect       # Deselect PDF
DELETE /api/selection/{session_id}/clear          # Clear all selections
GET    /api/selection/{session_id}/info           # Get selection info
GET    /api/selection/{session_id}/stats          # Get statistics
POST   /api/selection/{session_id}/search         # Search selected PDFs
```

### Memory

```http
GET    /api/memory/{chat_id}/history              # Get conversation history
DELETE /api/memory/{chat_id}/clear                # Clear conversation
DELETE /api/memory/{chat_id}                      # Delete conversation
GET    /api/memory/{chat_id}/summary              # Get summary
```

### Query & Metadata

```http
POST   /api/classify                              # Classify query intent
POST   /api/metadata/search                       # Search within collection
GET    /api/collections/{name}/pdfs               # List PDFs in collection
GET    /api/collections/{name}/pdfs/{file}/metadata  # Get PDF metadata
```

### Response Format (SSE)

All chat endpoints stream Server-Sent Events:

```javascript
data: {"type": "chat_id", "chat_id": "chat_123"}
data: {"type": "content", "content": "The answer is..."}
data: {"type": "sources", "sources": [{...}]}
data: {"type": "end"}
```

**📖 [Complete API Documentation →](./docs/API.md)**

---

## 📊 Evaluation Framework

Paper Memory includes a comprehensive evaluation system using RAGAS (Retrieval-Augmented Generation Assessment).

### Evaluation Scripts

Located in `backend/src/eval/`:

1. **`eval_mcq_ragas.py`**: Multiple-choice question evaluation with RAGAS metrics
2. **`eval_open_ended_ragas.py`**: Open-ended question evaluation with RAGAS metrics
3. **`eval_mcq.py`**: Basic MCQ accuracy evaluation
4. **`rag_mcq.py`**: RAG pipeline for MCQ datasets
5. **`rag_open_ended.py`**: RAG pipeline for open-ended datasets

### Dataset Format

**MCQ Dataset (CSV)**:

```csv
question,option_a,option_b,option_c,option_d,correct_answer
"What is the capital of France?","London","Paris","Berlin","Madrid","b"
```

**Open-Ended Dataset (CSV)**:

```csv
question,ground_truth
"Explain photosynthesis","Photosynthesis is the process by which plants..."
```

### Running Evaluations

```bash
# Navigate to eval directory
cd backend/src/eval

# 1. Run MCQ evaluation with RAGAS
python eval_mcq_ragas.py

# 2. Run open-ended evaluation with RAGAS
python eval_open_ended_ragas.py

# 3. Basic MCQ accuracy test
python eval_mcq.py

# Results are saved to: backend/src/eval/results/
```

### RAGAS Metrics

- **Context Precision**: How relevant are the retrieved chunks?
- **Context Recall**: Are all necessary chunks retrieved?
- **Faithfulness**: Is the answer grounded in the context?
- **Answer Relevancy**: Does the answer address the question?
- **Answer correctness**: How close is the answer to ground truth?

### Custom Dataset Creation

```python
# Create your own evaluation dataset
import pandas as pd

# MCQ format
mcq_data = {
    'question': ['Q1', 'Q2', ...],
    'option_a': ['A1', 'A2', ...],
    'option_b': ['B1', 'B2', ...],
    'option_c': ['C1', 'C2', ...],
    'option_d': ['D1', 'D2', ...],
    'correct_answer': ['a', 'b', ...]
}

df = pd.DataFrame(mcq_data)
df.to_csv('backend/src/eval/datasets/my_mcq.csv', index=False)

# Open-ended format
open_data = {
    'question': ['Q1', 'Q2', ...],
    'ground_truth': ['Ground truth answer 1', 'Ground truth answer 2', ...]
}

df = pd.DataFrame(open_data)
df.to_csv('backend/src/eval/datasets/my_open_ended.csv', index=False)
```

---

## 🔧 Configuration

### Environment Variables

**Backend (.env)**:

```bash
# === LLM Configuration ===
# Choose ONE provider (priority order: Ollama > Gemini > Groq)

# Option 1: Google Gemini (Recommended for best quality)
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
DEFAULT_MODEL_PROVIDER=gemini

# Option 2: Groq (Fast inference)
# GROQ_API_KEY=your_groq_api_key
# GROQ_MODEL=llama-3.1-8b-instant
# DEFAULT_MODEL_PROVIDER=groq

# Option 3: Ollama (Local/Private)
# USE_LOCAL_LLM=true
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.1:latest

# === Optional: Evaluation ===
# GROQ_API_KEY=your_groq_key_for_ragas_eval
# GROQ_MODEL=llama-3.1-8b-instant

# === Database Settings ===
# CHROMA_DB_PATH=data/chroma_db  # Default
# PDF_STORAGE_PATH=data/pdfs     # Default
# MEMORY_STORAGE_PATH=data/memory  # Default

# === Embedding Model ===
# EMBEDDING_MODEL=all-MiniLM-L6-v2  # Default

# === RAG Settings ===
# TOP_K=10                # Number of chunks to retrieve
# CHUNK_SIZE=1000         # Characters per chunk
# CHUNK_OVERLAP=200       # Overlap between chunks
# TEMPERATURE=0.1         # LLM temperature
```

**Frontend (.env.local)**:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Customization Options

#### Chunking Parameters

```python
# In document_processor.py
def chunk_text_content(
    text: str,
    max_chars: int = 1000,    # Increase for larger contexts
    overlap: int = 200        # Increase for better continuity
) -> List[str]:
    ...
```

#### Embedding Model

```python
# In chat_service.py or document_processor.py
self.embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Change to other models:
    # - sentence-transformers/all-mpnet-base-v2 (better quality)
    # - BAAI/bge-small-en-v1.5 (faster)
)
```

#### LLM Parameters

```python
# In llm_factory.py
ChatGroq(
    model=model,
    temperature=0.1,  # Lower = more deterministic
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=2048   # Add for longer responses
)
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Commit**: `git commit -m 'Add amazing feature'`
6. **Push**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

---

## 📝 License

This project is licensed under the MIT License.

---

## ❓ FAQ

**Q: Can I use this with local LLMs only?**
A: Yes! Set `USE_LOCAL_LLM=true` and configure Ollama. No API keys needed.

**Q: How many PDFs can I upload?**
A: No hard limit. Performance depends on your hardware. Tested with 100+ PDFs.

**Q: Can I change the embedding model?**
A: Yes, modify the `model_name` in the embedding initialization. Restart the backend.

**Q: How are conversations stored?**
A: As JSON files in `data/memory/`. Each chat session has a unique file.

**Q: How do I update to the latest version?**
A: `git pull origin main`, reinstall dependencies, restart services.

**Q: Is my data secure?**
A: Data stays on your server. For cloud deployments, use HTTPS and proper security.

**Q: Can I customize the UI?**
A: Yes! All frontend code is in `frontend/src/components/`. Edit freely.

---

<div align="center">

**Built with ❤️ using FastAPI, Next.js, and LangChain**

⭐ **Star this repository if you find it helpful!** ⭐

</div>
