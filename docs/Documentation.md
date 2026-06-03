# PaperMemory — Technical Documentation

> **Version**: 2.0.0 | **Last Updated**: May 2025
> **Maintainer**: Kamran Khan | [GitHub](https://github.com/KAMRANKHANALWI/papermemory)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack & Design Decisions](#2-tech-stack--design-decisions)
3. [Project Structure](#3-project-structure)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Setup & Installation](#5-setup--installation)
6. [API Reference](#6-api-reference)
7. [Evaluation Framework (RAGAS)](#7-evaluation-framework-ragas)
8. [Configuration Reference](#8-configuration-reference)
9. [Developer Notes](#9-developer-notes)

---

## 1. Project Overview

**PaperMemory** is a Retrieval-Augmented Generation (RAG) system that lets users upload collections of PDF documents and ask natural language questions against them. Answers are grounded strictly in the uploaded documents — not in general model knowledge.

It was built as a research tool, originally scoped for querying gut microbiome research papers, but designed to be domain-agnostic.

### What it solves

Standard LLMs hallucinate and cannot reason over private or domain-specific document collections. PaperMemory solves this by:

- Indexing your PDFs into a persistent vector store (ChromaDB)
- Retrieving the most relevant chunks on every query
- Passing those chunks as grounded context to the LLM
- Streaming the response back token-by-token via SSE

### Key Capabilities

| Capability | Description |
|---|---|
| Multi-collection management | Organize PDFs into named collections (e.g., "Gut Microbiome", "Clinical Trials") |
| Selective PDF querying | Cherry-pick specific files across collections for a session |
| ChatALL mode | Broadcast a query across all collections simultaneously |
| Smart query routing | LLM classifies intent and routes to the right handler automatically |
| Conversation memory | Chat history persists across sessions via JSON storage |
| Multi-LLM support | Switch between Gemini, Groq, and local Ollama models |
| RAGAS evaluation | Built-in pipeline to measure RAG quality across 5 metrics |

---

## 2. Tech Stack & Design Decisions

### Backend

| Component | Technology | Why |
|---|---|---|
| API Framework | FastAPI | Async support, automatic OpenAPI docs, Pydantic validation |
| Vector Database | ChromaDB (client + persistent) | Lightweight, file-based persistence, no separate DB server needed |
| Embedding Model | `all-MiniLM-L6-v2` (HuggingFace) | Fast, runs locally, good semantic quality for English text |
| PDF Parsing | PyMuPDF (fitz) | Layout-preserving extraction, heading detection, page metadata |
| LLM Orchestration | LangChain | Prompt management, embedding wrappers, LLM abstractions |
| Streaming | Server-Sent Events (SSE) | Real-time token streaming without WebSocket complexity |
| Memory Storage | JSON files | Simple, zero-dependency, sufficient for single-server deployment |

### Frontend

| Component | Technology | Why |
|---|---|---|
| Framework | Next.js 15 + React 19 | App Router, Server Components, latest features |
| Styling | Tailwind CSS | Utility-first, rapid UI development |
| Markdown Rendering | `react-markdown` | Renders LLM responses with syntax highlighting |
| Streaming Client | `EventSource` API | Native SSE consumption in the browser |

### LLM Providers (Priority Order)

```
1. Ollama  (if USE_LOCAL_LLM=true)   → fully local, no API key
2. Gemini  (if GOOGLE_API_KEY set)   → best quality, recommended
3. Groq    (if GROQ_API_KEY set)     → fastest inference
```

Only one provider is active at runtime. The LLM factory picks based on environment variable presence.

---

## 3. Project Structure

```
papermemory/
├── backend/
│   ├── src/
│   │   ├── app.py                  # FastAPI application entry point, all route definitions
│   │   ├── services/
│   │   │   ├── chat_service.py         # Core RAG logic: retrieve + generate
│   │   │   ├── document_processor.py   # PDF parsing, chunking, embedding, indexing
│   │   │   ├── collection_manager.py   # ChromaDB CRUD: create/delete/rename collections
│   │   │   ├── query_classifier.py     # LLM-based intent classification
│   │   │   ├── memory_service.py       # Conversation history read/write (JSON)
│   │   │   ├── pdf_selection_service.py # Session-based multi-PDF selection state
│   │   │   ├── metadata_service.py     # Document metadata queries (headings, pages, chunks)
│   │   │   └── file_search_service.py  # Content search within a specific PDF
│   │   ├── llm/
│   │   │   └── llm_factory.py          # Instantiates the correct LLM based on .env config
│   │   └── eval/
│   │       ├── eval_mcq_ragas.py       # MCQ evaluation with RAGAS metrics
│   │       ├── eval_open_ended_ragas.py # Open-ended evaluation with RAGAS metrics
│   │       ├── eval_mcq.py             # Basic MCQ accuracy (no RAGAS)
│   │       ├── rag_mcq.py              # RAG pipeline for MCQ datasets
│   │       ├── rag_open_ended.py       # RAG pipeline for open-ended datasets
│   │       ├── datasets/               # CSV evaluation datasets go here
│   │       └── results/                # Evaluation output CSVs written here
│   ├── data/
│   │   ├── chroma_db/              # ChromaDB persistent storage (auto-created)
│   │   ├── pdfs/                   # Original uploaded PDF files (auto-created)
│   │   └── memory/                 # Conversation JSON files (auto-created)
│   ├── requirements.txt
│   ├── requirements-lock.txt
│   └── .env                        # Environment variables (not committed)
│
├── frontend/
│   ├── src/
│   │   ├── app/                    # Next.js App Router pages
│   │   └── components/             # React UI components
│   │       ├── Sidebar.tsx             # Collection list, PDF selector, navigation
│   │       ├── ChatArea.tsx            # Message thread, SSE consumer
│   │       ├── CollectionManager.tsx   # Upload, delete, rename UI
│   │       └── ...
│   ├── public/
│   ├── .env.local                  # Frontend environment variables
│   └── package.json
│
├── docs/
│   └── API.md                      # Full API reference
│
└── README.md
```

### Key File Roles at a Glance

- **`app.py`** — The single source of truth for all HTTP routes. Every endpoint is registered here. This is the first file to open when debugging a specific API call.
- **`chat_service.py`** — Where the RAG pipeline lives. Handles vector retrieval, context assembly, prompt construction, and LLM streaming.
- **`document_processor.py`** — Called during PDF upload. Extracts text with PyMuPDF, splits into chunks, generates embeddings, stores in ChromaDB.
- **`query_classifier.py`** — Before answering, every query goes through here. The LLM decides whether this is a content search, a request to list files, a count query, or a file-specific question.
- **`llm_factory.py`** — Single place to swap LLM providers. Reads env vars and returns the appropriate LangChain LLM instance.
- **`memory_service.py`** — Reads and writes `data/memory/{chat_id}.json`. Each conversation is one file.

---

## 4. Architecture Deep Dive

### System Layers

```
┌─────────────────────────────────────────────────┐
│                  Frontend (Next.js)              │
│   Sidebar · Chat Area · Collection Manager       │
└─────────────────────┬───────────────────────────┘
                      │  HTTP + SSE
┌─────────────────────┴───────────────────────────┐
│              API Layer  (app.py)                 │
│   FastAPI routes · CORS · Pydantic validation    │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│                 Service Layer                    │
│  ChatService  ·  DocumentProcessor               │
│  CollectionManager  ·  QueryClassifier           │
│  MemoryService  ·  PDFSelectionService           │
│  MetadataService  ·  FileSearchService           │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│                  Data Layer                      │
│  ChromaDB (vectors) · JSON files (memory)        │
│  File system (original PDFs)                     │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│                  LLM Factory                     │
│  Ollama  →  Gemini  →  Groq  (priority order)   │
└─────────────────────────────────────────────────┘
```

---

### Data Flow 1: PDF Upload

When a user uploads one or more PDFs to a collection, this is the exact sequence:

```
User uploads paper.pdf
       │
       ▼
POST /api/collections/{name}/upload
       │
       ▼
DocumentProcessor.process_pdf(file)
  ├─ PyMuPDF extracts text page by page
  ├─ Headings detected from font size / bold formatting
  ├─ Text split into chunks (1000 chars, 200 overlap)
  ├─ Each chunk tagged with metadata:
  │     { filename, collection, page_numbers, title, heading }
  ├─ HuggingFace all-MiniLM-L6-v2 generates embeddings
  └─ Chunks + embeddings stored in ChromaDB collection
       │
       ▼
Original PDF saved to data/pdfs/{collection}/{filename}
       │
       ▼
Response: { files_processed, chunks_created, collection }
```

**ChromaDB storage**: Each collection in the app maps to a ChromaDB collection. Documents are stored with their vector embeddings and full metadata for retrieval and filtering.

---

### Data Flow 2: Query / Chat

When a user asks a question, every chat request goes through this pipeline:

```
User sends query: "What is the role of butyrate in gut immunity?"
       │
       ▼
GET /api/chat/single/{collection}?query=...&chat_id=...
       │
       ▼
QueryClassifier.classify(query)
  └─ LLM determines intent:
       • content_search  → proceed to vector search  ✓
       • list_pdfs       → return PDF list directly
       • count_pdfs      → return PDF count directly
       • file_specific_search → route to FileSearchService
       │
       ▼  (content_search path)
MemoryService.get_history(chat_id)
  └─ Load prior conversation turns for context
       │
       ▼
ChromaDB.similarity_search(query_embedding, top_k=10)
  └─ Returns top 10 most similar chunks from the collection
       │
       ▼
ChatService.build_prompt(query, chunks, history)
  └─ Constructs:
       [System]: "Answer only from the provided context..."
       [Context]: chunk1 + chunk2 + ... + chunk10
       [History]: prior Q&A turns
       [User]: current query
       │
       ▼
LLM.stream(prompt)
  └─ Tokens streamed via SSE:
       data: {"type": "content", "content": "Butyrate"}
       data: {"type": "content", "content": " is a short-chain"}
       ...
       data: {"type": "sources", "sources": [{...}]}
       data: {"type": "end"}
       │
       ▼
MemoryService.save_turn(chat_id, query, response)
  └─ Appended to data/memory/{chat_id}.json
```

---

### Data Flow 3: Selective PDF Mode

This mode allows querying a specific subset of PDFs across multiple collections:

```
User selects: [paper1.pdf (research), paper3.pdf (clinical)]
       │
       ▼
POST /api/selection/{session_id}/select  (called per PDF)
  └─ PDFSelectionService stores selection in memory:
       session → [ {filename, collection}, ... ]
       │
       ▼
GET /api/selection/{session_id}/chat?query=...
  └─ ChatService retrieves chunks ONLY from selected PDFs
       (ChromaDB filtered by filename metadata)
       │
       ▼
Normal LLM pipeline → SSE stream
```

---

### Query Classification Types

The `QueryClassifier` uses few-shot LLM prompting. Classification types and routing:

| Classification | Triggered When | Handler |
|---|---|---|
| `content_search` | General question | Vector similarity search → LLM |
| `list_pdfs` | "what files do you have?", "show me all PDFs" | MetadataService returns PDF list |
| `count_pdfs` | "how many papers?", "count the documents" | MetadataService returns count |
| `file_specific_search` | "what does paper.pdf say about X?" | FileSearchService filters by filename |
| `list_collections` | "what collections exist?" (ChatALL only) | CollectionManager list |

---

## 5. Setup & Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Node.js | 18.x+ |
| LLM API Key | At least one of: Google Gemini, Groq, or local Ollama |

---

### Backend Setup

```bash
# 1. Clone the repository
git clone https://github.com/KAMRANKHANALWI/papermemory.git
cd papermemory

# 2. Set up Python environment
cd backend
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements-lock.txt

# 4. Create required data directories
mkdir -p data/chroma_db data/pdfs data/memory

# 5. Configure environment variables
cp .env.example .env            # or create .env manually (see Section 8)

# 6. Start the backend
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

Verify: open `http://localhost:8000/docs` — Swagger UI should load with all endpoints.

---

### Frontend Setup

```bash
# From project root
cd frontend

# Install dependencies
npm install

# Set API URL
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start dev server
npm run dev
```

Open `http://localhost:3000` in your browser.

---

### Ollama Setup (Local LLM)

If running without API keys:

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.1

# Start Ollama server
ollama serve

# Set in .env:
USE_LOCAL_LLM=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:latest
```

---

### Quick Verification Checklist

```
□ Backend running at localhost:8000
□ /docs loads Swagger UI
□ Frontend running at localhost:3000
□ Can create a collection from the UI
□ Can upload a PDF (check terminal for "chunks_created")
□ Can ask a question and see streaming response
```

---

## 6. API Reference

**Base URL**: `http://localhost:8000`
**Interactive Docs**: `http://localhost:8000/docs`

All endpoints are prefixed with `/api/`. Chat endpoints stream Server-Sent Events (SSE). All other endpoints return standard JSON.

---

### Standard Response Envelope

```json
{
  "status": "success",
  "message": "Human-readable description",
  "data": { }
}
```

### SSE Stream Format (all chat endpoints)

```
data: {"type": "chat_id", "chat_id": "chat_abc123"}
data: {"type": "search_results", "count": 8}
data: {"type": "content", "content": "The answer is"}
data: {"type": "content", "content": " based on..."}
data: {"type": "sources", "sources": [{...}]}
data: {"type": "end"}
```

| Event Type | Description |
|---|---|
| `chat_id` | Unique ID for this conversation turn |
| `search_results` | Number of chunks retrieved from ChromaDB |
| `content` | One streamed text token |
| `sources` | List of document chunks used to generate the answer |
| `error` | Error message if something failed mid-stream |
| `end` | Signals stream is complete |

---

### 6.1 Collections API

#### `GET /api/collections` — List All Collections

Returns all collections and their PDF counts.

**Response**
```json
{
  "collections": [
    {
      "name": "gut_microbiome",
      "pdf_count": 12,
      "chunk_count": 289,
      "pdfs": ["sonnenburg_2016.pdf", "clemente_2012.pdf"]
    }
  ]
}
```

---

#### `POST /api/collections/{collection_name}/upload` — Upload PDFs

Uploads one or more PDFs into a named collection. Creates the collection if it doesn't exist.

**Request** — `multipart/form-data`

| Parameter | Type | Description |
|---|---|---|
| `collection_name` | path | Target collection name |
| `files` | form-data | One or more `.pdf` files |

**cURL Example**
```bash
curl -X POST \
  -F "files=@sonnenburg_2016.pdf" \
  -F "files=@clemente_2012.pdf" \
  http://localhost:8000/api/collections/gut_microbiome/upload
```

**Response**
```json
{
  "status": "success",
  "result": {
    "files_processed": 2,
    "chunks_created": 89,
    "collection": "gut_microbiome",
    "processed_files": ["sonnenburg_2016.pdf", "clemente_2012.pdf"]
  }
}
```

---

#### `DELETE /api/collections/{collection_name}` — Delete Collection

Deletes the collection, all its chunks from ChromaDB, and all associated PDF files from disk.

**Response**
```json
{
  "status": "success",
  "message": "Collection 'gut_microbiome' deleted",
  "pdfs_deleted": 12,
  "chunks_deleted": 289
}
```

---

#### `PUT /api/collections/rename` — Rename Collection

**Request Body**
```json
{
  "old_name": "gut_microbiome",
  "new_name": "microbiome_research"
}
```

**Response**
```json
{
  "status": "success",
  "message": "Collection renamed from 'gut_microbiome' to 'microbiome_research'"
}
```

---

#### `GET /api/collections/{collection_name}/pdfs` — List PDFs in Collection

**Response**
```json
{
  "collection_name": "gut_microbiome",
  "pdf_count": 12,
  "pdfs": [
    {
      "filename": "sonnenburg_2016.pdf",
      "chunk_count": 23,
      "title": "Diet-induced alterations in gut microflora"
    }
  ]
}
```

---

#### `DELETE /api/collections/{collection_name}/pdfs/{filename}` — Delete Single PDF

Removes the PDF's chunks from ChromaDB and deletes the file from disk.

**Response**
```json
{
  "status": "success",
  "message": "PDF 'sonnenburg_2016.pdf' deleted",
  "chunks_deleted": 23,
  "pdf_file_deleted": true
}
```

---

#### `PUT /api/collections/pdfs/rename` — Rename PDF

**Request Body**
```json
{
  "collection_name": "gut_microbiome",
  "old_filename": "sonnenburg_2016.pdf",
  "new_filename": "sonnenburg_diet_2016.pdf"
}
```

---

#### `GET /api/collections/{collection_name}/pdfs/{filename}/view` — View PDF

Returns the raw PDF file (`application/pdf`). Opens inline in browsers that support PDF rendering.

---

### 6.2 Chat API

All chat endpoints return Server-Sent Events. Consume with `EventSource` in browser or `stream=True` in Python `requests`.

#### `GET /api/chat/single/{collection_name}` — Chat with One Collection

**Query Parameters**

| Parameter | Required | Default | Description |
|---|---|---|---|
| `query` | Yes | — | The user's question |
| `chat_id` | No | auto-generated | Pass to continue a conversation |
| `num_results` | No | 10 | Number of chunks to retrieve |

**Example**
```bash
curl "http://localhost:8000/api/chat/single/gut_microbiome?query=What+role+does+fiber+play+in+microbiome+health&chat_id=chat_001"
```

**SSE Stream**
```
data: {"type": "chat_id", "chat_id": "chat_001"}
data: {"type": "search_results", "count": 10}
data: {"type": "content", "content": "Dietary fiber"}
data: {"type": "content", "content": " serves as a primary"}
data: {"type": "content", "content": " substrate for fermentation"}
...
data: {"type": "sources", "sources": [
  {
    "content": "Dietary fiber is fermented by colonic bacteria...",
    "filename": "sonnenburg_2016.pdf",
    "collection": "gut_microbiome",
    "similarity": 0.91,
    "page_numbers": "5-7",
    "title": "Diet and the Gut Microbiome"
  }
]}
data: {"type": "end"}
```

---

#### `GET /api/chat/all` — ChatALL Mode

Queries across **all** collections simultaneously. Each collection contributes its top `k_per_collection` chunks.

**Query Parameters**

| Parameter | Required | Default | Description |
|---|---|---|---|
| `query` | Yes | — | The user's question |
| `chat_id` | No | auto-generated | Conversation ID |
| `k_per_collection` | No | 1 | Chunks pulled per collection |

**Example**
```bash
curl "http://localhost:8000/api/chat/all?query=Compare+the+microbiome+of+infants+vs+adults"
```

**SSE Stream**: Same format as single collection. Sources will show `"collection"` field to identify which collection each chunk came from.

---

#### `GET /api/chat/smart/{collection_name}` — Smart (Classified) Chat

Same as single-collection chat but with automatic query classification. The system decides how to handle the query before passing it to the RAG pipeline.

**Routing behavior**:
- "What PDFs do you have?" → returns file list, no LLM call
- "How many papers are in this collection?" → returns count
- "What does sonnenburg_2016.pdf say about fiber?" → searches only that file
- "Tell me about microbiome diversity" → normal semantic search

---

### 6.3 PDF Selection API

This API manages session-based PDF selections. A session is identified by a `session_id` you generate (e.g., a UUID).

#### `POST /api/selection/{session_id}/select` — Select a PDF

**Request Body**
```json
{
  "filename": "sonnenburg_2016.pdf",
  "collection_name": "gut_microbiome"
}
```

**Response**
```json
{
  "status": "success",
  "message": "PDF selected",
  "selection": {
    "session_id": "session_abc",
    "total_selected": 3,
    "collections_involved": ["gut_microbiome", "clinical_trials"],
    "selected_pdfs": [
      {
        "filename": "sonnenburg_2016.pdf",
        "collection_name": "gut_microbiome",
        "title": "Diet and Gut Microflora",
        "pages": 12,
        "chunks": 23
      }
    ]
  }
}
```

---

#### `POST /api/selection/{session_id}/deselect` — Deselect a PDF

**Request Body**
```json
{
  "filename": "sonnenburg_2016.pdf",
  "collection_name": "gut_microbiome"
}
```

---

#### `POST /api/selection/{session_id}/batch-select` — Select Multiple PDFs

**Request Body**
```json
{
  "pdfs": [
    {"filename": "paper1.pdf", "collection_name": "gut_microbiome"},
    {"filename": "paper2.pdf", "collection_name": "clinical_trials"}
  ]
}
```

---

#### `DELETE /api/selection/{session_id}/clear` — Clear All Selections

---

#### `GET /api/selection/{session_id}/info` — Get Current Selection

**Response**
```json
{
  "session_id": "session_abc",
  "total_selected": 3,
  "collections_involved": ["gut_microbiome"],
  "selected_pdfs": [ ... ],
  "created_at": "2025-01-15T10:30:00",
  "updated_at": "2025-01-15T11:45:00"
}
```

---

#### `GET /api/selection/{session_id}/stats` — Selection Statistics

**Response**
```json
{
  "total_selected": 5,
  "collections_involved": 2,
  "pdfs_by_collection": {
    "gut_microbiome": 3,
    "clinical_trials": 2
  },
  "total_chunks": 112,
  "total_pages": 67
}
```

---

#### `GET /api/selection/{session_id}/chat` — Chat with Selected PDFs

**Query Parameters**: Same as `/api/chat/single`, with `num_results` defaulting to 25.

**Example**
```bash
curl "http://localhost:8000/api/selection/session_abc/chat?query=What+do+these+papers+say+about+Firmicutes"
```

**SSE Stream**: Same format. Retrieval is restricted to the selected PDFs only.

---

### 6.4 Memory API

#### `GET /api/memory/{chat_id}/history` — Get Conversation History

**Query Parameters**

| Parameter | Default | Description |
|---|---|---|
| `max_messages` | 10 | Number of most recent messages to return |

**Response**
```json
{
  "chat_id": "chat_001",
  "messages": [
    {
      "role": "user",
      "content": "What is the role of fiber in microbiome health?",
      "timestamp": "2025-01-15T10:30:00",
      "collection_name": "gut_microbiome"
    },
    {
      "role": "assistant",
      "content": "Dietary fiber serves as a primary substrate...",
      "timestamp": "2025-01-15T10:30:05",
      "collection_name": "gut_microbiome"
    }
  ],
  "message_count": 2,
  "created_at": "2025-01-15T10:30:00"
}
```

---

#### `DELETE /api/memory/{chat_id}/clear` — Clear Conversation Messages

Clears all messages but keeps the `chat_id` record. Useful for "new topic" without losing the session.

---

#### `DELETE /api/memory/{chat_id}` — Delete Conversation

Permanently deletes the conversation JSON file.

---

#### `GET /api/memory/{chat_id}/summary` — Conversation Summary

**Response**
```json
{
  "chat_id": "chat_001",
  "total_messages": 8,
  "user_messages": 4,
  "assistant_messages": 4,
  "collections": ["gut_microbiome"],
  "created_at": "2025-01-15T10:30:00"
}
```

---

### 6.5 Metadata & Search API

#### `POST /api/classify` — Classify Query Intent

Runs the query through the LLM classifier without executing a search. Useful for debugging routing.

**Request Body**
```json
{
  "query": "What does sonnenburg_2016.pdf say about Bacteroidetes?",
  "is_chatall_mode": false
}
```

**Response**
```json
{
  "query": "What does sonnenburg_2016.pdf say about Bacteroidetes?",
  "classification": "file_specific_search",
  "extracted_filename": "sonnenburg_2016.pdf",
  "explanation": "User is asking about a specific named file"
}
```

---

#### `POST /api/metadata/search` — Semantic Search Without LLM

Retrieves matching chunks from a collection without calling the LLM. Useful for testing retrieval quality independently.

**Request Body**
```json
{
  "query": "butyrate gut immunity",
  "collection_name": "gut_microbiome",
  "max_results": 10
}
```

**Response**
```json
{
  "query": "butyrate gut immunity",
  "collection": "gut_microbiome",
  "total_results": 7,
  "results": [
    {
      "filename": "clemente_2012.pdf",
      "content": "Butyrate produced by Firmicutes...",
      "similarity": 0.93,
      "page_numbers": "3-4",
      "title": "The Impact of the Gut Microbiota",
      "chunk_id": 15
    }
  ]
}
```

---

#### `POST /api/metadata/file-search` — Search Within a Specific File

**Request Body**
```json
{
  "query": "Bacteroidetes abundance",
  "collection_name": "gut_microbiome",
  "filename": "sonnenburg_2016.pdf",
  "max_results": 10
}
```

---

#### `GET /api/collections/{collection_name}/pdfs/{filename}/metadata` — Get PDF Metadata

**Response**
```json
{
  "filename": "sonnenburg_2016.pdf",
  "collection": "gut_microbiome",
  "total_chunks": 23,
  "headings": ["Abstract", "Introduction", "Methods", "Results", "Discussion"],
  "title": "Diet-induced alterations in gut microflora",
  "pages": 12,
  "file_size": "1.8 MB",
  "created_at": "2025-01-15T10:30:00"
}
```

---

### 6.6 Error Handling

**Error Response Format**
```json
{
  "status": "error",
  "message": "Collection 'research' not found",
  "error_code": "COLLECTION_NOT_FOUND"
}
```

**HTTP Status Codes**

| Code | Meaning | Common Cause |
|---|---|---|
| 200 | Success | — |
| 400 | Bad Request | Invalid parameters |
| 404 | Not Found | Collection or PDF does not exist |
| 422 | Validation Error | Request body failed Pydantic schema |
| 500 | Internal Server Error | LLM failure, ChromaDB error |

**Common Error Codes**

| Code | Description |
|---|---|
| `COLLECTION_NOT_FOUND` | Named collection does not exist in ChromaDB |
| `PDF_NOT_FOUND` | PDF not found in collection |
| `INVALID_FILE_TYPE` | Uploaded file is not a PDF |
| `NO_PDFS_SELECTED` | Selection chat called with empty selection |
| `LLM_ERROR` | LLM API call failed (timeout, quota, etc.) |
| `EMBEDDING_ERROR` | Embedding model failed during upload |

---

## 7. Evaluation Framework (RAGAS)

PaperMemory includes a full evaluation pipeline using [RAGAS](https://docs.ragas.io/) — a framework specifically designed to measure RAG system quality without requiring human annotations.

### Why RAGAS

Standard accuracy metrics (like exact match) don't work well for RAG because answers are generated, not selected. RAGAS measures the quality of both retrieval and generation independently, which makes it possible to diagnose *where* the pipeline is failing.

### Metrics Explained

| Metric | What It Measures | Range | Ideal |
|---|---|---|---|
| **Context Precision** | Of the chunks retrieved, how many are actually relevant? | 0–1 | High → retrieval is precise |
| **Context Recall** | Were all the chunks needed to answer the question retrieved? | 0–1 | High → nothing important was missed |
| **Faithfulness** | Is the generated answer grounded in the retrieved context? | 0–1 | High → no hallucination |
| **Answer Relevancy** | Does the answer address what was actually asked? | 0–1 | High → on-topic response |
| **Answer Correctness** | How close is the generated answer to the ground truth? | 0–1 | High → factually accurate |

A well-performing RAG system should score above **0.7** on all five metrics. Low Context Recall with high Faithfulness means the LLM is accurate but missing relevant information — a retrieval problem. Low Faithfulness with high Context Precision means good retrieval but the LLM is going off-script — a generation problem.

---

### Evaluation Scripts

All scripts are in `backend/src/eval/`.

| Script | Purpose |
|---|---|
| `eval_mcq_ragas.py` | Run MCQ dataset through RAG + compute all 5 RAGAS metrics |
| `eval_open_ended_ragas.py` | Run open-ended QA dataset + compute all 5 RAGAS metrics |
| `eval_mcq.py` | Simple MCQ accuracy (correct/incorrect), no RAGAS |
| `rag_mcq.py` | RAG pipeline wrapper for MCQ datasets (used by eval scripts) |
| `rag_open_ended.py` | RAG pipeline wrapper for open-ended datasets |

---

### Dataset Formats

**MCQ Dataset** (`datasets/your_mcq.csv`)
```csv
question,option_a,option_b,option_c,option_d,correct_answer
"Which bacteria primarily produces butyrate?","Lactobacillus","Firmicutes","E. coli","Bacteroides","b"
"What is the primary function of the gut microbiome?","Oxygen production","Digestion aid","Bone formation","Blood clotting","b"
```

**Open-Ended Dataset** (`datasets/your_open_ended.csv`)
```csv
question,ground_truth
"What role does dietary fiber play in microbiome health?","Dietary fiber is fermented by colonic bacteria, producing short-chain fatty acids like butyrate..."
"Describe the Firmicutes to Bacteroidetes ratio in obesity.","Studies have shown that obese individuals tend to have a higher Firmicutes to Bacteroidetes ratio..."
```

The project's gut microbiome evaluation used ~1,065 QA pairs generated from research PDFs. These datasets are available on HuggingFace.

---

### Running Evaluations

```bash
cd backend/src/eval

# Full RAGAS evaluation on MCQ dataset
python eval_mcq_ragas.py

# Full RAGAS evaluation on open-ended dataset
python eval_open_ended_ragas.py

# Quick accuracy check (MCQ only, no RAGAS)
python eval_mcq.py
```

Results are written to `backend/src/eval/results/` as timestamped CSV files.

**Sample output** (`results/ragas_results_20250115.csv`):
```
question, context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness
"What role does butyrate play?", 0.88, 0.79, 0.92, 0.86, 0.81
"Describe Firmicutes abundance...", 0.91, 0.83, 0.95, 0.88, 0.79
...
Average: 0.89, 0.81, 0.93, 0.87, 0.80
```

---

### Evaluation Configuration

The RAGAS evaluator requires an LLM (for computing metrics). Configure via `.env`:

```bash
# RAGAS uses Groq for evaluation LLM (fast and cheap)
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant
```

RAGAS runs as a batch process, not as part of the live API. It does not affect the production chat pipeline.

---

## 8. Configuration Reference

### Backend Environment Variables (`backend/.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| **LLM — Choose One Provider** | | | |
| `GOOGLE_API_KEY` | If using Gemini | — | Google AI Studio API key |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model name |
| `GROQ_API_KEY` | If using Groq | — | Groq API key |
| `GROQ_MODEL` | No | `llama-3.1-8b-instant` | Groq model name |
| `USE_LOCAL_LLM` | No | `false` | Set `true` to use Ollama |
| `OLLAMA_BASE_URL` | If using Ollama | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | If using Ollama | `llama3.1:latest` | Ollama model to use |
| `DEFAULT_MODEL_PROVIDER` | No | auto-detected | Force provider: `gemini`, `groq`, `ollama` |
| **Storage Paths** | | | |
| `CHROMA_DB_PATH` | No | `data/chroma_db` | ChromaDB persistent storage |
| `PDF_STORAGE_PATH` | No | `data/pdfs` | Original PDF file storage |
| `MEMORY_STORAGE_PATH` | No | `data/memory` | Conversation JSON storage |
| **Embedding** | | | |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| **RAG Parameters** | | | |
| `TOP_K` | No | `10` | Chunks retrieved per query |
| `CHUNK_SIZE` | No | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | No | `200` | Overlap between consecutive chunks |
| `TEMPERATURE` | No | `0.1` | LLM temperature (0 = deterministic) |

### Frontend Environment Variables (`frontend/.env.local`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `NEXT_PUBLIC_API_URL` | Yes | — | Backend URL, e.g. `http://localhost:8000` |

---

## 9. Developer Notes

### Adding a New LLM Provider

1. Open `backend/src/llm/llm_factory.py`
2. Add a new branch in the factory function checking for your provider's env var
3. Return a LangChain-compatible LLM instance
4. Add corresponding env vars to `.env.example` and Section 8 of this doc

### Changing the Embedding Model

The embedding model is set in `document_processor.py` and `chat_service.py`. Both must use the **same model** — if they differ, query vectors won't match stored chunk vectors.

```python
# In both files:
self.embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # example swap
)
```

After changing the model, **delete `data/chroma_db/` and re-upload all PDFs**. Old embeddings are incompatible with a new model.

### Adjusting Chunk Size

Larger chunks give the LLM more context per retrieved piece but reduce precision. Smaller chunks are more precise but may truncate mid-sentence.

```python
# In document_processor.py
def chunk_text_content(text, max_chars=1000, overlap=200):
    ...
```

Good starting points: `max_chars=1500, overlap=300` for long technical papers.

### CORS for Production

The backend allows all origins in development. For production:

```python
# In app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],  # restrict this
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

### Conversation Storage

Each conversation is stored as `data/memory/{chat_id}.json`. These files grow indefinitely. For long-running deployments, implement periodic cleanup:

```bash
# Delete conversations older than 30 days
find data/memory/ -name "*.json" -mtime +30 -delete
```

### Adding Authentication

The API has no authentication. To add API key auth, use FastAPI's dependency injection:

```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403)

# Apply to routes:
@app.get("/api/chat/single/{name}", dependencies=[Depends(verify_api_key)])
```

### Running on GPU Server (Ollama)

If running on an A100 or similar GPU server with multiple users, you can run two Ollama instances on different ports for parallel requests:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve &
OLLAMA_HOST=0.0.0.0:11435 ollama serve &
```

Set `OLLAMA_BASE_URL` to the primary instance. Load balancing across instances requires a proxy (nginx or similar).

---

## Appendix: Technology Versions

| Package | Version |
|---|---|
| Python | 3.11+ |
| FastAPI | 0.116.1 |
| LangChain | 0.3.27 |
| ChromaDB | latest (client mode) |
| PyMuPDF | latest |
| RAGAS | 0.4.2 |
| Next.js | 15.5.3 |
| React | 19 |
| Node.js | 18.x+ |

---

*Documentation written for PaperMemory.*