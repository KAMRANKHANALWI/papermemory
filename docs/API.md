# 📡 API Documentation

Complete API reference for Paper Memory v2.0.0

**Base URL**: `http://localhost:8000`

**Interactive Docs**: http://localhost:8000/docs (Swagger UI)

---

## Table of Contents

- [Authentication](#authentication)
- [Response Format](#response-format)
- [Collections API](#collections-api)
- [Chat API](#chat-api)
- [PDF Selection API](#pdf-selection-api)
- [Memory API](#memory-api)
- [Metadata API](#metadata-api)
- [Query Classification](#query-classification)
- [Error Handling](#error-handling)

---

## Authentication

Currently, no authentication is required. For production use, implement JWT or API key authentication.

---

## Response Format

### Standard JSON Response

```json
{
  "status": "success" | "error",
  "message": "Description",
  "data": { ... }
}
```

### Server-Sent Events (SSE)

Chat endpoints use SSE for streaming responses:

```
data: {"type": "chat_id", "chat_id": "chat_abc123"}
data: {"type": "content", "content": "The answer is"}
data: {"type": "content", "content": " machine learning"}
data: {"type": "sources", "sources": [{...}]}
data: {"type": "end"}

```

**Event Types**:
- `chat_id`: Unique conversation identifier
- `content`: Streamed text tokens
- `sources`: Retrieved document sources
- `search_results`: Number of results found
- `error`: Error message
- `end`: Stream completion

---

## Collections API

### Get All Collections

```http
GET /api/collections
```

**Response**:
```json
{
  "collections": [
    {
      "name": "research_papers",
      "pdf_count": 15,
      "chunk_count": 342,
      "pdfs": ["paper1.pdf", "paper2.pdf"]
    }
  ]
}
```

---

### Create Collection

Collections are created automatically when uploading first PDF. To pre-create:

```http
POST /api/collections/{collection_name}
```

**Parameters**:
- `collection_name` (path): Name of the collection

**Response**:
```json
{
  "status": "success",
  "message": "Collection created",
  "collection_name": "research_papers"
}
```

---

### Upload PDFs to Collection

```http
POST /api/collections/{collection_name}/upload
Content-Type: multipart/form-data
```

**Parameters**:
- `collection_name` (path): Target collection
- `files` (form-data): One or more PDF files

**Request Example**:
```bash
curl -X POST \
  -F "files=@paper1.pdf" \
  -F "files=@paper2.pdf" \
  http://localhost:8000/api/collections/research/upload
```

**Response**:
```json
{
  "status": "success",
  "result": {
    "files_processed": 2,
    "chunks_created": 89,
    "collection": "research",
    "processed_files": ["paper1.pdf", "paper2.pdf"]
  }
}
```

---

### Delete Collection

```http
DELETE /api/collections/{collection_name}
```

**Parameters**:
- `collection_name` (path): Collection to delete

**Response**:
```json
{
  "status": "success",
  "message": "Collection 'research' deleted",
  "pdfs_deleted": 15,
  "chunks_deleted": 342
}
```

---

### Rename Collection

```http
PUT /api/collections/rename
Content-Type: application/json
```

**Request Body**:
```json
{
  "old_name": "old_collection",
  "new_name": "new_collection"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Collection renamed from 'old_collection' to 'new_collection'"
}
```

---

### Get Collection PDFs

```http
GET /api/collections/{collection_name}/pdfs
```

**Response**:
```json
{
  "collection_name": "research",
  "pdf_count": 15,
  "pdfs": [
    {
      "filename": "paper1.pdf",
      "chunk_count": 23,
      "title": "Introduction to Machine Learning"
    }
  ]
}
```

---

### Delete PDF from Collection

```http
DELETE /api/collections/{collection_name}/pdfs/{filename}
```

**Parameters**:
- `collection_name` (path): Collection name
- `filename` (path): PDF filename (URL-encoded)

**Response**:
```json
{
  "status": "success",
  "message": "PDF 'paper1.pdf' deleted",
  "chunks_deleted": 23,
  "pdf_file_deleted": true
}
```

---

### Rename PDF

```http
PUT /api/collections/pdfs/rename
Content-Type: application/json
```

**Request Body**:
```json
{
  "collection_name": "research",
  "old_filename": "old_name.pdf",
  "new_filename": "new_name.pdf"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "PDF renamed from 'old_name.pdf' to 'new_name.pdf'"
}
```

---

### View PDF

```http
GET /api/collections/{collection_name}/pdfs/{filename}/view
```

**Parameters**:
- `collection_name` (path): Collection name
- `filename` (path): PDF filename (URL-encoded)

**Response**: PDF file (application/pdf)

Opens PDF in browser or downloads depending on browser settings.

---

## Chat API

### Chat with Single Collection

```http
GET /api/chat/single/{collection_name}
```

**Parameters**:
- `collection_name` (path): Collection to search
- `query` (query): User's question
- `chat_id` (query, optional): Conversation ID for context
- `num_results` (query, optional): Number of chunks (default: 10)

**Request Example**:
```bash
curl "http://localhost:8000/api/chat/single/research?query=What%20is%20ML&chat_id=chat_123"
```

**Response** (Server-Sent Events):
```
data: {"type": "chat_id", "chat_id": "chat_123"}

data: {"type": "search_results", "count": 8}

data: {"type": "content", "content": "Machine"}

data: {"type": "content", "content": " learning"}

data: {"type": "content", "content": " is a subset"}

data: {"type": "sources", "sources": [
  {
    "content": "Machine learning is...",
    "filename": "ml_paper.pdf",
    "collection": "research",
    "similarity": 0.92,
    "page_numbers": "5-7",
    "title": "Introduction to ML"
  }
]}

data: {"type": "end"}

```

---

### Chat Across All Collections (ChatALL)

```http
GET /api/chat/all
```

**Parameters**:
- `query` (query): User's question
- `chat_id` (query, optional): Conversation ID
- `k_per_collection` (query, optional): Chunks per collection (default: 1)

**Request Example**:
```bash
curl "http://localhost:8000/api/chat/all?query=Compare%20approaches"
```

**Response**: Same SSE format as single collection, but aggregates results from all collections.

---

### Classify and Chat

```http
GET /api/chat/smart/{collection_name}
```

Automatically classifies query intent and routes to appropriate handler.

**Parameters**: Same as single collection chat

**Response**: Same SSE format with automatic routing:
- `list_pdfs` → Returns PDF list
- `count_pdfs` → Returns count
- `file_specific_search` → Searches specific file
- `content_search` → Normal semantic search

---

## PDF Selection API

Allows users to select specific PDFs across collections for targeted querying.

### Select PDF

```http
POST /api/selection/{session_id}/select
Content-Type: application/json
```

**Parameters**:
- `session_id` (path): Unique session identifier

**Request Body**:
```json
{
  "filename": "paper1.pdf",
  "collection_name": "research"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "PDF selected",
  "selection": {
    "session_id": "session_123",
    "total_selected": 5,
    "collections_involved": ["research", "technical"],
    "selected_pdfs": [
      {
        "filename": "paper1.pdf",
        "collection_name": "research",
        "title": "ML Paper",
        "pages": 12,
        "chunks": 23
      }
    ]
  }
}
```

---

### Deselect PDF

```http
POST /api/selection/{session_id}/deselect
Content-Type: application/json
```

**Request Body**:
```json
{
  "filename": "paper1.pdf",
  "collection_name": "research"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "PDF deselected"
}
```

---

### Batch Select PDFs

```http
POST /api/selection/{session_id}/batch-select
Content-Type: application/json
```

**Request Body**:
```json
{
  "pdfs": [
    {"filename": "paper1.pdf", "collection_name": "research"},
    {"filename": "paper2.pdf", "collection_name": "technical"}
  ]
}
```

**Response**:
```json
{
  "status": "success",
  "message": "2 PDFs selected",
  "total_selected": 7
}
```

---

### Clear All Selections

```http
DELETE /api/selection/{session_id}/clear
```

**Response**:
```json
{
  "status": "success",
  "message": "All selections cleared"
}
```

---

### Get Selection Info

```http
GET /api/selection/{session_id}/info
```

**Response**:
```json
{
  "session_id": "session_123",
  "total_selected": 5,
  "collections_involved": ["research", "technical"],
  "selected_pdfs": [
    {
      "filename": "paper1.pdf",
      "collection_name": "research",
      "title": "ML Introduction",
      "pages": 12,
      "chunks": 23
    }
  ],
  "created_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T11:45:00"
}
```

---

### Get Selection Statistics

```http
GET /api/selection/{session_id}/stats
```

**Response**:
```json
{
  "total_selected": 8,
  "collections_involved": 3,
  "pdfs_by_collection": {
    "research": 5,
    "technical": 2,
    "tutorials": 1
  },
  "total_chunks": 234,
  "total_pages": 156
}
```

---

### Search Within Selected PDFs

```http
POST /api/selection/{session_id}/search
Content-Type: application/json
```

**Request Body**:
```json
{
  "query": "machine learning algorithms",
  "num_results": 25
}
```

**Response**:
```json
{
  "query": "machine learning algorithms",
  "total_results": 18,
  "total_selected_pdfs": 8,
  "collections_searched": 3,
  "results": [
    {
      "filename": "ml_paper.pdf",
      "collection": "research",
      "content": "Machine learning algorithms include...",
      "similarity": 0.94,
      "page_numbers": "7-9",
      "title": "ML Fundamentals"
    }
  ]
}
```

---

### Chat with Selected PDFs

```http
GET /api/selection/{session_id}/chat
```

**Parameters**:
- `session_id` (path): Session identifier
- `query` (query): User's question
- `chat_id` (query, optional): Conversation ID
- `num_results` (query, optional): Number of chunks (default: 25)

**Request Example**:
```bash
curl "http://localhost:8000/api/selection/session_123/chat?query=Compare%20papers"
```

**Response** (SSE): Same format as regular chat, but only searches selected PDFs.

---

## Memory API

Manages conversation history and context.

### Get Conversation History

```http
GET /api/memory/{chat_id}/history
```

**Parameters**:
- `chat_id` (path): Conversation identifier
- `max_messages` (query, optional): Limit messages (default: 10)

**Response**:
```json
{
  "chat_id": "chat_123",
  "messages": [
    {
      "role": "user",
      "content": "What is machine learning?",
      "timestamp": "2024-01-15T10:30:00",
      "collection_name": "research"
    },
    {
      "role": "assistant",
      "content": "Machine learning is...",
      "timestamp": "2024-01-15T10:30:05",
      "collection_name": "research"
    }
  ],
  "message_count": 2,
  "created_at": "2024-01-15T10:30:00"
}
```

---

### Clear Conversation

```http
DELETE /api/memory/{chat_id}/clear
```

Clears all messages but keeps the conversation record.

**Response**:
```json
{
  "status": "success",
  "message": "Conversation cleared",
  "chat_id": "chat_123"
}
```

---

### Delete Conversation

```http
DELETE /api/memory/{chat_id}
```

Permanently deletes the entire conversation.

**Response**:
```json
{
  "status": "success",
  "message": "Conversation deleted",
  "chat_id": "chat_123"
}
```

---

### Get Conversation Summary

```http
GET /api/memory/{chat_id}/summary
```

**Response**:
```json
{
  "chat_id": "chat_123",
  "summary": "Conversation with 10 messages (5 from user, 5 from assistant). Collections discussed: research, technical",
  "total_messages": 10,
  "user_messages": 5,
  "assistant_messages": 5,
  "collections": ["research", "technical"],
  "created_at": "2024-01-15T10:30:00"
}
```

---

### Add Memory Entry

```http
POST /api/memory/add
Content-Type: application/json
```

**Request Body**:
```json
{
  "chat_id": "chat_123",
  "content": "Remember to focus on supervised learning",
  "collection_name": "research"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Memory added",
  "chat_id": "chat_123"
}
```

---

## Metadata API

### Get Collection Statistics

```http
GET /api/collections/{collection_name}/stats
```

**Response**:
```json
{
  "collection_name": "research",
  "pdf_count": 15,
  "total_chunks": 342,
  "total_pages": 187,
  "created_at": "2024-01-10",
  "last_updated": "2024-01-15"
}
```

---

### Get All Collections PDFs

```http
GET /api/metadata/all-pdfs
```

Returns PDFs from all collections.

**Response**:
```json
{
  "total_collections": 3,
  "total_pdfs": 45,
  "collections": {
    "research": {
      "pdf_count": 15,
      "pdfs": ["paper1.pdf", "paper2.pdf"]
    },
    "technical": {
      "pdf_count": 20,
      "pdfs": ["doc1.pdf", "doc2.pdf"]
    }
  }
}
```

---

### Get PDF Metadata

```http
GET /api/collections/{collection_name}/pdfs/{filename}/metadata
```

**Response**:
```json
{
  "filename": "paper1.pdf",
  "collection": "research",
  "total_chunks": 23,
  "headings": ["Introduction", "Methods", "Results", "Discussion"],
  "title": "Machine Learning Fundamentals",
  "pages": 12,
  "file_size": "2.4 MB",
  "created_at": "2024-01-15T10:30:00"
}
```

---

### Search Within Collection

```http
POST /api/metadata/search
Content-Type: application/json
```

**Request Body**:
```json
{
  "query": "machine learning algorithms",
  "collection_name": "research",
  "max_results": 10
}
```

**Response**:
```json
{
  "query": "machine learning algorithms",
  "collection": "research",
  "total_results": 25,
  "results": [
    {
      "filename": "ml_paper.pdf",
      "content": "Machine learning algorithms...",
      "similarity": 0.92,
      "page_numbers": "5-7",
      "title": "ML Fundamentals",
      "chunk_id": 5
    }
  ]
}
```

---

### File-Specific Search

```http
POST /api/metadata/file-search
Content-Type: application/json
```

Search within a specific PDF file.

**Request Body**:
```json
{
  "query": "supervised learning",
  "collection_name": "research",
  "filename": "ml_paper.pdf",
  "max_results": 10
}
```

**Response**:
```json
{
  "query": "supervised learning",
  "filename": "ml_paper.pdf",
  "collection": "research",
  "total_results": 8,
  "results": [
    {
      "content": "Supervised learning involves...",
      "similarity": 0.95,
      "page_numbers": "12-14",
      "chunk_id": 12
    }
  ]
}
```

---

## Query Classification

### Classify Query

```http
POST /api/classify
Content-Type: application/json
```

Uses LLM to classify user query intent.

**Request Body**:
```json
{
  "query": "What does paper.pdf say about results?",
  "is_chatall_mode": false
}
```

**Response**:
```json
{
  "query": "What does paper.pdf say about results?",
  "classification": "file_specific_search",
  "extracted_filename": "paper.pdf",
  "explanation": "User is asking about a specific file",
  "is_chatall_mode": false
}
```

**Classification Types**:
- `list_pdfs`: User wants to see available documents
- `count_pdfs`: User wants count of documents
- `list_collections`: User wants to see collections (ChatALL only)
- `file_specific_search`: Query targets specific PDF file
- `content_search`: Normal semantic search (default)

---

### Validate Collection Name

```http
POST /api/validate/name
Content-Type: application/json
```

**Request Body**:
```json
{
  "name": "new_collection"
}
```

**Response**:
```json
{
  "valid": true,
  "message": "Name is valid",
  "sanitized_name": "new_collection"
}
```

**Validation Rules**:
- Alphanumeric, underscores, hyphens only
- 1-50 characters
- No spaces or special characters

---

## Error Handling

### Error Response Format

```json
{
  "status": "error",
  "message": "Descriptive error message",
  "error_code": "ERROR_CODE",
  "details": {
    "field": "additional context"
  }
}
```

### HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid input parameters |
| 404 | Not Found | Collection/PDF not found |
| 422 | Validation Error | Pydantic validation failed |
| 500 | Internal Error | Server error during processing |

### Common Error Codes

**COLLECTION_NOT_FOUND**
```json
{
  "status": "error",
  "message": "Collection 'research' not found",
  "error_code": "COLLECTION_NOT_FOUND"
}
```

**PDF_NOT_FOUND**
```json
{
  "status": "error",
  "message": "PDF 'paper.pdf' not found in collection 'research'",
  "error_code": "PDF_NOT_FOUND"
}
```

**INVALID_FILE_TYPE**
```json
{
  "status": "error",
  "message": "Only PDF files are supported",
  "error_code": "INVALID_FILE_TYPE"
}
```

**NO_PDFS_SELECTED**
```json
{
  "status": "error",
  "message": "No PDFs selected. Please select PDFs first.",
  "error_code": "NO_PDFS_SELECTED"
}
```

**LLM_ERROR**
```json
{
  "status": "error",
  "message": "LLM generation failed: Connection timeout",
  "error_code": "LLM_ERROR"
}
```

---

## Rate Limiting

Currently no rate limiting is implemented. For production:

```python
# Recommended: 100 requests per minute per IP
# Use FastAPI-Limiter or similar middleware
```

---

## CORS Configuration

Default CORS settings (development):

```python
allow_origins=["*"]
allow_credentials=True
allow_methods=["*"]
allow_headers=["*"]
```

**Production**: Restrict `allow_origins` to your frontend domain.

---

## Pagination

For endpoints returning large lists:

**Parameters**:
- `page` (query): Page number (default: 1)
- `page_size` (query): Items per page (default: 20, max: 100)

**Response includes**:
```json
{
  "items": [...],
  "total": 150,
  "page": 1,
  "page_size": 20,
  "total_pages": 8
}
```

---

## Webhooks (Future)

Planned webhook support for:
- Document processing completion
- Collection updates
- Memory events

**Configuration**:
```json
{
  "webhook_url": "https://your-app.com/webhook",
  "events": ["document.processed", "collection.updated"]
}
```

---

## SDK Support (Future)

Planned SDKs:
- **Python**: `pip install papermemory-sdk`
- **TypeScript**: `npm install @papermemory/sdk`
- **Go**: `go get github.com/papermemory/go-sdk`

**Example Usage**:
```python
from papermemory import PaperMemory

client = PaperMemory(base_url="http://localhost:8000")
response = client.chat.single("research", "What is ML?")
```

---

## Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8000/

# Upload PDF
curl -X POST -F "files=@paper.pdf" \
  http://localhost:8000/api/collections/research/upload

# Chat
curl "http://localhost:8000/api/chat/single/research?query=What%20is%20ML"

# Get collections
curl http://localhost:8000/api/collections
```

### Using Python

```python
import requests

# Upload PDF
with open("paper.pdf", "rb") as f:
    files = {"files": f}
    response = requests.post(
        "http://localhost:8000/api/collections/research/upload",
        files=files
    )
print(response.json())

# Chat
response = requests.get(
    "http://localhost:8000/api/chat/single/research",
    params={"query": "What is ML?"},
    stream=True
)
for line in response.iter_lines():
    if line:
        print(line.decode())
```

### Using JavaScript

```javascript
// Upload PDF
const formData = new FormData();
formData.append('files', fileInput.files[0]);

await fetch('http://localhost:8000/api/collections/research/upload', {
  method: 'POST',
  body: formData
});

// Chat with SSE
const eventSource = new EventSource(
  'http://localhost:8000/api/chat/single/research?query=What+is+ML'
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```




