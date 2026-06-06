# System Architecture

This document describes the architecture of GutMiScholar and the flow of data from document ingestion to answer generation.

---

# Overview

GutMiScholar is a literature-grounded question answering system for scientific PDFs.

The system allows users to upload research papers, organize them into collections, retrieve relevant passages through semantic search, and generate answers grounded in retrieved literature.

The architecture is organized into independent layers responsible for:

* User interaction
* Request routing
* Query orchestration
* Document retrieval
* Reranking
* Language model inference
* Source attribution

---

# Architectural Principles

The system is designed around the following principles:

* Retrieval before generation
* Collection-oriented document organization
* Configurable model providers
* Local-first deployment support
* Transparent source attribution
* Separation of orchestration and business logic
* Streaming responses

---

# High-Level Architecture

```mermaid
flowchart LR

    User

    Frontend["Next.js Frontend"]

    Router["FastAPI Routers"]

    Controller["Controllers"]

    Orchestrator["Chat Orchestrator"]

    Classifier["Query Classifier"]

    Retrieval["Retrieval Service"]

    Metadata["Metadata Service"]

    FileSearch["File Search Service"]

    Selection["PDF Selection Service"]

    Memory["Memory Service"]

    Reranker["Reranker"]

    Chroma["ChromaDB"]

    LLM["Ollama / Gemini / Groq"]

    User --> Frontend

    Frontend --> Router

    Router --> Controller

    Controller --> Orchestrator

    Orchestrator --> Classifier

    Classifier --> Retrieval
    Classifier --> Metadata
    Classifier --> FileSearch
    Classifier --> Selection

    Retrieval --> Chroma
    FileSearch --> Chroma
    Selection --> Chroma

    Retrieval --> Reranker
    FileSearch --> Reranker
    Selection --> Reranker

    Reranker --> LLM

    Orchestrator --> Memory
```

---

# Frontend Architecture

The frontend is implemented using Next.js, React, TypeScript, and Tailwind CSS.

Responsibilities include:

* Collection management
* PDF upload workflows
* PDF selection workflows
* Chat interface
* Source display
* Streaming response rendering
* Conversation export

The frontend communicates with the backend through REST APIs and Server-Sent Events (SSE).

---

# Backend Architecture

The backend is implemented using FastAPI.

The backend is divided into four primary layers:

```text
Routers
    ↓
Controllers
    ↓
Orchestrators
    ↓
Services
```

## Routers

Routers expose HTTP endpoints and validate incoming requests.

Examples:

* chat.py
* collections.py
* search.py
* metadata.py
* selection.py

Routers do not contain retrieval or generation logic.

---

## Controllers

Controllers handle request processing and response formatting.

Examples:

* chat_controller.py
* selection_controller.py

Controllers delegate business logic to orchestrators.

---

## Orchestrators

Orchestrators coordinate complex workflows involving multiple services.

Current orchestrator:

* chat_orchestrator.py

Responsibilities:

* Query classification
* Retrieval workflow selection
* Prompt construction
* Source preparation
* Memory integration
* Streaming response coordination

---

## Services

Services contain the core business logic.

Examples:

* retrieval_service.py
* metadata_service.py
* memory_service.py
* pdf_selection_service.py
* collection_manager.py
* document_processor.py
* file_search_service.py

Services remain independent and reusable.

---

# Document Ingestion Pipeline

Documents must be processed before they become searchable.

```mermaid
flowchart LR

    PDF[PDF File]

    Storage[PDF Storage]

    Parser[PDF Parser]

    Chunking[Text Chunking]

    Embedding[Embedding Model]

    Chroma[ChromaDB]

    PDF --> Storage

    Storage --> Parser

    Parser --> Chunking

    Chunking --> Embedding

    Embedding --> Chroma
```

---

## PDF Storage

Uploaded documents are stored under the configured data directory.

The storage layer manages:

* File persistence
* Collection organization
* Upload validation

---

## PDF Parsing

PDF content is extracted using PyMuPDF.

Extracted metadata includes:

* Filename
* Page numbers
* Document title
* Collection name

---

## Chunking

Documents are divided into overlapping chunks before embedding.

Default configuration:

```text
Chunk Size: 1000
Chunk Overlap: 200
```

Chunking improves retrieval granularity and recall.

---

## Embeddings

Text chunks are converted into vector representations using Sentence Transformers.

Default model:

```text
all-MiniLM-L6-v2
```

---

## Vector Storage

Embedded chunks are stored in ChromaDB.

Each collection maintains its own vector index and associated metadata.

---

# Query Processing Pipeline

Every user query follows the workflow below.

```mermaid
flowchart LR

    Query[User Query]

    Classification[Query Classification]

    Retrieval[Retrieval]

    Reranking[Reranking]

    Prompt[Prompt Construction]

    Generation[Answer Generation]

    Response[Response + Sources]

    Query --> Classification

    Classification --> Retrieval

    Retrieval --> Reranking

    Reranking --> Prompt

    Prompt --> Generation

    Generation --> Response
```

---

# Query Classification Layer

Before retrieval begins, the system classifies user intent.

Supported classifications include:

* Content search
* File-specific search
* Collection listing
* PDF listing
* PDF counting

Classification determines which retrieval workflow is executed.

This allows the system to route requests without requiring separate user commands.

---

# Retrieval Layer

The retrieval layer performs semantic search against ChromaDB.

Retrieved chunks contain:

* Content
* Filename
* Collection
* Page numbers
* Similarity score

The retrieval strategy depends on the active search mode.

---

# Reranking Layer

Initial vector retrieval prioritizes recall.

A reranking stage is applied before answer generation to improve relevance.

Supported reranking strategies:

* Similarity-only ranking
* Cross-encoder reranking
* Ollama-based LLM reranking
* Gemini-based LLM reranking
* Groq-based LLM reranking

The active strategy is selected through configuration.

---

# LLM Layer

GutMiScholar supports multiple inference providers.

## Ollama

Local inference.

Suitable for:

* Private deployments
* Sensitive documents
* Offline usage

Document content remains within the local environment.

---

## Groq

Cloud-hosted inference focused on low-latency generation.

---

## Gemini

Cloud-hosted inference through Google Gemini models.

---

## Provider Selection

The active provider is selected through configuration.

Switching providers does not require application code changes.

---

# Conversation Memory

The memory layer stores conversation history used during answer generation.

Responsibilities include:

* Conversation context preservation
* Multi-turn interaction support
* Prompt enrichment

Memory is managed independently from document retrieval.

---

# Retrieval Modes

GutMiScholar supports multiple retrieval scopes.

---

## Single Collection

Searches only the active collection.

```text
User Query
    ↓
Selected Collection
    ↓
Retrieval
    ↓
Answer
```

This mode provides the most focused retrieval scope.

---

## All Collections

Searches across every available collection.

```text
User Query
    ↓
Collection A
Collection B
Collection C
    ↓
Merged Results
    ↓
Reranking
    ↓
Answer
```

---

## Selected PDFs

Searches only user-selected PDFs.

```text
User Query
    ↓
Selected PDFs
    ↓
Retrieval
    ↓
Answer
```

This mode allows targeted exploration of specific documents.

---

## File-Specific Search

Searches within a single document identified by filename.

```text
User Query
    ↓
Specific PDF
    ↓
Retrieval
    ↓
Answer
```

This mode is automatically activated when a query references a specific PDF.

---

# Source Attribution

Generated answers include references to retrieved passages.

Source attribution enables users to:

* Verify generated responses
* Inspect supporting evidence
* Trace information back to source documents

The system is designed to keep generated responses connected to retrieved literature.

---

# Response Streaming

Responses are delivered using Server-Sent Events (SSE).

Benefits include:

* Progressive answer rendering
* Reduced perceived latency
* Continuous response updates

The frontend displays generated content as it becomes available.

---

# Deployment Architecture

The application can be deployed locally or through Docker.

```mermaid
flowchart LR

    Browser

    Frontend["Next.js"]

    Backend["FastAPI"]

    Chroma["ChromaDB"]

    Ollama["Ollama (Optional)"]

    Gemini["Gemini API"]

    Groq["Groq API"]

    Browser --> Frontend

    Frontend --> Backend

    Backend --> Chroma

    Backend --> Ollama

    Backend --> Gemini

    Backend --> Groq
```

Docker Compose orchestrates frontend and backend services.

---

# Configuration

The application is configured through environment variables.

Examples include:

```text
USE_LOCAL_LLM
DEFAULT_MODEL_PROVIDER

OLLAMA_MODEL
GROQ_MODEL
GEMINI_MODEL

RERANKER_TYPE

CHUNK_SIZE
CHUNK_OVERLAP

TOP_K
```

Configuration changes do not require application code modifications.

---

# Design Goals

The architecture was designed around the following goals:

* Literature-grounded responses
* Collection-based retrieval
* User-controlled retrieval scope
* Configurable inference providers
* Local-first deployment support
* Transparent source attribution
* Streaming response delivery
* Modular retrieval and reranking pipelines

These goals guided the architectural decisions throughout the system.
