"""
FastAPI Application with Complete RAG Features
"""
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uuid

# Services
from src.services.document_processor import DocumentProcessor
from src.services.chat_service import ChatService
from src.services.collection_manager import CollectionManager
from src.services.query_classifier import QueryClassifier
from src.services.metadata_service import MetadataService
from src.services.file_search_service import FileSearchService
from src.services.naming_service import NamingService
from src.services.memory_service import MemoryService

# Models
from src.models import (
    QueryClassificationRequest,
    QueryClassificationResponse,
    GenerateNameRequest,
    GenerateNameResponse,
    ValidateNameRequest,
    ValidateNameResponse,
    RenameCollectionRequest,
    RenamePDFRequest,
    AddMemoryRequest,
    FileSearchRequest,
    CollectionPDFsResponse,
    AllCollectionsPDFsResponse,
    CollectionStatsResponse,
    FileSearchResponse,
    ConversationHistoryResponse,
    MemorySummaryResponse,
    OperationResponse,
    PDFDetail,
)

# Utils
from src.utils.response_generator import generate_chat_response

app = FastAPI(
    title="Enhanced Document Chat Backend",
    description="Complete RAG system with query classification, memory, and advanced features",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)

# Initialize services
document_processor = DocumentProcessor()
chat_service = ChatService()
collection_manager = CollectionManager()
metadata_service = MetadataService()
file_search_service = FileSearchService()
naming_service = NamingService(chat_service.llm, chat_service.chroma_client)
memory_service = MemoryService()
query_classifier = QueryClassifier(chat_service.llm)

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/check")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Enhanced Document Chat API",
        "version": "2.0.0"
    }

# ============================================================================
# COLLECTION ENDPOINTS
# ============================================================================

@app.get("/api/collections")
async def get_collections():
    """Get all available collections"""
    return collection_manager.get_all_collections()

@app.post("/api/collections/{collection_name}/upload")
async def upload_documents(
    collection_name: str,
    files: List[UploadFile] = File(...)
):
    """Upload PDFs to a collection"""
    try:
        result = await document_processor.process_files(files, collection_name)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection"""
    return collection_manager.delete_collection(collection_name)

@app.put("/api/collections/rename", response_model=OperationResponse)
async def rename_collection(request: RenameCollectionRequest):
    """Rename a collection"""
    result = collection_manager.rename_collection(request.old_name, request.new_name)
    return OperationResponse(**result)

@app.post("/api/collections/{collection_name}/pdfs/add")
async def add_pdfs_to_collection(
    collection_name: str,
    files: List[UploadFile] = File(...)
):
    """Add new PDFs to an existing collection"""
    try:
        result = await document_processor.process_files(files, collection_name)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/collections/{collection_name}/pdfs/{filename}")
async def delete_pdf_from_collection(collection_name: str, filename: str):
    """Delete a specific PDF from a collection"""
    result = collection_manager.delete_pdf_from_collection(collection_name, filename)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@app.put("/api/collections/pdfs/rename", response_model=OperationResponse)
async def rename_pdf_in_collection(request: RenamePDFRequest):
    """Rename a PDF within a collection"""
    result = collection_manager.rename_pdf_in_collection(
        request.collection_name,
        request.old_filename,
        request.new_filename
    )
    return OperationResponse(**result)

# ============================================================================
# METADATA ENDPOINTS
# ============================================================================

@app.get("/api/collections/{collection_name}/pdfs", response_model=CollectionPDFsResponse)
async def get_collection_pdfs(collection_name: str):
    """Get list of all PDFs in a collection with metadata"""
    try:
        # Get vectorstore for collection
        from langchain_chroma import Chroma
        
        vectorstore = Chroma(
            client=chat_service.chroma_client,
            collection_name=collection_name,
            embedding_function=chat_service.embedding_model,
            persist_directory="data/chroma_db"
        )
        
        # Get PDF list
        filenames, stats = metadata_service.get_single_collection_pdfs(vectorstore)
        
        # Format response
        pdfs = [
            PDFDetail(**pdf_detail)
            for pdf_detail in stats["pdf_details"]
        ]
        
        return CollectionPDFsResponse(
            collection_name=collection_name,
            total_pdfs=stats["total_pdfs"],
            total_chunks=stats["total_chunks"],
            pdfs=pdfs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/collections/{collection_name}/stats", response_model=CollectionStatsResponse)
async def get_collection_stats(collection_name: str):
    """Get detailed statistics for a collection"""
    try:
        from langchain_chroma import Chroma
        
        vectorstore = Chroma(
            client=chat_service.chroma_client,
            collection_name=collection_name,
            embedding_function=chat_service.embedding_model,
            persist_directory="data/chroma_db"
        )
        
        filenames, stats = metadata_service.get_single_collection_pdfs(vectorstore)
        
        pdfs = [PDFDetail(**pdf_detail) for pdf_detail in stats["pdf_details"]]
        
        return CollectionStatsResponse(
            name=collection_name,
            total_pdfs=stats["total_pdfs"],
            total_chunks=stats["total_chunks"],
            pdfs=pdfs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metadata/all-pdfs", response_model=AllCollectionsPDFsResponse)
async def get_all_pdfs():
    """Get list of all PDFs across all collections"""
    try:
        from langchain_chroma import Chroma
        
        # Load all collections
        collections = chat_service.chroma_client.list_collections()
        all_vectorstores = {}
        
        for col in collections:
            vectorstore = Chroma(
                client=chat_service.chroma_client,
                collection_name=col.name,
                embedding_function=chat_service.embedding_model,
                persist_directory="data/chroma_db"
            )
            all_vectorstores[col.name] = vectorstore
        
        # Get PDFs from all collections
        all_pdfs, stats = metadata_service.get_chatall_collection_pdfs(all_vectorstores)
        
        # Format response
        collection_responses = []
        for coll_detail in stats["collection_details"]:
            pdfs = [PDFDetail(**pdf) for pdf in coll_detail["pdfs"]]
            collection_responses.append(
                CollectionPDFsResponse(
                    collection_name=coll_detail["collection_name"],
                    total_pdfs=coll_detail["pdf_count"],
                    total_chunks=coll_detail["chunk_count"],
                    pdfs=pdfs
                )
            )
        
        return AllCollectionsPDFsResponse(
            total_collections=stats["total_collections"],
            total_pdfs=stats["total_pdfs_across_all"],
            total_chunks=stats["total_chunks_across_all"],
            collections=collection_responses
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# QUERY CLASSIFICATION ENDPOINTS
# ============================================================================

@app.post("/api/chat/classify", response_model=QueryClassificationResponse)
async def classify_query(request: QueryClassificationRequest):
    """Classify user query intent"""
    try:
        classification, filename = query_classifier.classify_query(
            request.query,
            request.is_chatall_mode
        )
        
        return QueryClassificationResponse(
            classification=classification,
            filename=filename,
            confidence=1.0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# FILE-SPECIFIC SEARCH ENDPOINTS
# ============================================================================

@app.post("/api/search/file", response_model=FileSearchResponse)
async def search_specific_file(request: FileSearchRequest):
    """Search within a specific PDF file"""
    try:
        from langchain_chroma import Chroma
        
        if not request.collection_name:
            raise HTTPException(status_code=400, detail="collection_name required for file search")
        
        vectorstore = Chroma(
            client=chat_service.chroma_client,
            collection_name=request.collection_name,
            embedding_function=chat_service.embedding_model,
            persist_directory="data/chroma_db"
        )
        
        context, results, found = file_search_service.search_specific_file(
            vectorstore,
            request.filename,
            request.query,
            request.num_results
        )
        
        return FileSearchResponse(
            filename=request.filename,
            collection_name=request.collection_name,
            found=found,
            num_results=len(results),
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/file-all", response_model=FileSearchResponse)
async def search_file_all_collections(request: FileSearchRequest):
    """Search for a specific file across all collections"""
    try:
        from langchain_chroma import Chroma
        
        # Load all collections
        collections = chat_service.chroma_client.list_collections()
        all_vectorstores = {}
        
        for col in collections:
            vectorstore = Chroma(
                client=chat_service.chroma_client,
                collection_name=col.name,
                embedding_function=chat_service.embedding_model,
                persist_directory="data/chroma_db"
            )
            all_vectorstores[col.name] = vectorstore
        
        context, results, found, collection_name = file_search_service.search_specific_file_chatall(
            all_vectorstores,
            request.filename,
            request.query,
            request.num_results
        )
        
        return FileSearchResponse(
            filename=request.filename,
            collection_name=collection_name,
            found=found,
            num_results=len(results),
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SMART NAMING ENDPOINTS
# ============================================================================

@app.post("/api/collections/generate-name", response_model=GenerateNameResponse)
async def generate_collection_name(request: GenerateNameRequest):
    """Generate a smart collection name from filenames"""
    try:
        suggested_name = naming_service.generate_smart_collection_name(
            request.filenames,
            request.upload_type
        )
        
        # Validate the generated name
        is_valid, validated_name, message = naming_service.validate_name(suggested_name)
        
        return GenerateNameResponse(
            suggested_name=validated_name if is_valid else suggested_name,
            is_valid=is_valid,
            validation_message=message if not is_valid else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/collections/validate-name", response_model=ValidateNameResponse)
async def validate_collection_name(request: ValidateNameRequest):
    """Validate a collection name"""
    try:
        is_valid, validated_name, message = naming_service.validate_name(request.name)
        
        return ValidateNameResponse(
            is_valid=is_valid,
            validated_name=validated_name if is_valid else None,
            message=message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# CONVERSATION MEMORY ENDPOINTS
# ============================================================================

@app.post("/api/memory/{chat_id}/add", response_model=OperationResponse)
async def add_message_to_memory(chat_id: str, request: AddMemoryRequest):
    """Add a message to conversation memory"""
    try:
        memory_service.add_message(
            chat_id,
            request.role,
            request.content,
            request.collection_name
        )
        
        return OperationResponse(
            status="success",
            message="Message added to memory"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/{chat_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(chat_id: str, max_messages: int = Query(10)):
    """Get conversation history"""
    memory = memory_service.get_memory(chat_id)
    
    if not memory:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    from src.models import MemoryMessage
    
    messages = [
        MemoryMessage(
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp,
            collection_name=msg.collection_name
        )
        for msg in memory.messages
    ]
    
    return ConversationHistoryResponse(
        chat_id=chat_id,
        message_count=len(messages),
        messages=messages
    )

@app.delete("/api/memory/{chat_id}", response_model=OperationResponse)
async def clear_conversation_memory(chat_id: str):
    """Clear conversation memory"""
    success = memory_service.clear_memory(chat_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return OperationResponse(
        status="success",
        message="Memory cleared"
    )

@app.get("/api/memory/{chat_id}/summary", response_model=MemorySummaryResponse)
async def get_conversation_summary(chat_id: str):
    """Get conversation summary"""
    memory = memory_service.get_memory(chat_id)
    
    if not memory:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    summary = memory_service.get_summary(chat_id)
    
    return MemorySummaryResponse(
        chat_id=chat_id,
        summary=summary or "No summary available",
        total_messages=len(memory.messages)
    )

# ============================================================================
# CHAT ENDPOINTS (Original + Enhanced)
# ============================================================================

@app.get("/api/chat/single/{collection_name}/{message}")
async def chat_single_collection(
    collection_name: str,
    message: str,
    chat_id: Optional[str] = Query(None)
):
    """Chat with a single collection"""
    return StreamingResponse(
        generate_chat_response(message, collection_name, "single", chat_id),
        media_type="text/event-stream"
    )

@app.get("/api/chat/all/{message}")
async def chat_all_collections(
    message: str,
    chat_id: Optional[str] = Query(None)
):
    """Chat with all collections (ChatALL mode)"""
    return StreamingResponse(
        generate_chat_response(message, None, "chatall", chat_id),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
