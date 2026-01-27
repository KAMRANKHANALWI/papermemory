"""
FastAPI Application with Complete RAG Features
"""

# from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Body, APIRouter
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uuid
import logging
import time
import json
from fastapi.responses import Response, FileResponse
import mimetypes


# Services
from src.services.document_processor import DocumentProcessor
from src.services.chat_service import ChatService
from src.services.collection_manager import CollectionManager
from src.services.query_classifier import QueryClassifier
from src.services.metadata_service import MetadataService
from src.services.file_search_service import FileSearchService
from src.services.naming_service import NamingService
from src.services.memory_service import MemoryService
from src.utils.response_generator import generate_chat_response
from src.services.pdf_storage_service import PDFStorageService

# New Custom Select
from src.services.pdf_selection_service import PDFSelectionService, SelectedPDF
from src.models.selection_models import (
    SelectPDFRequest,
    DeselectPDFRequest,
    BatchSelectPDFsRequest,
    SelectedPDFsSearchRequest,
    PDFSelectionResponse,
    SelectionSessionResponse,
    SelectedPDFsSearchResponse,
    SelectionStatsResponse,
    SelectedPDFInfo,
)

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


app = FastAPI(
    title="Enhanced Document Chat Backend",
    description="Complete RAG system with query classification, memory, and advanced features",
    version="2.0.0",
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
# document_processor = DocumentProcessor()
chat_service = ChatService()
collection_manager = CollectionManager()
metadata_service = MetadataService()
file_search_service = FileSearchService()
naming_service = NamingService(chat_service.llm, chat_service.chroma_client)
memory_service = MemoryService()
query_classifier = QueryClassifier(chat_service.llm)
# Initialize PDF Selection Service
pdf_selection_service = PDFSelectionService()
# New Local
router = APIRouter()
chat_service = ChatService()
# PDF Storage Manager - for storing/serving original PDFs
pdf_storage = PDFStorageService(base_path="data/pdfs")
document_processor = DocumentProcessor(pdf_storage=pdf_storage)


# Local api stuff check:
@router.get("/api/model-info")
async def get_model_info():
    """
    Get information about the currently configured LLM.

    Returns:
        Dict with model provider, model name, and whether it's local
    """
    try:
        model_info = chat_service.get_model_info()
        return {"status": "success", "model_info": model_info}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# HEALTH CHECK
# ============================================================================


@app.get("/check")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Enhanced Document Chat API", "version": "2.0.0"}


# ============================================================================
# COLLECTION ENDPOINTS
# ============================================================================


@app.get("/api/collections")
async def get_collections():
    """Get all available collections"""
    return collection_manager.get_all_collections()


@app.post("/api/collections/{collection_name}/upload")
async def upload_documents(collection_name: str, files: List[UploadFile] = File(...)):
    """Upload PDFs to a collection"""
    try:
        result = await document_processor.process_files(files, collection_name)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.delete("/api/collections/{collection_name}")
# async def delete_collection(collection_name: str):
#     """Delete a collection"""
#     return collection_manager.delete_collection(collection_name)


@app.delete("/api/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete an entire collection"""

    # Delete from vector database
    result = collection_manager.delete_collection(collection_name)

    # ✅ NEW: Also delete all PDF files
    pdf_storage.delete_collection_pdfs(collection_name)

    return result


@app.put("/api/collections/rename", response_model=OperationResponse)
async def rename_collection(request: RenameCollectionRequest):
    """Rename a collection"""
    result = collection_manager.rename_collection(request.old_name, request.new_name)
    return OperationResponse(**result)


@app.post("/api/collections/{collection_name}/pdfs/add")
async def add_pdfs_to_collection(
    collection_name: str, files: List[UploadFile] = File(...)
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

    # Delete from vector database
    result = collection_manager.delete_pdf_from_collection(collection_name, filename)

    # Also delete the stored PDF file
    pdf_deleted = pdf_storage.delete_pdf(collection_name, filename)

    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])

    return {**result, "pdf_file_deleted": pdf_deleted}


@app.put("/api/collections/pdfs/rename", response_model=OperationResponse)
async def rename_pdf_in_collection(request: RenamePDFRequest):
    """Rename a PDF within a collection"""

    # Rename in vector database
    result = collection_manager.rename_pdf_in_collection(
        request.collection_name, request.old_filename, request.new_filename
    )

    # Also rename the stored PDF file
    pdf_renamed = pdf_storage.rename_pdf(
        request.collection_name, request.old_filename, request.new_filename
    )

    return OperationResponse(**result)


# "VIEW PDF" ENDPOINT:


@app.get("/api/collections/{collection_name}/pdfs/{filename}/view")
async def view_pdf(collection_name: str, filename: str):
    """
    Serve PDF file for viewing in browser.
    Opens PDF in a new tab.
    """
    # Get PDF path
    pdf_path = pdf_storage.get_pdf_path(collection_name, filename)

    if not pdf_path:
        raise HTTPException(
            status_code=404,
            detail=f"PDF '{filename}' not found in collection '{collection_name}'",
        )

    # Return PDF file
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f"inline; filename={filename}"},
    )


# ============================================================================
# METADATA ENDPOINTS
# ============================================================================


@app.get(
    "/api/collections/{collection_name}/pdfs", response_model=CollectionPDFsResponse
)
async def get_collection_pdfs(collection_name: str):
    """Get list of all PDFs in a collection with metadata"""
    try:
        # Get vectorstore for collection
        from langchain_chroma import Chroma

        vectorstore = Chroma(
            client=chat_service.chroma_client,
            collection_name=collection_name,
            embedding_function=chat_service.embedding_model,
            persist_directory="data/chroma_db",
        )

        # Get PDF list
        filenames, stats = metadata_service.get_single_collection_pdfs(vectorstore)

        # Format response
        pdfs = [PDFDetail(**pdf_detail) for pdf_detail in stats["pdf_details"]]

        return CollectionPDFsResponse(
            collection_name=collection_name,
            total_pdfs=stats["total_pdfs"],
            total_chunks=stats["total_chunks"],
            pdfs=pdfs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/collections/{collection_name}/stats", response_model=CollectionStatsResponse
)
async def get_collection_stats(collection_name: str):
    """Get detailed statistics for a collection"""
    try:
        from langchain_chroma import Chroma

        vectorstore = Chroma(
            client=chat_service.chroma_client,
            collection_name=collection_name,
            embedding_function=chat_service.embedding_model,
            persist_directory="data/chroma_db",
        )

        filenames, stats = metadata_service.get_single_collection_pdfs(vectorstore)

        pdfs = [PDFDetail(**pdf_detail) for pdf_detail in stats["pdf_details"]]

        return CollectionStatsResponse(
            name=collection_name,
            total_pdfs=stats["total_pdfs"],
            total_chunks=stats["total_chunks"],
            pdfs=pdfs,
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
                persist_directory="data/chroma_db",
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
                    pdfs=pdfs,
                )
            )

        return AllCollectionsPDFsResponse(
            total_collections=stats["total_collections"],
            total_pdfs=stats["total_pdfs_across_all"],
            total_chunks=stats["total_chunks_across_all"],
            collections=collection_responses,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# QUERY CLASSIFICATION ENDPOINTS
# ============================================================================

# @app.post("/api/chat/classify", response_model=QueryClassificationResponse)
# async def classify_query(request: QueryClassificationRequest):
#     """Classify user query intent"""
#     try:
#         classification, filename = query_classifier.classify_query(
#             request.query,
#             request.is_chatall_mode
#         )

#         return QueryClassificationResponse(
#             classification=classification,
#             filename=filename,
#             confidence=1.0
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

logger = logging.getLogger(__name__)


@app.post("/api/chat/classify", response_model=QueryClassificationResponse)
async def classify_query(request: QueryClassificationRequest):
    """
    Classify user query intent with detailed logging

    Args:
        request: QueryClassificationRequest with query and is_chatall_mode

    Returns:
        QueryClassificationResponse with classification, filename, and confidence
    """
    request_start = time.time()

    logger.info("╔" + "═" * 78 + "╗")
    logger.info(f"║ 📥 CLASSIFICATION REQUEST")
    logger.info(
        f"║ Query: {request.query[:60]}" + ("..." if len(request.query) > 60 else "")
    )
    logger.info(f"║ ChatALL Mode: {request.is_chatall_mode}")
    logger.info("╚" + "═" * 78 + "╝")

    try:
        # Classify query using LLM
        classification, filename = query_classifier.classify_query(
            request.query, request.is_chatall_mode
        )

        # Calculate total time
        total_time = (time.time() - request_start) * 1000

        # Log results
        logger.info("╔" + "═" * 78 + "╗")
        logger.info(f"║ 📤 CLASSIFICATION RESPONSE")
        logger.info(f"║ Classification: {classification}")
        logger.info(f"║ Filename: {filename or 'None'}")
        logger.info(f"║ Total Time: {total_time:.2f}ms")
        logger.info("╚" + "═" * 78 + "╝")

        return QueryClassificationResponse(
            classification=classification, filename=filename, confidence=1.0
        )

    except Exception as e:
        elapsed = (time.time() - request_start) * 1000
        logger.error(
            f"❌ Classification failed after {elapsed:.2f}ms: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


# ============================================================================
# FILE-SPECIFIC SEARCH ENDPOINTS
# ============================================================================


@app.post("/api/search/file", response_model=FileSearchResponse)
async def search_specific_file(request: FileSearchRequest):
    """Search within a specific PDF file"""
    try:
        from langchain_chroma import Chroma

        if not request.collection_name:
            raise HTTPException(
                status_code=400, detail="collection_name required for file search"
            )

        vectorstore = Chroma(
            client=chat_service.chroma_client,
            collection_name=request.collection_name,
            embedding_function=chat_service.embedding_model,
            persist_directory="data/chroma_db",
        )

        context, results, found = file_search_service.search_specific_file(
            vectorstore, request.filename, request.query, request.num_results
        )

        return FileSearchResponse(
            filename=request.filename,
            collection_name=request.collection_name,
            found=found,
            num_results=len(results),
            results=results,
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
                persist_directory="data/chroma_db",
            )
            all_vectorstores[col.name] = vectorstore

        context, results, found, collection_name = (
            file_search_service.search_specific_file_chatall(
                all_vectorstores, request.filename, request.query, request.num_results
            )
        )

        return FileSearchResponse(
            filename=request.filename,
            collection_name=collection_name,
            found=found,
            num_results=len(results),
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# NAMING ENDPOINTS
# ============================================================================


@app.post("/api/collections/validate-name", response_model=ValidateNameResponse)
async def validate_collection_name(request: ValidateNameRequest):
    """Validate a collection name"""
    try:
        is_valid, validated_name, message = naming_service.validate_name(request.name)

        return ValidateNameResponse(
            is_valid=is_valid,
            validated_name=validated_name if is_valid else None,
            message=message,
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
            chat_id, request.role, request.content, request.collection_name
        )

        return OperationResponse(status="success", message="Message added to memory")
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
            collection_name=msg.collection_name,
        )
        for msg in memory.messages
    ]

    return ConversationHistoryResponse(
        chat_id=chat_id, message_count=len(messages), messages=messages
    )


@app.delete("/api/memory/{chat_id}", response_model=OperationResponse)
async def clear_conversation_memory(chat_id: str):
    """Clear conversation memory"""
    success = memory_service.clear_memory(chat_id)

    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")

    return OperationResponse(status="success", message="Memory cleared")


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
        total_messages=len(memory.messages),
    )


# ============================================================================
# CHAT ENDPOINTS
# ============================================================================


@app.get("/api/chat/single/{collection_name}/{message}")
async def chat_single_collection(
    collection_name: str, message: str, chat_id: Optional[str] = Query(None)
):
    """Chat with a single collection"""
    return StreamingResponse(
        generate_chat_response(message, collection_name, "single", chat_id),
        media_type="text/event-stream",
    )


@app.get("/api/chat/all/{message}")
async def chat_all_collections(message: str, chat_id: Optional[str] = Query(None)):
    """Chat with all collections (ChatALL mode)"""
    return StreamingResponse(
        generate_chat_response(message, None, "chatall", chat_id),
        media_type="text/event-stream",
    )


# =============================================================================
# NEW CUSTOM PDFs SELECT ENDPOINTS
# =============================================================================


@app.post(
    "/api/selection/{session_id}/select",
    response_model=PDFSelectionResponse,
    tags=["PDF Selection"],
)
async def select_pdf(session_id: str, request: SelectPDFRequest):
    """
    Select a PDF for targeted querying.

    - **session_id**: Unique session identifier (e.g., user_id or chat_id)
    - **filename**: Name of the PDF to select
    - **collection_name**: Collection containing the PDF
    """
    try:
        # DEBUG: Print all collections
        all_colls = chat_service.chroma_client.list_collections()
        print(f"🔍 DEBUG: All collections in ChromaDB:")
        for col in all_colls:
            print(f"   - '{col.name}'")
        print(f"🔍 DEBUG: Looking for: '{request.collection_name}'")

        # Get vectorstore for the collection
        # vectorstore = collection_manager.get_collection(request.collection_name)
        vectorstore = collection_manager.get_collection(
            request.collection_name, chat_service.embedding_model
        )

        if not vectorstore:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{request.collection_name}' not found",
            )

        # Select the PDF
        success, message = pdf_selection_service.select_pdf(
            session_id=session_id,
            filename=request.filename,
            collection_name=request.collection_name,
            vectorstore=vectorstore,
        )

        if not success:
            raise HTTPException(status_code=400, detail=message)

        # Get updated selection
        selection_data = pdf_selection_service.get_selected_pdfs(session_id)

        return PDFSelectionResponse(
            success=True,
            message=message,
            total_selected=selection_data["total_selected"],
            selected_pdfs=[
                SelectedPDFInfo(**pdf) for pdf in selection_data["selected_pdfs"]
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/selection/{session_id}/deselect",
    response_model=PDFSelectionResponse,
    tags=["PDF Selection"],
)
async def deselect_pdf(session_id: str, request: DeselectPDFRequest):
    """
    Remove a PDF from selection.

    - **session_id**: Unique session identifier
    - **filename**: Name of the PDF to deselect
    - **collection_name**: Collection containing the PDF
    """
    try:
        success, message = pdf_selection_service.deselect_pdf(
            session_id=session_id,
            filename=request.filename,
            collection_name=request.collection_name,
        )

        if not success:
            raise HTTPException(status_code=400, detail=message)

        # Get updated selection
        selection_data = pdf_selection_service.get_selected_pdfs(session_id)

        return PDFSelectionResponse(
            success=True,
            message=message,
            total_selected=selection_data["total_selected"] if selection_data else 0,
            selected_pdfs=(
                [SelectedPDFInfo(**pdf) for pdf in selection_data["selected_pdfs"]]
                if selection_data
                else []
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/selection/{session_id}/batch-select",
    response_model=PDFSelectionResponse,
    tags=["PDF Selection"],
)
async def batch_select_pdfs(session_id: str, request: BatchSelectPDFsRequest):
    """
    Select multiple PDFs at once.

    - **session_id**: Unique session identifier
    - **selections**: List of {filename, collection_name} pairs
    """
    try:
        results = []
        for selection in request.selections:
            filename = selection.get("filename")
            collection_name = selection.get("collection_name")

            if not filename or not collection_name:
                continue

            vectorstore = collection_manager.get_collection(
                collection_name, chat_service.embedding_model
            )
            if not vectorstore:
                continue

            success, message = pdf_selection_service.select_pdf(
                session_id=session_id,
                filename=filename,
                collection_name=collection_name,
                vectorstore=vectorstore,
            )
            results.append((filename, success, message))

        # Get updated selection
        selection_data = pdf_selection_service.get_selected_pdfs(session_id)

        success_count = sum(1 for _, success, _ in results if success)

        return PDFSelectionResponse(
            success=True,
            message=f"Selected {success_count}/{len(request.selections)} PDFs",
            total_selected=selection_data["total_selected"],
            selected_pdfs=[
                SelectedPDFInfo(**pdf) for pdf in selection_data["selected_pdfs"]
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/api/selection/{session_id}/clear",
    response_model=PDFSelectionResponse,
    tags=["PDF Selection"],
)
async def clear_selection(session_id: str):
    """
    Clear all selected PDFs from the session.

    - **session_id**: Unique session identifier
    """
    try:
        success, message = pdf_selection_service.clear_selection(session_id)

        if not success:
            raise HTTPException(status_code=404, detail=message)

        return PDFSelectionResponse(
            success=True, message=message, total_selected=0, selected_pdfs=[]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/selection/{session_id}",
    response_model=SelectionSessionResponse,
    tags=["PDF Selection"],
)
async def get_selection(session_id: str):
    """
    Get all selected PDFs for a session.

    - **session_id**: Unique session identifier
    """
    try:
        selection_data = pdf_selection_service.get_selected_pdfs(session_id)

        if not selection_data:
            raise HTTPException(status_code=404, detail="No selection session found")

        return SelectionSessionResponse(
            session_id=selection_data["session_id"],
            total_selected=selection_data["total_selected"],
            collections_involved=selection_data["collections_involved"],
            selected_pdfs=[
                SelectedPDFInfo(**pdf) for pdf in selection_data["selected_pdfs"]
            ],
            created_at=selection_data["created_at"],
            updated_at=selection_data["updated_at"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/selection/{session_id}/stats",
    response_model=SelectionStatsResponse,
    tags=["PDF Selection"],
)
async def get_selection_stats(session_id: str):
    """
    Get statistics about current selection.

    - **session_id**: Unique session identifier
    """
    try:
        selection_data = pdf_selection_service.get_selected_pdfs(session_id)

        if not selection_data:
            return SelectionStatsResponse(
                total_selected=0,
                collections_involved=[],
                pdfs_by_collection={},
                total_chunks=0,
                total_pages=0,
            )

        # Calculate stats
        pdfs_by_collection = {}
        total_chunks = 0
        total_pages = 0

        for pdf in selection_data["selected_pdfs"]:
            coll_name = pdf["collection_name"]
            pdfs_by_collection[coll_name] = pdfs_by_collection.get(coll_name, 0) + 1
            total_chunks += pdf.get("chunks", 0)
            total_pages += pdf.get("pages", 0)

        return SelectionStatsResponse(
            total_selected=selection_data["total_selected"],
            collections_involved=selection_data["collections_involved"],
            pdfs_by_collection=pdfs_by_collection,
            total_chunks=total_chunks,
            total_pages=total_pages,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/selection/{session_id}/search",
    response_model=SelectedPDFsSearchResponse,
    tags=["PDF Selection"],
)
async def search_selected_pdfs(session_id: str, request: SelectedPDFsSearchRequest):
    """
    Search only within selected PDFs across collections.

    - **session_id**: Unique session identifier
    - **query**: Search query
    - **num_results**: Number of results to return (default: 25)
    """
    try:
        # Get all collections
        # all_collections = collection_manager.get_all_collections(chat_service.embedding_model)
        all_collections = collection_manager.get_all_collections_vectorstores(
            chat_service.embedding_model
        )

        if not all_collections:
            raise HTTPException(status_code=404, detail="No collections available")

        # Get selection info
        selection_data = pdf_selection_service.get_selected_pdfs(session_id)

        if not selection_data or selection_data["total_selected"] == 0:
            raise HTTPException(
                status_code=400, detail="No PDFs selected. Please select PDFs first."
            )

        # Perform search
        context, results, total_results = pdf_selection_service.search_selected_pdfs(
            session_id=session_id,
            query=request.query,
            all_collections=all_collections,
            num_results=request.num_results,
        )

        return SelectedPDFsSearchResponse(
            query=request.query,
            total_results=total_results,
            total_selected_pdfs=selection_data["total_selected"],
            collections_searched=selection_data["collections_involved"],
            results=results,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/selection/{session_id}/chat")
async def chat_with_selected_pdfs(
    session_id: str,
    query: str = Query(..., description="User's question"),
    chat_id: Optional[str] = Query(None, description="Chat session ID"),
    num_results: int = Query(25, description="Number of search results to use"),
):
    """
    Chat with selected PDFs using streaming response.
    Uses your existing ChatService.generate_response method.
    """

    async def generate():
        try:
            print(f"📝 Starting chat with selected PDFs for session: {session_id}")
            print(f"   Query: {query}")

            # 1. Generate chat_id
            current_chat_id = chat_id or f"chat_{session_id}_{int(time.time())}"
            yield f"data: {json.dumps({'type': 'chat_id', 'chat_id': current_chat_id})}\n\n"

            # 2. Check if PDFs are selected
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    stats_response = await client.get(
                        f"http://localhost:8000/api/selection/{session_id}/stats"
                    )
                    if stats_response.status_code != 200:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'No PDFs selected'})}\n\n"
                        return

                    stats = stats_response.json()
                    if stats.get("total_selected", 0) == 0:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Please select PDFs first'})}\n\n"
                        return

                    print(f"✅ Found {stats.get('total_selected')} selected PDFs")
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to verify selection: {str(e)}'})}\n\n"
                return

            # 3. Search within selected PDFs
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    search_response = await client.post(
                        f"http://localhost:8000/api/selection/{session_id}/search",
                        json={"query": query, "num_results": num_results},
                    )

                    if search_response.status_code != 200:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Search failed'})}\n\n"
                        return

                    search_result = search_response.json()
                    results = search_result.get("results", [])
                    print(f"🔍 Found {len(results)} results")

                    if not results:
                        yield f"data: {json.dumps({'type': 'content', 'content': 'No relevant information found in the selected PDFs.'})}\n\n"
                        yield f"data: {json.dumps({'type': 'end'})}\n\n"
                        return

                    yield f"data: {json.dumps({'type': 'search_results', 'count': len(results)})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Search failed: {str(e)}'})}\n\n"
                return

            # 4. Format context from search results
            context_parts = []
            for i, result in enumerate(results[:10], 1):  # Use top 10 results
                filename = result.get("filename", "Unknown")
                content = result.get("content", "")
                page = result.get("page_numbers", "N/A")
                collection = result.get("collection", "")

                context_parts.append(
                    f"[Source {i}] From '{filename}' ({collection} collection, Page {page}):\n{content}"
                )

            context = "\n\n".join(context_parts)
            print(f"📄 Built context ({len(context)} chars)")

            # 5. Stream LLM response using your ChatService.generate_response
            try:
                print("🤖 Streaming LLM response...")
                full_response = ""

                # THIS IS THE KEY CHANGE - use generate_response instead of stream_chat_response
                async for chunk in chat_service.generate_response(query, context):
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

                print(f"✅ Response complete ({len(full_response)} chars)")

            except Exception as e:
                error_msg = f"LLM error: {str(e)}"
                print(f"❌ {error_msg}")
                import traceback

                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                return

            # 6. Send sources
            sources_data = []
            for result in results[:20]:  # Send top 20 sources
                sources_data.append(
                    {
                        "content": result.get("content", ""),
                        "filename": result.get("filename", ""),
                        "collection": result.get("collection", ""),
                        "similarity": result.get("similarity", 0),
                        "page_numbers": result.get("page_numbers", ""),
                        "title": result.get("title", ""),
                    }
                )

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data})}\n\n"
            print(f"📎 Sent {len(sources_data)} sources")

            # 7. Save to memory (if memory_service is available)
            try:
                await memory_service.add_message(current_chat_id, "user", query)
                await memory_service.add_message(
                    current_chat_id, "assistant", full_response
                )
                print("💾 Saved to memory")
            except Exception as e:
                print(f"⚠️ Memory save failed: {e}")

            # 8. End
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
            print(f"✅ Chat complete")

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
