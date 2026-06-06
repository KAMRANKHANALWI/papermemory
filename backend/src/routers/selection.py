from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from typing import Optional

import logging

from src.services.dependencies import reranker
from src.config import AppConfig

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
from src.services.dependencies import (
    chat_service,
    collection_manager,
    pdf_selection_service,
)


from src.controllers.selection_controller import (
    chat_with_selected_pdfs as selection_chat_handler,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/selection", tags=["selection"])

@router.post("/{session_id}/select", response_model=PDFSelectionResponse)
async def select_pdf(session_id: str, request: SelectPDFRequest):
    """Select a PDF for targeted querying"""
    try:
        vectorstore = collection_manager.get_collection(
            request.collection_name, chat_service.embedding_model
        )
        if not vectorstore:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{request.collection_name}' not found",
            )

        success, message = pdf_selection_service.select_pdf(
            session_id=session_id,
            filename=request.filename,
            collection_name=request.collection_name,
            vectorstore=vectorstore,
        )
        if not success:
            raise HTTPException(status_code=400, detail=message)

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


@router.post("/{session_id}/deselect", response_model=PDFSelectionResponse)
async def deselect_pdf(session_id: str, request: DeselectPDFRequest):
    """Remove a PDF from selection"""
    try:
        success, message = pdf_selection_service.deselect_pdf(
            session_id=session_id,
            filename=request.filename,
            collection_name=request.collection_name,
        )
        if not success:
            raise HTTPException(status_code=400, detail=message)

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


@router.post("/{session_id}/batch-select", response_model=PDFSelectionResponse)
async def batch_select_pdfs(session_id: str, request: BatchSelectPDFsRequest):
    """Select multiple PDFs at once"""
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


@router.delete("/{session_id}/clear", response_model=PDFSelectionResponse)
async def clear_selection(session_id: str):
    """Clear all selected PDFs from the session"""
    try:
        session = pdf_selection_service.get_or_create_session(session_id)
        session.clear_all()
        return PDFSelectionResponse(
            success=True,
            message="Selection cleared",
            total_selected=0,
            selected_pdfs=[],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}", response_model=SelectionSessionResponse)
async def get_selection(session_id: str):
    """Get all selected PDFs for a session"""
    try:
        session = pdf_selection_service.get_or_create_session(session_id)
        selection_data = session.to_dict()
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/stats", response_model=SelectionStatsResponse)
async def get_selection_stats(session_id: str):
    """Get statistics about current selection"""
    try:
        session = pdf_selection_service.get_or_create_session(session_id)
        selection_data = session.to_dict()
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


@router.post("/{session_id}/search", response_model=SelectedPDFsSearchResponse)
async def search_selected_pdfs(session_id: str, request: SelectedPDFsSearchRequest):
    """Search only within selected PDFs"""
    try:
        all_collections = collection_manager.get_all_collections_vectorstores(
            chat_service.embedding_model
        )
        if not all_collections:
            raise HTTPException(status_code=404, detail="No collections available")

        session = pdf_selection_service.get_or_create_session(session_id)
        if session.get_selection_count() == 0:
            raise HTTPException(
                status_code=400, detail="No PDFs selected. Please select PDFs first."
            )

        selection_data = session.to_dict()
        context, results, total_results = pdf_selection_service.search_selected_pdfs(
            session_id=session_id,
            query=request.query,
            all_collections=all_collections,
            num_results=request.num_results,
        )

        results = reranker.rerank(
            query=request.query, chunks=results, top_k=AppConfig.TOP_K
        )

        total_results = len(results)

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


@router.get("/{session_id}/chat")
async def chat_with_selected_pdfs(
    session_id: str,
    query: str = Query(...),
    chat_id: Optional[str] = Query(None),
    num_results: int = Query(25),
    request: Request = None,
):

    return StreamingResponse(
        selection_chat_handler(
            session_id=session_id,
            query=query,
            chat_id=chat_id,
            num_results=num_results,
            request=request,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
