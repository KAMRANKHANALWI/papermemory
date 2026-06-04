from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
import time
import json
import logging

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
    memory_service,
    collection_manager,
    pdf_selection_service,
    query_classifier,
    metadata_service,
)

from src.prompts import get_selected_pdf_prompt
from typing import Optional

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
    num_results: int = Query(20),
    request: Request = None,
):
    async def generate():
        try:
            current_chat_id = chat_id or f"chat_{session_id}_{int(time.time())}"
            yield f"data: {json.dumps({'type': 'chat_id', 'chat_id': current_chat_id})}\n\n"

            session = pdf_selection_service.get_or_create_session(session_id)
            if session.get_selection_count() == 0:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Please select PDFs first'})}\n\n"
                return
            
            selection_data = session.to_dict()

            classification, _ = query_classifier.classify_query(
                query,
                is_chatall_mode=False,
            )

            if classification in ["count_pdfs", "list_pdfs"]:

                filenames = [
                    pdf["filename"]
                    for pdf in selection_data["selected_pdfs"]
                ]

                stats = {
                    "total_pdfs": len(filenames)
                }

                response = metadata_service.build_metadata_response(
                    classification=classification,
                    query=query,
                    filenames=filenames,
                    stats=stats,
                )

                yield (
                    f"data: {json.dumps({'type': 'content', 'content': response})}\n\n"
                )

                yield f"data: {json.dumps({'type': 'end'})}\n\n"

                return

            all_collections = collection_manager.get_all_collections_vectorstores(
                chat_service.embedding_model
            )
            if not all_collections:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No collections available'})}\n\n"
                return

            try:
                context, results, total_results = (
                    pdf_selection_service.search_selected_pdfs(
                        session_id=session_id,
                        query=query,
                        all_collections=all_collections,
                        num_results=num_results,
                    )
                )
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Search failed: {str(e)}'})}\n\n"
                return

            if total_results == 0:
                yield f"data: {json.dumps({'type': 'content', 'content': 'No relevant information found in the selected PDFs.'})}\n\n"
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                return

            # Build context from results
            context_parts = []
            for i, result in enumerate(results, 1):
                context_parts.append(
                    f"[Source {i}] From '{result.get('filename', 'Unknown')}' "
                    f"({result.get('collection', '')} collection, "
                    f"Page {result.get('page_numbers', 'N/A')}):\n"
                    f"{result.get('content', '')}"
                )
            context = "\n\n".join(context_parts)

            # Sources BEFORE streaming
            sources_data = [
                {
                    "content": r.get("content", ""),
                    "filename": r.get("filename", ""),
                    "collection": r.get("collection", ""),
                    "similarity": r.get("similarity", 0),
                    "page_numbers": r.get("page_numbers", ""),
                    "title": r.get("title", ""),
                }
                for r in results
            ]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data})}\n\n"

            # Stream LLM response
            full_response = ""
            try:
                system_prompt = f"""
                {get_selected_pdf_prompt()}

                Context:

                {context}
                """
                async for chunk in chat_service.generate_response(query=query, system_prompt=system_prompt):
                    if await request.is_disconnected():
                        logger.info("Client disconnected")
                        break
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'LLM error: {str(e)}'})}\n\n"
                return

            # Save to memory
            try:
                memory_service.add_message(current_chat_id, "user", query)
                memory_service.add_message(current_chat_id, "assistant", full_response)
            except Exception as e:
                logger.warning(f"Memory save failed: {e}")

            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        except Exception as e:
            logger.error(f"Unexpected error in selection chat: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
