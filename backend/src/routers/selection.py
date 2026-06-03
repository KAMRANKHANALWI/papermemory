from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from typing import Optional
import time
import json
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
    memory_service,
    collection_manager,
    pdf_selection_service,
)

from src.prompts import get_selected_pdf_prompt
from src.services.dependencies import metadata_service
from src.services.dependencies import query_classifier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/selection", tags=["selection"])

TOP_CONTEXT_CHUNKS = 7

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
    query: str = Query(..., description="User's question"),
    chat_id: Optional[str] = Query(None, description="Chat session ID"),
    num_results: int = Query(25, description="Number of search results to use"),
    request: Request = None,
):
    """
    Chat with selected PDFs using streaming response.

    NOTE:
    1. Retrieve candidate chunks using vector search.
    2. Re-rank candidates using CrossEncoder.
    3. Send top reranked chunks to the LLM.
    """

    async def generate():
        try:
            # ── 1. Generate chat ID ────────────────────────────────────────
            current_chat_id = chat_id or f"chat_{session_id}_{int(time.time())}"
            yield f"data: {json.dumps({'type': 'chat_id', 'chat_id': current_chat_id})}\n\n"

            # ── 2. Check session has selected PDFs ─────────────────────────
            session = pdf_selection_service.get_or_create_session(session_id)
            if session.get_selection_count() == 0:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Please select PDFs first'})}\n\n"
                return
            
            # ----------------------------------------
            # Metadata Queries (NO LLM)
            # ----------------------------------------

            classification, _ = query_classifier.classify_query(
                query,
                is_chatall_mode=False,
            )

            if classification in ["list_pdfs", "count_pdfs"]:

                selection_data = session.to_dict()

                filenames = sorted(
                    [
                        pdf["filename"]
                        for pdf in selection_data["selected_pdfs"]
                    ]
                )

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
                    f"data: "
                    f"{json.dumps({'type': 'content', 'content': response})}"
                    f"\n\n"
                )

                yield (
                    f"data: "
                    f"{json.dumps({'type': 'end'})}"
                    f"\n\n"
                )

                return

            # ── 3. Load all collection vectorstores ────────────────────────
            all_collections = collection_manager.get_all_collections_vectorstores(
                chat_service.embedding_model
            )
            if not all_collections:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No collections available'})}\n\n"
                return

            # ── 4. Search within selected PDFs only (re-ranked)
            try:
                context, results, total_results = (
                    pdf_selection_service.search_selected_pdfs(
                        session_id=session_id,
                        query=query,
                        all_collections=all_collections,
                        num_results=num_results,
                    )
                )

                # ==========================
                # RERANK RESULTS
                # ==========================
                results = reranker.rerank(
                    query=query,
                    chunks=results,
                    top_k=AppConfig.TOP_K,
                )

                total_results = len(results)

                # print("\nTOP CHUNKS AFTER RERANKING")
                # print("=" * 80)

                # for i, chunk in enumerate(results, start=1):
                #     print(
                #         f"{i}. score={chunk.get('rerank_score')} "
                #         f"page={chunk.get('page_numbers')}"
                #     )

                if total_results == 0:
                    yield f"data: {json.dumps({'type': 'content', 'content': 'No relevant information found in the selected PDFs.'})}\n\n"
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    return
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Search failed: {str(e)}'})}\n\n"
                return

            # ── 5. Build context string from top results ───────────────────
            context_parts = []
            for i, result in enumerate(results[:TOP_CONTEXT_CHUNKS], 1):
                context_parts.append(
                    f"[Source {i}] From '{result.get('filename', 'Unknown')}' "
                    f"({result.get('collection', '')} collection, "
                    f"Page {result.get('page_numbers', 'N/A')}):\n"
                    f"{result.get('content', '')}"
                )
            context = "\n\n".join(context_parts)

            # ── 6. Send sources BEFORE streaming (frontend shows while LLM runs) ──
            sources_data = [
                {
                    "content": r.get("content", ""),
                    "filename": r.get("filename", ""),
                    "collection": r.get("collection", ""),
                    "similarity": r.get("similarity", 0),
                    "page_numbers": r.get("page_numbers", ""),
                    "title": r.get("title", ""),
                    "rerank_score": r.get("rerank_score", 0),
                }
                for r in results[:TOP_CONTEXT_CHUNKS]
            ]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data})}\n\n"

            # ── 7. Stream LLM response token by token ─────────────────────
            full_response = ""
            try:
                print("\nTOP CHUNKS AFTER RERANKING")
                print("=" * 80)

                for i, chunk in enumerate(results, start=1):
                    print(
                        f"{i}. score={chunk.get('rerank_score')} "
                        f"page={chunk.get('page_numbers')}"
                    )
                    
                # print("\nCONTEXT SENT TO LLM")
                # print("=" * 100)
                # print(context)
                # print("=" * 100)

                # async for chunk in chat_service.generate_response(query, context):
                async for chunk in chat_service.generate_response(
                    query=query,
                    context=context,
                    system_prompt=get_selected_pdf_prompt(),
                ):
                    # Stop if user closed the browser tab
                    if await request.is_disconnected():
                        logger.info("Client disconnected — stopping stream")
                        break
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'LLM error: {str(e)}'})}\n\n"
                return

            # ── 8. Save conversation to memory ─────────────────────────────
            try:
                memory_service.add_message(current_chat_id, "user", query)
                memory_service.add_message(current_chat_id, "assistant", full_response)
            except Exception as e:
                logger.warning(f"Memory save failed: {e}")

            # ── 9. Signal stream complete ──────────────────────────────────
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        except Exception as e:
            logger.error(f"Unexpected error in selection chat: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
