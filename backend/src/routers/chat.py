from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.models import (
    QueryClassificationRequest,
    QueryClassificationResponse,
    ChatRequest,
)
from src.services.dependencies import query_classifier
from src.controllers.chat_controller import (
    generate_chat_response,
    generate_chat_response_eval,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/classify", response_model=QueryClassificationResponse)
async def classify_query(request: QueryClassificationRequest):
    """Classify user query intent"""
    try:
        classification, filename = query_classifier.classify_query(
            request.query, request.is_chatall_mode
        )
        return QueryClassificationResponse(
            classification=classification, filename=filename, confidence=1.0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@router.post("/single/{collection_name}")
async def chat_single_collection(
    collection_name: str,
    body: ChatRequest,
    request: Request,
):
    """Chat with a single collection"""
    if body.eval:
        return await generate_chat_response_eval(
            message=body.message,
            collection_name=collection_name,
            chat_mode="single",
            chat_id=body.chat_id,
        )

    return StreamingResponse(
        generate_chat_response(
            message=body.message,
            collection_name=collection_name,
            chat_mode="single",
            chat_id=body.chat_id,
            eval_mode=False,
            request=request,
        ),
        media_type="text/event-stream",
    )


@router.post("/all")
async def chat_all_collections(
    body: ChatRequest,
    request: Request,
):
    """Chat across all collections"""
    if body.eval:
        return await generate_chat_response_eval(
            message=body.message,
            collection_name=None,
            chat_mode="chatall",
            chat_id=body.chat_id,
        )

    return StreamingResponse(
        generate_chat_response(
            message=body.message,
            collection_name=None,
            chat_mode="chatall",
            chat_id=body.chat_id,
            eval_mode=False,
            request=request,
        ),
        media_type="text/event-stream",
    )