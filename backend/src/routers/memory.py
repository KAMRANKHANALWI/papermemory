from fastapi import APIRouter, HTTPException, Query

from src.models import (
    AddMemoryRequest,
    ConversationHistoryResponse,
    MemorySummaryResponse,
    OperationResponse,
    MemoryMessage,
)
from src.services.dependencies import memory_service

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.post("/{chat_id}/add", response_model=OperationResponse)
async def add_message_to_memory(chat_id: str, request: AddMemoryRequest):
    """Add a message to conversation memory"""
    try:
        memory_service.add_message(
            chat_id, request.role, request.content, request.collection_name
        )
        return OperationResponse(status="success", message="Message added to memory")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{chat_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(chat_id: str, max_messages: int = Query(10)):
    """Get conversation history"""
    memory = memory_service.get_memory(chat_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Chat not found")

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


@router.delete("/{chat_id}", response_model=OperationResponse)
async def clear_conversation_memory(chat_id: str):
    """Clear conversation memory"""
    success = memory_service.clear_memory(chat_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    return OperationResponse(status="success", message="Memory cleared")


@router.get("/{chat_id}/summary", response_model=MemorySummaryResponse)
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