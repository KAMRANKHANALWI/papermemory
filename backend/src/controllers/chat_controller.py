import json
import logging
from typing import Optional, AsyncGenerator, Dict, Any, List
from langchain_chroma import Chroma
from fastapi import Request
from src.config import AppConfig

from src.services.dependencies import (
    chat_service,
    memory_service,
    metadata_service,
    file_search_service,
    query_classifier,
)

from src.prompts import (
    get_scientific_rag_prompt,
    get_collection_prompt,
    get_file_specific_prompt,
)

from src.orchestrators.chat_orchestrator import ChatOrchestrator
from src.services.retrieval_service import RetrievalService

orchestrator = ChatOrchestrator(
    chat_service=chat_service,
    memory_service=memory_service,
    metadata_service=metadata_service,
    file_search_service=file_search_service,
)

logger = logging.getLogger(__name__)


# -------------------------
# HELPERS
# -------------------------
def get_vectorstore(collection_name: str) -> Chroma:
    return Chroma(
        client=chat_service.chroma_client,
        collection_name=collection_name,
        embedding_function=chat_service.embedding_model,
    )


# -------------------------
# MAIN ENTRY POINT
# -------------------------
async def generate_chat_response(
    message: str,
    collection_name: Optional[str],
    chat_mode: str,
    chat_id: Optional[str] = None,
    eval_mode: bool = False,
    request: Request = None,
) -> AsyncGenerator[str, None]:

    try:
        if not chat_id:
            import uuid

            chat_id = str(uuid.uuid4())

        yield f"data: {json.dumps({'type': 'chat_id', 'chat_id': chat_id})}\n\n"

        if not eval_mode:
            try:
                memory_service.add_message(chat_id, "user", message, collection_name)
            except Exception as e:
                logger.warning(f"Memory add failed: {e}")

        is_chatall = chat_mode == "chatall"

        classification, filename = query_classifier.classify_query(
            message, is_chatall_mode=is_chatall
        )

        try:
            conversation_history = memory_service.get_formatted_history(
                chat_id, max_messages=AppConfig.MAX_HISTORY
            )
        except Exception:
            conversation_history = []

        full_response = ""

        if classification in ["list_pdfs", "count_pdfs"]:
            handler = orchestrator.handle_metadata_query(
                message=message,
                classification=classification,
                collection_name=collection_name,
                is_chatall=is_chatall,
                get_vectorstore=get_vectorstore,
                request=request,
            )

        elif classification == "list_collections" and is_chatall:
            handler = orchestrator.handle_list_collections(
                message=message,
                conversation_history=conversation_history,
                collection_prompt=get_collection_prompt(),
                request=request,
            )

        elif classification == "file_specific_search" and filename:

            from src.services.dependencies import reranker

            handler = orchestrator.handle_file_specific_search(
                message=message,
                filename=filename,
                collection_name=collection_name,
                is_chatall=is_chatall,
                conversation_history=conversation_history,
                get_vectorstore=get_vectorstore,
                file_specific_prompt=get_file_specific_prompt(filename),
                content_search_handler=handle_content_search,
                retrieval_service=RetrievalService,
                reranker=reranker,
                request=request,
            )
        else:
            handler = handle_content_search(
                message, collection_name, is_chatall, conversation_history, request
            )

        async for event in handler:
            full_response += ChatOrchestrator.collect_content_from_event(event)
            yield event

        if full_response and not eval_mode:
            try:
                memory_service.add_message(
                    chat_id, "assistant", full_response, collection_name
                )
            except Exception as e:
                logger.warning(f"Memory add failed: {e}")

        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    except Exception as e:
        logger.error("Error in generate_chat_response", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


# -------------------------
# HANDLERS
# -------------------------


async def handle_content_search(
    message: str,
    collection_name: Optional[str],
    is_chatall: bool,
    conversation_history: List[Dict],
    request: Request = None,
) -> AsyncGenerator[str, None]:

    from src.services.dependencies import reranker

    context, sources = RetrievalService.retrieve_content(
        message=message,
        is_chatall=is_chatall,
        collection_name=collection_name,
        chroma_client=chat_service.chroma_client,
        get_vectorstore=get_vectorstore,
        reranker=reranker,
        logger=logger,
    )

    if not sources:
        yield (f"data: " f"{json.dumps({'type': 'sources', 'sources': []})}" f"\n\n")
        return

    yield (f"data: " f"{json.dumps({'type': 'sources', 'sources': sources})}" f"\n\n")

    base_prompt = get_scientific_rag_prompt()

    system_prompt = ChatOrchestrator.build_system_prompt_with_history(
        base_prompt,
        conversation_history,
        context,
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": message,
        },
    ]

    async for chunk in ChatOrchestrator.stream_llm_response(
        chat_service.llm.astream(messages),
        request,
    ):
        yield chunk


# -------------------------
# EVAL MODE
# -------------------------
async def generate_chat_response_eval(
    message: str,
    collection_name: Optional[str],
    chat_mode: str,
    chat_id: Optional[str] = None,
) -> Dict[str, Any]:

    collected_sources = []
    full_response = ""

    async for event in generate_chat_response(
        message=message,
        collection_name=collection_name,
        chat_mode=chat_mode,
        chat_id=chat_id,
        eval_mode=True,
    ):
        if not event.startswith("data: "):
            continue

        payload = json.loads(event[6:].strip())

        if payload.get("type") == "sources":
            collected_sources.extend(payload.get("sources", []))
        elif payload.get("type") == "content":
            full_response += payload.get("content", "")
        elif payload.get("type") == "end":
            break

    return {
        "question": message,
        "answer": full_response.strip(),
        "contexts": [s["content"] for s in collected_sources],
        "sources": collected_sources,
        "collection": collection_name,
    }
