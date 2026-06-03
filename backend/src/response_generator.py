"""
Response generator for chat endpoints with query classification, memory,
and parent-page retrieval support.
"""

import json
import logging
from typing import Optional, AsyncGenerator, Dict, Any, List
from langchain_chroma import Chroma
from fastapi import Request
from src.config import AppConfig

from src.services.shared import (
    chat_service,
    memory_service,
    metadata_service,
    file_search_service,
    query_classifier,
)

from src.prompts import (
    get_scientific_rag_prompt,
    get_metadata_prompt,
    get_collection_prompt,
    get_file_specific_prompt,
)

from src.services.chat_orchestrator import ChatOrchestrator
from src.services.retrieval_service import RetrievalService

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
        print("\nINSIDE GENERATE_CHAT_RESPONSE\n")
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
            handler = handle_metadata_query(
                message,
                classification,
                collection_name,
                is_chatall,
                conversation_history,
                request,
            )
        elif classification == "list_collections" and is_chatall:
            handler = handle_list_collections(message, conversation_history, request)
        elif classification == "file_specific_search" and filename:
            handler = handle_file_specific_search(
                message,
                filename,
                collection_name,
                is_chatall,
                conversation_history,
                request,
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


async def handle_metadata_query(
    message: str,
    collection_name: Optional[str],
    is_chatall: bool,
    conversation_history: List[Dict],
    request: Request = None,
) -> AsyncGenerator[str, None]:

    if is_chatall:
        vectorstores = {
            col.name: get_vectorstore(col.name)
            for col in chat_service.chroma_client.list_collections()
        }
        all_pdfs, stats = metadata_service.get_chatall_collection_pdfs(vectorstores)
        context = metadata_service.format_chatall_pdf_list_for_llm(all_pdfs, stats)
    else:
        if not collection_name:
            raise ValueError("Collection name required")
        vectorstore = get_vectorstore(collection_name)
        filenames, stats = metadata_service.get_single_collection_pdfs(vectorstore)
        context = metadata_service.format_pdf_list_for_llm(filenames, stats)

    # base_prompt = "You are a document assistant. Provide clear, friendly responses about available documents."
    base_prompt = get_metadata_prompt()
    system_prompt = ChatOrchestrator.build_system_prompt_with_history(
        base_prompt, conversation_history, context
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    async for chunk in ChatOrchestrator.stream_llm_response(
        chat_service.llm.astream(messages), request
    ):
        yield chunk


async def handle_list_collections(
    message: str,
    conversation_history: List[Dict],
    request: Request = None,
) -> AsyncGenerator[str, None]:

    collections = chat_service.chroma_client.list_collections()
    lines = []
    for col in collections:
        count = chat_service.chroma_client.get_collection(col.name).count()
        lines.append(f"• {col.name} ({count} chunks)")

    context = f"AVAILABLE COLLECTIONS:\nTotal: {len(collections)}\n\n" + "\n".join(
        lines
    )

    # base_prompt = "You are a document assistant. Provide clear responses about available collections."
    base_prompt = get_collection_prompt()
    system_prompt = ChatOrchestrator.build_system_prompt_with_history(
        base_prompt, conversation_history, context
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    async for chunk in ChatOrchestrator.stream_llm_response(
        chat_service.llm.astream(messages), request
    ):
        yield chunk


async def handle_file_specific_search(
    message: str,
    filename: str,
    collection_name: Optional[str],
    is_chatall: bool,
    conversation_history: List[Dict],
    request: Request = None,
) -> AsyncGenerator[str, None]:

    from src.services.shared import reranker

    if is_chatall:
        vectorstores = {
            col.name: get_vectorstore(col.name)
            for col in chat_service.chroma_client.list_collections()
        }
        context, results, found, _ = file_search_service.search_specific_file_chatall(
            vectorstores, filename, message, num_results=AppConfig.RERANKING_SAMPLE_SIZE
        )
        all_chunks = results
    else:
        if not collection_name:
            raise ValueError("Collection name required")
        vectorstore = get_vectorstore(collection_name)
        raw_results = vectorstore.similarity_search_with_score(
            message, k=AppConfig.RERANKING_SAMPLE_SIZE, filter={"filename": filename}
        )
        all_chunks = [
            {
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "unknown"),
                "page_numbers": doc.metadata.get("page_numbers", "[]"),
                "title": doc.metadata.get("title", "No Title"),
                "similarity": round(1 - float(score), 4),
                "collection": collection_name,
            }
            for doc, score in raw_results
        ]

    if not all_chunks:
        message_text = f'File "{filename}" not found. Searching all documents...'
        yield f"data: {json.dumps({'type': 'content', 'content': message_text})}\n\n"
        async for event in handle_content_search(
            message, collection_name, is_chatall, conversation_history, request
        ):
            yield event
        return

    # Rerank chunks
    top_chunks = reranker.rerank(message, all_chunks, top_k=AppConfig.TOP_K)

    print("\nTOP CHUNKS AFTER RERANKING")
    print("=" * 100)

    for i, chunk in enumerate(top_chunks, start=1):
        print(f"\nChunk #{i}")
        print(f"Score: {chunk.get('rerank_score', chunk.get('similarity'))}")
        print(f"Pages: {chunk.get('page_numbers')}")

        preview = chunk["content"][:500]
        print(preview)
        print("-" * 50)

    # Build context from reranked chunks directly
    context_parts = []
    for chunk in top_chunks:
        header = f"[Source: {chunk['filename']} | Pages {chunk['page_numbers']}]"
        context_parts.append(f"{header}\n{chunk['content']}")
    context = "\n\n---\n\n".join(context_parts)

    sources = [
        {
            "content": c["content"],
            "filename": c["filename"],
            "collection": c.get("collection", collection_name),
            "page_numbers": c["page_numbers"],
            "similarity": c.get("rerank_score", c["similarity"]),
            "rerank_score": c.get("rerank_score", c["similarity"]),
            "title": c.get("title", "No Title"),
        }
        for c in top_chunks
    ]
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    # base_prompt = f"You are a document assistant answering about: {filename}. Use ONLY context information."
    base_prompt = get_file_specific_prompt(filename)
    system_prompt = ChatOrchestrator.build_system_prompt_with_history(
        base_prompt, conversation_history, context
    )

    print("\n" + "=" * 100)
    print("FILE SPECIFIC SEARCH")
    print("=" * 100)

    print(f"\nQuestion:\n{message}")

    print("\nContext Length:")
    print(len(context))

    print("\nContext Sent To LLM:")
    print(context)

    print("\n" + "=" * 100)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    async for chunk in ChatOrchestrator.stream_llm_response(
        chat_service.llm.astream(messages), request
    ):
        yield chunk


async def handle_content_search(
    message: str,
    collection_name: Optional[str],
    is_chatall: bool,
    conversation_history: List[Dict],
    request: Request = None,
) -> AsyncGenerator[str, None]:

    from src.services.shared import reranker

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
