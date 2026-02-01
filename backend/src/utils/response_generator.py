"""
Response generator for chat endpoints with query classification and memory support
"""

import json
import logging
from typing import Optional, AsyncGenerator, Dict, Any, List
from langchain_chroma import Chroma

from src.services.query_classifier import QueryClassifier
from src.services.metadata_service import MetadataService
from src.services.file_search_service import FileSearchService
from src.services.chat_service import ChatService
from src.services.memory_service import MemoryService

logger = logging.getLogger(__name__)

# -------------------------
# Service initialization
# -------------------------
chat_service = ChatService()
query_classifier = QueryClassifier(chat_service.llm)
metadata_service = MetadataService()
file_search_service = FileSearchService()
memory_service = MemoryService()


# -------------------------
# HELPERS FUNCTIONS
# -------------------------
def get_vectorstore(collection_name: str) -> Chroma:
    return Chroma(
        client=chat_service.chroma_client,
        collection_name=collection_name,
        embedding_function=chat_service.embedding_model,
        persist_directory="data/chroma_db",
    )


async def stream_llm_response(response_stream):
    for chunk in response_stream:
        if hasattr(chunk, "content") and chunk.content:
            yield f"data: {json.dumps({'type': 'content', 'content': chunk.content})}\n\n"


def collect_content_from_event(event: str) -> str:
    if not event.startswith("data: "):
        return ""
    try:
        payload = json.loads(event[6:].strip())
        if payload.get("type") == "content":
            return payload.get("content", "")
    except Exception:
        pass
    return ""


def build_system_prompt_with_history(
    base_prompt: str, conversation_history: List[Dict], context: str
) -> str:
    if not conversation_history:
        return f"{base_prompt}\n\nContext:\n{context}"

    history_lines = []
    for msg in conversation_history[:-1]:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if len(content) > 200:
            content = content[:200] + "..."
        history_lines.append(f"{role}: {content}")

    history_text = "\n".join(history_lines)

    return (
        f"{base_prompt}\n\n"
        f"Previous conversation:\n{history_text}\n\n"
        f"Current context:\n{context}\n\n"
        f"Maintain context from previous conversation."
    )


# -------------------------
# MAIN FUNCTIONS
# -------------------------
async def generate_chat_response(
    message: str,
    collection_name: Optional[str],
    chat_mode: str,
    chat_id: Optional[str] = None,
    eval_mode: bool = False,
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
                chat_id, max_messages=10
            )
        except Exception:
            conversation_history = []

        full_response = ""

        if classification in ["list_pdfs", "count_pdfs"]:
            handler = handle_metadata_query(
                message, classification, collection_name, is_chatall, conversation_history
            )

        elif classification == "list_collections" and is_chatall:
            handler = handle_list_collections(message, conversation_history)

        elif classification == "file_specific_search" and filename:
            handler = handle_file_specific_search(
                message, filename, collection_name, is_chatall, conversation_history
            )

        else:
            handler = handle_content_search(
                message, collection_name, is_chatall, conversation_history
            )

        async for event in handler:
            full_response += collect_content_from_event(event)
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
    classification: str,
    collection_name: Optional[str],
    is_chatall: bool,
    conversation_history: List[Dict],
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

    base_prompt = "You are a document assistant. Provide clear, friendly responses about available documents."
    system_prompt = build_system_prompt_with_history(
        base_prompt, conversation_history, context
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    async for chunk in stream_llm_response(chat_service.llm.stream(messages)):
        yield chunk



async def handle_list_collections(
    message: str, conversation_history: List[Dict]
) -> AsyncGenerator[str, None]:

    collections = chat_service.chroma_client.list_collections()
    lines = []

    for col in collections:
        count = chat_service.chroma_client.get_collection(col.name).count()
        lines.append(f"• {col.name} ({count} chunks)")

    context = f"AVAILABLE COLLECTIONS:\nTotal: {len(collections)}\n\n" + "\n".join(lines)

    base_prompt = "You are a document assistant. Provide clear responses about available collections."
    system_prompt = build_system_prompt_with_history(
        base_prompt, conversation_history, context
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    async for chunk in stream_llm_response(chat_service.llm.stream(messages)):
        yield chunk



async def handle_file_specific_search(
    message: str,
    filename: str,
    collection_name: Optional[str],
    is_chatall: bool,
    conversation_history: List[Dict],
) -> AsyncGenerator[str, None]:

    if is_chatall:
        vectorstores = {
            col.name: get_vectorstore(col.name)
            for col in chat_service.chroma_client.list_collections()
        }
        context, results, found, _ = file_search_service.search_specific_file_chatall(
            vectorstores, filename, message, num_results=10
        )
    else:
        if not collection_name:
            raise ValueError("Collection name required")
        vectorstore = get_vectorstore(collection_name)
        context, results, found = file_search_service.search_specific_file(
            vectorstore, filename, message, num_results=10, collection_name=collection_name
        )

    if not found:
        not_found_msg = f'File "{filename}" not found. Searching all documents...'
        yield f"data: {json.dumps({'type': 'content', 'content': not_found_msg})}\n\n"

        async for event in handle_content_search(
            message, collection_name, is_chatall, conversation_history
        ):
            yield event
        return

    sources = [
        {
            "content": r["content"],
            "filename": r["filename"],
            "collection": r.get("collection"),
            "similarity": r["similarity"],
            "page_numbers": r.get("pages"),
            "title": r.get("title"),
        }
        for r in results
    ]

    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    base_prompt = f"You are a document assistant answering about: {filename}. Use ONLY context information."
    system_prompt = build_system_prompt_with_history(
        base_prompt, conversation_history, context
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    async for chunk in stream_llm_response(chat_service.llm.stream(messages)):
        yield chunk



async def handle_content_search(
    message: str,
    collection_name: Optional[str],
    is_chatall: bool,
    conversation_history: List[Dict],
) -> AsyncGenerator[str, None]:

    all_results = []

    if is_chatall:
        for col in chat_service.chroma_client.list_collections():
            try:
                vectorstore = get_vectorstore(col.name)
                results = vectorstore.similarity_search_with_score(message, k=4)
                for doc, score in results:
                    all_results.append(
                        {
                            "content": doc.page_content,
                            "filename": doc.metadata.get("filename", "unknown"),
                            "title": doc.metadata.get("title", "No Title"),
                            "pages": doc.metadata.get("page_numbers", "[]"),
                            "similarity": round(1 - score, 4),
                            "collection": col.name,
                        }
                    )
            except Exception as e:
                logger.warning(f"Search failed for {col.name}: {e}")

        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        all_results = all_results[:25]

    else:
        if not collection_name:
            raise ValueError("Collection name required")
        vectorstore = get_vectorstore(collection_name)
        results = vectorstore.similarity_search_with_score(message, k=10)

        for doc, score in results:
            all_results.append(
                {
                    "content": doc.page_content,
                    "filename": doc.metadata.get("filename", "unknown"),
                    "title": doc.metadata.get("title", "No Title"),
                    "pages": doc.metadata.get("page_numbers", "[]"),
                    "similarity": round(1 - score, 4),
                    "collection": collection_name,
                }
            )

    context_parts = []
    for r in all_results:
        src = f"Source: {r['filename']} (Collection: {r['collection']})"
        if r["pages"] != "[]":
            pages = r["pages"].strip("[]").replace("'", "").split(",")
            if pages and pages[0]:
                src += f" - p. {', '.join(pages)}"
        context_parts.append(f"{r['content']}\n\n{src}")

    context = "\n\n".join(context_parts)

    yield f"data: {json.dumps({'type': 'sources', 'sources': all_results})}\n\n"

    scope = "across all collections" if is_chatall else f"from {collection_name}"
    base_prompt = f"You are a document assistant answering from documents {scope}. Use ONLY context information."

    system_prompt = build_system_prompt_with_history(
        base_prompt, conversation_history, context
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    async for chunk in stream_llm_response(chat_service.llm.stream(messages)):
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
