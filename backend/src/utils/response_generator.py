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

# Initialize services
chat_service = ChatService()
query_classifier = QueryClassifier(chat_service.llm)
metadata_service = MetadataService()
file_search_service = FileSearchService()
memory_service = MemoryService()


async def generate_chat_response(
    message: str,
    collection_name: Optional[str],
    chat_mode: str,
    chat_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Generate streaming chat response with query classification and memory."""
    try:
        if not chat_id:
            import uuid

            chat_id = str(uuid.uuid4())

        yield f"data: {json.dumps({'type': 'chat_id', 'chat_id': chat_id})}\n\n"

        # Add user message to memory
        try:
            memory_service.add_message(chat_id, "user", message, collection_name)
            logger.info(f"📝 Added user message to memory: {chat_id}")
        except Exception as e:
            logger.warning(f"Memory add failed: {e}")

        is_chatall = chat_mode == "chatall"
        logger.info(f"🔍 Classifying query: '{message}' (mode: {chat_mode})")

        classification, filename = query_classifier.classify_query(
            message, is_chatall_mode=is_chatall
        )
        logger.info(
            f"✅ Classification: {classification}"
            + (f" | File: {filename}" if filename else "")
        )

        # Get conversation history
        conversation_history = []
        try:
            conversation_history = memory_service.get_formatted_history(
                chat_id, max_messages=10
            )
            logger.info(
                f"💾 Retrieved {len(conversation_history)} messages from memory"
            )
        except Exception as e:
            logger.warning(f"Memory retrieve failed: {e}")

        # Handle based on classification
        full_response = ""

        if classification in ["list_pdfs", "count_pdfs"]:
            async for event in handle_metadata_query(
                message,
                classification,
                collection_name,
                is_chatall,
                conversation_history,
            ):
                if event.startswith("data: "):
                    try:
                        data = json.loads(event[6:].strip())
                        if data.get("type") == "content":
                            full_response += data.get("content", "")
                    except:
                        pass
                yield event

        elif classification == "list_collections" and is_chatall:
            async for event in handle_list_collections(message, conversation_history):
                if event.startswith("data: "):
                    try:
                        data = json.loads(event[6:].strip())
                        if data.get("type") == "content":
                            full_response += data.get("content", "")
                    except:
                        pass
                yield event

        elif classification == "file_specific_search" and filename:
            async for event in handle_file_specific_search(
                message, filename, collection_name, is_chatall, conversation_history
            ):
                if event.startswith("data: "):
                    try:
                        data = json.loads(event[6:].strip())
                        if data.get("type") == "content":
                            full_response += data.get("content", "")
                    except:
                        pass
                yield event

        else:
            async for event in handle_content_search(
                message, collection_name, is_chatall, conversation_history
            ):
                if event.startswith("data: "):
                    try:
                        data = json.loads(event[6:].strip())
                        if data.get("type") == "content":
                            full_response += data.get("content", "")
                    except:
                        pass
                yield event

        # Add assistant response to memory
        if full_response:
            try:
                memory_service.add_message(
                    chat_id, "assistant", full_response, collection_name
                )
                logger.info(f"📝 Added assistant response to memory: {chat_id}")
            except Exception as e:
                logger.warning(f"Memory add failed: {e}")

        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    except Exception as e:
        logger.error(f"Error in generate_chat_response: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


def build_system_prompt_with_history(
    base_prompt: str, conversation_history: List[Dict], context: str
) -> str:
    """Build system prompt with conversation history."""
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

    return f"{base_prompt}\n\nPrevious conversation:\n{history_text}\n\nCurrent context:\n{context}\n\nMaintain context from previous conversation."


async def handle_metadata_query(
    message: str,
    classification: str,
    collection_name: Optional[str],
    is_chatall: bool,
    conversation_history: List[Dict],
) -> AsyncGenerator[str, None]:
    """Handle list_pdfs and count_pdfs queries."""
    try:
        if is_chatall:
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

            all_pdfs, stats = metadata_service.get_chatall_collection_pdfs(
                all_vectorstores
            )
            context = metadata_service.format_chatall_pdf_list_for_llm(all_pdfs, stats)
        else:
            if not collection_name:
                raise ValueError("Collection name required")

            vectorstore = Chroma(
                client=chat_service.chroma_client,
                collection_name=collection_name,
                embedding_function=chat_service.embedding_model,
                persist_directory="data/chroma_db",
            )

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

        response_stream = chat_service.llm.stream(messages)

        for chunk in response_stream:
            if hasattr(chunk, "content") and chunk.content:
                yield f"data: {json.dumps({'type': 'content', 'content': chunk.content})}\n\n"

    except Exception as e:
        logger.error(f"Error in handle_metadata_query: {e}")
        raise


async def handle_list_collections(
    message: str, conversation_history: List[Dict]
) -> AsyncGenerator[str, None]:
    """Handle list_collections query."""
    try:
        collections = chat_service.chroma_client.list_collections()

        collection_info = []
        for col in collections:
            collection = chat_service.chroma_client.get_collection(col.name)
            count = collection.count()
            collection_info.append(f"• {col.name} ({count} chunks)")

        context = f"AVAILABLE COLLECTIONS:\nTotal: {len(collections)}\n\n" + "\n".join(
            collection_info
        )

        base_prompt = "You are a document assistant. Provide clear responses about available collections."
        system_prompt = build_system_prompt_with_history(
            base_prompt, conversation_history, context
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        response_stream = chat_service.llm.stream(messages)

        for chunk in response_stream:
            if hasattr(chunk, "content") and chunk.content:
                yield f"data: {json.dumps({'type': 'content', 'content': chunk.content})}\n\n"

    except Exception as e:
        logger.error(f"Error in handle_list_collections: {e}")
        raise


async def handle_file_specific_search(
    message: str,
    filename: str,
    collection_name: Optional[str],
    is_chatall: bool,
    conversation_history: List[Dict],
) -> AsyncGenerator[str, None]:
    """Handle file-specific search."""
    try:
        if is_chatall:
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

            context, search_results, found, found_collection = (
                file_search_service.search_specific_file_chatall(
                    all_vectorstores, filename, message, num_results=25
                )
            )

            if not found:
                not_found_msg = (
                    f'File "{filename}" not found. Searching all documents...'
                )
                yield f"data: {json.dumps({'type': 'content', 'content': not_found_msg})}\n\n"
                async for event in handle_content_search(
                    message, None, True, conversation_history
                ):
                    yield event
                return
        else:
            if not collection_name:
                raise ValueError("Collection name required")

            vectorstore = Chroma(
                client=chat_service.chroma_client,
                collection_name=collection_name,
                embedding_function=chat_service.embedding_model,
                persist_directory="data/chroma_db",
            )

            context, search_results, found = file_search_service.search_specific_file(
                vectorstore, filename, message, num_results=25, collection_name=collection_name
            )

            if not found:
                not_found_msg = (
                    f'File "{filename}" not found. Searching all documents...'
                )
                yield f"data: {json.dumps({'type': 'content', 'content': not_found_msg})}\n\n"
                async for event in handle_content_search(
                    message, collection_name, False, conversation_history
                ):
                    yield event
                return

        sources = [
            {
                "content": result["content"],
                "filename": result["filename"],
                "collection": result.get("collection"),
                "collection": result.get("collection"),
                "similarity": result["similarity"],
                "page_numbers": result.get("pages"),
                "title": result.get("title"),
            }
            for result in search_results
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

        response_stream = chat_service.llm.stream(messages)

        for chunk in response_stream:
            if hasattr(chunk, "content") and chunk.content:
                yield f"data: {json.dumps({'type': 'content', 'content': chunk.content})}\n\n"

    except Exception as e:
        logger.error(f"Error in handle_file_specific_search: {e}")
        raise


async def handle_content_search(
    message: str,
    collection_name: Optional[str],
    is_chatall: bool,
    conversation_history: List[Dict],
) -> AsyncGenerator[str, None]:
    """Handle regular content search"""
    try:
        all_results = []

        if is_chatall:
            collections = chat_service.chroma_client.list_collections()

            for col in collections:
                try:
                    vectorstore = Chroma(
                        client=chat_service.chroma_client,
                        collection_name=col.name,
                        embedding_function=chat_service.embedding_model,
                        persist_directory="data/chroma_db",
                    )

                    results = vectorstore.similarity_search_with_score(message, k=8)

                    for doc, score in results:
                        # doc is a Document object, access with attributes not .get()
                        all_results.append(
                            {
                                "content": doc.page_content,  # Attribute, not dict
                                "filename": doc.metadata.get("filename", "unknown"),
                                "title": doc.metadata.get("title", "No Title"),
                                "pages": doc.metadata.get("page_numbers", "[]"),
                                "similarity": round(1 - score, 4),
                                "collection": col.name,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Error searching collection {col.name}: {e}")

            all_results.sort(key=lambda x: x["similarity"], reverse=True)
            all_results = all_results[:25]

        else:
            if not collection_name:
                raise ValueError("Collection name required")

            vectorstore = Chroma(
                client=chat_service.chroma_client,
                collection_name=collection_name,
                embedding_function=chat_service.embedding_model,
                persist_directory="data/chroma_db",
            )

            results = vectorstore.similarity_search_with_score(message, k=25)

            for doc, score in results:
                # doc is a Document object
                all_results.append(
                    {
                        "content": doc.page_content,  # Attribute
                        "filename": doc.metadata.get("filename", "unknown"),
                        "title": doc.metadata.get("title", "No Title"),
                        "pages": doc.metadata.get("page_numbers", "[]"),
                        "similarity": round(1 - score, 4),
                        # "collection": col.name,
                        "collection": collection_name,
                    }
                )

        # Build context
        context_parts = []
        for result in all_results:
            source_info = f"Source: {result['filename']}"
            if "collection" in result:
                source_info += f" (Collection: {result['collection']})"
            if result["pages"] != "[]":
                pages = result["pages"].strip("[]").replace("'", "").split(",")
                if pages and pages[0]:
                    source_info += f" - p. {', '.join(pages)}"
            context_parts.append(f"{result['content']}\n\n{source_info}")

        context = "\n\n".join(context_parts)

        # Send sources - all_results are now dicts
        sources = [
            {
                "content": result["content"],
                "filename": result["filename"],
                "collection": result.get("collection"),
                "similarity": result["similarity"],
                "page_numbers": result.get("pages"),
                "title": result.get("title"),
            }
            for result in all_results
        ]

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        collection_info = (
            f"across all collections" if is_chatall else f"from {collection_name}"
        )
        base_prompt = f"You are a document assistant answering from documents {collection_info}. Use ONLY context information."

        system_prompt = build_system_prompt_with_history(
            base_prompt, conversation_history, context
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        response_stream = chat_service.llm.stream(messages)

        for chunk in response_stream:
            if hasattr(chunk, "content") and chunk.content:
                yield f"data: {json.dumps({'type': 'content', 'content': chunk.content})}\n\n"

    except Exception as e:
        logger.error(f"Error in handle_content_search: {e}", exc_info=True)
        raise
