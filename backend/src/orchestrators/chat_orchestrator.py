import json
import logging
from typing import List, Dict
from fastapi import Request

from src.services.retrieval_service import RetrievalService

from src.prompts import (
    get_scientific_rag_prompt_v2,
    get_selected_pdf_prompt
)

logger = logging.getLogger(__name__)


class ChatOrchestrator:

    def __init__(
        self,
        chat_service,
        memory_service=None,
        metadata_service=None,
        file_search_service=None,
        pdf_selection_service=None,
        collection_manager=None,
        query_classifier=None,
    ):
        self.chat_service = chat_service
        self.memory_service = memory_service
        self.metadata_service = metadata_service
        self.file_search_service = file_search_service
        self.pdf_selection_service = pdf_selection_service
        self.collection_manager = collection_manager
        self.query_classifier = query_classifier

    @staticmethod
    async def stream_llm_response(response_stream, request: Request = None):

        async for chunk in response_stream:

            if request and await request.is_disconnected():
                logger.info("Client disconnected — stopping LLM stream")
                break

            if hasattr(chunk, "content") and chunk.content:
                yield (
                    f"data: {json.dumps({'type': 'content', 'content': chunk.content})}\n\n"
                )

    @staticmethod
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

    @staticmethod
    def build_system_prompt_with_history(
        base_prompt: str,
        conversation_history: List[Dict],
        context: str,
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

    async def handle_metadata_query(
        self,
        message,
        classification,
        collection_name,
        is_chatall,
        get_vectorstore,
    ):
        if is_chatall:
            vectorstores = {
                col.name: get_vectorstore(col.name)
                for col in self.chat_service.chroma_client.list_collections()
            }

            all_pdfs, stats = self.metadata_service.get_chatall_collection_pdfs(
                vectorstores
            )

            filenames = []

            for pdfs in all_pdfs.values():
                filenames.extend(pdfs)

            filenames.sort()

        else:
            if not collection_name:
                raise ValueError("Collection name required")

            vectorstore = get_vectorstore(collection_name)

            filenames, stats = self.metadata_service.get_single_collection_pdfs(
                vectorstore
            )

        response = self.metadata_service.build_metadata_response(
            classification=classification,
            query=message,
            filenames=filenames,
            stats=stats,
        )

        yield (
            f"data: {json.dumps({'type': 'content', 'content': response})}\n\n"
        )

    async def handle_list_collections(
        self,
        message,
        conversation_history,
        collection_prompt,
        request=None,
    ):
        collections = self.chat_service.chroma_client.list_collections()

        lines = [
            f"• {col.name} "
            f"({self.chat_service.chroma_client.get_collection(col.name).count()} chunks)"
            for col in collections
        ]

        context = (
            f"AVAILABLE COLLECTIONS:\n"
            f"Total: {len(collections)}\n\n" + "\n".join(lines)
        )

        system_prompt = self.build_system_prompt_with_history(
            collection_prompt,
            conversation_history,
            context,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        async for chunk in self.stream_llm_response(
            self.chat_service.llm.astream(messages),
            request,
        ):
            yield chunk

    async def handle_file_specific_search(
        self,
        message,
        filename,
        collection_name,
        is_chatall,
        conversation_history,
        get_vectorstore,
        file_specific_prompt,
        scientific_prompt,
        content_search_handler,
        request=None,
    ):

        filename = filename.strip() if filename else filename

        if is_chatall:
            vectorstores = {
                col.name: get_vectorstore(col.name)
                for col in self.chat_service.chroma_client.list_collections()
            }

            context, results, found, _ = (
                self.file_search_service.search_specific_file_chatall(
                    vectorstores,
                    filename,
                    message,
                    num_results=10,
                )
            )

        else:
            if not collection_name:
                raise ValueError("Collection name required")

            vectorstore = get_vectorstore(collection_name)

            context, results, found = self.file_search_service.search_specific_file(
                vectorstore,
                filename,
                message,
                num_results=10,
                collection_name=collection_name,
            )

        if not found:
            not_found_msg = f'File "{filename}" not found. Searching all documents...'

            yield (
                f"data: {json.dumps({'type': 'content', 'content': not_found_msg})}\n\n"
            )

            async for event in content_search_handler(
                message=message,
                collection_name=collection_name,
                is_chatall=is_chatall,
                conversation_history=conversation_history,
                get_vectorstore=get_vectorstore,
                scientific_prompt=get_scientific_rag_prompt_v2(),
                request=request,
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

        yield (f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n")

        system_prompt = self.build_system_prompt_with_history(
            file_specific_prompt,
            conversation_history,
            context,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        async for chunk in self.stream_llm_response(
            self.chat_service.llm.astream(messages),
            request,
        ):
            yield chunk

    async def handle_content_search(
        self,
        message,
        collection_name,
        is_chatall,
        conversation_history,
        get_vectorstore,
        scientific_prompt,
        request=None,
    ):
        context, all_results = RetrievalService.retrieve_content(
            message=message,
            is_chatall=is_chatall,
            collection_name=collection_name,
            chroma_client=self.chat_service.chroma_client,
            get_vectorstore=get_vectorstore,
            logger=logger,
        )

        yield (f"data: {json.dumps({'type': 'sources', 'sources': all_results})}\n\n")

        system_prompt = self.build_system_prompt_with_history(
            scientific_prompt,
            conversation_history,
            context,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        async for chunk in self.stream_llm_response(
            self.chat_service.llm.astream(messages),
            request,
        ):
            yield chunk
            
    async def handle_selected_pdf_chat(
        self,
        session_id,
        query,
        chat_id,
        num_results,
        request=None,
    ):

        session = self.pdf_selection_service.get_or_create_session(
            session_id
        )

        if session.get_selection_count() == 0:

            yield (
                f"data: "
                f"{json.dumps({'type': 'error', 'message': 'Please select PDFs first'})}"
                f"\n\n"
            )

            return

        selection_data = session.to_dict()

        classification, _ = self.query_classifier.classify_query(
            query,
            is_chatall_mode=False,
        )

        # -------------------------
        # Metadata Queries
        # -------------------------

        if classification in ["count_pdfs", "list_pdfs"]:

            filenames = [
                pdf["filename"]
                for pdf in selection_data["selected_pdfs"]
            ]

            stats = {
                "total_pdfs": len(filenames)
            }

            response = self.metadata_service.build_metadata_response(
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

        # -------------------------
        # Retrieval
        # -------------------------

        all_collections = (
            self.collection_manager.get_all_collections_vectorstores(
                self.chat_service.embedding_model
            )
        )

        if not all_collections:

            yield (
                f"data: "
                f"{json.dumps({'type': 'error', 'message': 'No collections available'})}"
                f"\n\n"
            )

            return

        try:

            context, results, total_results = (
                self.pdf_selection_service.search_selected_pdfs(
                    session_id=session_id,
                    query=query,
                    all_collections=all_collections,
                    num_results=num_results,
                )
            )

        except Exception as e:

            yield (
                f"data: "
                f"{json.dumps({'type': 'error', 'message': f'Search failed: {str(e)}'})}"
                f"\n\n"
            )

            return

        if total_results == 0:

            yield (
                f"data: "
                f"{json.dumps({'type': 'content', 'content': 'No relevant information found in the selected PDFs.'})}"
                f"\n\n"
            )

            yield (
                f"data: "
                f"{json.dumps({'type': 'end'})}"
                f"\n\n"
            )

            return

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

        yield (
            f"data: "
            f"{json.dumps({'type': 'sources', 'sources': sources_data})}"
            f"\n\n"
        )

        system_prompt = get_selected_pdf_prompt()

        messages = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\nContext:\n{context}",
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        full_response = ""

        async for event in self.stream_llm_response(
            self.chat_service.llm.astream(messages),
            request,
        ):

            full_response += self.collect_content_from_event(
                event
            )

            yield event

        if self.memory_service:

            self.memory_service.add_message(
                chat_id,
                "user",
                query,
            )

            self.memory_service.add_message(
                chat_id,
                "assistant",
                full_response,
            )

        yield (
            f"data: "
            f"{json.dumps({'type': 'end'})}"
            f"\n\n"
        )
