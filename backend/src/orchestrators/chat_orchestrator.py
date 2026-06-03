import json
import logging
from typing import List, Dict
from fastapi import Request

logger = logging.getLogger(__name__)


class ChatOrchestrator:

    def __init__(
        self,
        chat_service,
        memory_service=None,
        metadata_service=None,
        file_search_service=None,
    ):
        self.chat_service = chat_service
        self.memory_service = memory_service
        self.metadata_service = metadata_service
        self.file_search_service = file_search_service

    @staticmethod
    async def stream_llm_response(
        response_stream,
        request: Request = None,
    ):
        async for chunk in response_stream:

            if request and await request.is_disconnected():
                logger.info("Client disconnected — stopping LLM stream")
                break

            if hasattr(chunk, "content") and chunk.content:
                yield (
                    f"data: "
                    f"{json.dumps({'type': 'content', 'content': chunk.content})}"
                    f"\n\n"
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
        request=None,
    ):
        if is_chatall:

            vectorstores = {
                col.name: get_vectorstore(col.name)
                for col in self.chat_service.chroma_client.list_collections()
            }

            all_pdfs, stats = (
                self.metadata_service.get_chatall_collection_pdfs(
                    vectorstores
                )
            )

            filenames = []

            for pdfs in all_pdfs.values():
                filenames.extend(pdfs)

            filenames.sort()

        else:

            if not collection_name:
                raise ValueError(
                    "Collection name required"
                )

            vectorstore = get_vectorstore(
                collection_name
            )

            filenames, stats = (
                self.metadata_service.get_single_collection_pdfs(
                    vectorstore
                )
            )

        direct_response = (
            self.metadata_service.build_metadata_response(
                classification=classification,
                query=message,
                filenames=filenames,
                stats=stats,
            )
        )

        if direct_response:

            yield (
                f"data: "
                f"{json.dumps({'type': 'content', 'content': direct_response})}"
                f"\n\n"
            )

            return
        
    async def handle_file_specific_search(
        self,
        message,
        filename,
        collection_name,
        is_chatall,
        conversation_history,
        get_vectorstore,
        file_specific_prompt,
        content_search_handler,
        retrieval_service,
        reranker,
        request=None,
    ):
        context, sources, found = (
            retrieval_service.retrieve_file_content(
                message=message,
                filename=filename,
                collection_name=collection_name,
                is_chatall=is_chatall,
                chroma_client=self.chat_service.chroma_client,
                get_vectorstore=get_vectorstore,
                file_search_service=self.file_search_service,
                reranker=reranker,
                logger=logger,
            )
        )

        if not found:

            message_text = (
                f'File "{filename}" not found. '
                f"Searching all documents..."
            )

            yield (
                f"data: "
                f"{json.dumps({'type': 'content', 'content': message_text})}"
                f"\n\n"
            )

            async for event in content_search_handler(
                message,
                collection_name,
                is_chatall,
                conversation_history,
                request,
            ):
                yield event

            return

        yield (
            f"data: "
            f"{json.dumps({'type': 'sources', 'sources': sources})}"
            f"\n\n"
        )

        system_prompt = self.build_system_prompt_with_history(
            file_specific_prompt,
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

        async for chunk in self.stream_llm_response(
            self.chat_service.llm.astream(messages),
            request,
        ):
            yield chunk
            
    async def handle_list_collections(
        self,
        message,
        conversation_history,
        collection_prompt,
        request=None,
    ):
        collections = self.chat_service.chroma_client.list_collections()

        lines = []

        for col in collections:
            count = self.chat_service.chroma_client.get_collection(
                col.name
            ).count()

            lines.append(
                f"• {col.name} ({count} chunks)"
            )

        context = (
            f"AVAILABLE COLLECTIONS:\n"
            f"Total: {len(collections)}\n\n"
            + "\n".join(lines)
        )

        system_prompt = self.build_system_prompt_with_history(
            collection_prompt,
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

        async for chunk in self.stream_llm_response(
            self.chat_service.llm.astream(messages),
            request,
        ):
            yield chunk