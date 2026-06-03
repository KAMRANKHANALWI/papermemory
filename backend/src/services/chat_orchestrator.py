import json
import logging
from typing import List, Dict
from fastapi import Request

logger = logging.getLogger(__name__)


class ChatOrchestrator:

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