import json
import logging
import time

from fastapi import Request

from src.services.dependencies import orchestrator

logger = logging.getLogger(__name__)


async def chat_with_selected_pdfs(
    session_id: str,
    query: str,
    chat_id: str | None,
    num_results: int,
    request: Request = None,
):
    try:

        current_chat_id = chat_id or f"chat_{session_id}_{int(time.time())}"

        yield (
            f"data: "
            f"{json.dumps({'type': 'chat_id', 'chat_id': current_chat_id})}"
            f"\n\n"
        )

        async for chunk in orchestrator.handle_selected_pdf_chat(
            session_id=session_id,
            query=query,
            chat_id=current_chat_id,
            num_results=num_results,
            request=request,
        ):
            yield chunk

    except Exception as e:

        logger.error(f"Unexpected error in selection chat: {e}")

        yield (f"data: " f"{json.dumps({'type': 'error', 'message': str(e)})}" f"\n\n")
