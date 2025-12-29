import json
import uuid
from typing import Optional
from src.services.chat_service import ChatService

chat_service = ChatService()

async def generate_chat_response(
    message: str, 
    collection_name: Optional[str], 
    mode: str, 
    chat_id: Optional[str] = None
):
    """Generate streaming chat response"""
    
    # Generate chat ID if new conversation
    if chat_id is None:
        chat_id = str(uuid.uuid4())
        yield f'data: {{"type": "chat_id", "chat_id": "{chat_id}"}}\n\n'
    
    try:
        # Search documents
        if mode == "single" and collection_name:
            context, search_results = chat_service.search_single_collection(message, collection_name)
        else:  # chatall mode
            context, search_results = chat_service.search_all_collections(message)
        
        # Send search results
        if search_results:
            yield f'data: {{"type": "search_results", "count": {len(search_results)}}}\n\n'
        
        # Generate and stream response
        async for chunk in chat_service.generate_response(message, context):
            safe_content = chunk.replace('"', '\\"').replace('\n', '\\n')
            yield f'data: {{"type": "content", "content": "{safe_content}"}}\n\n'
        
        # Send sources
        if search_results:
            sources_json = json.dumps(search_results)
            yield f'data: {{"type": "sources", "sources": {sources_json}}}\n\n'
        
        yield f'data: {{"type": "end"}}\n\n'
        
    except Exception as e:
        yield f'data: {{"type": "error", "message": "{str(e)}"}}\n\n'