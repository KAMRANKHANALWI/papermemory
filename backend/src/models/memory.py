"""
Models for conversation memory management
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class ConversationMessage(BaseModel):
    """Single message in conversation"""
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    collection_name: Optional[str] = Field(None, description="Associated collection")


class ConversationMemory(BaseModel):
    """Complete conversation memory for a chat session"""
    chat_id: str = Field(..., description="Unique chat identifier")
    messages: List[ConversationMessage] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    def add_message(self, role: str, content: str, collection_name: Optional[str] = None):
        """Add a message to the conversation"""
        message = ConversationMessage(
            role=role,
            content=content,
            collection_name=collection_name
        )
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()
    
    def get_formatted_history(self, max_messages: int = 10) -> List[Dict]:
        """Get formatted conversation history for LLM context"""
        recent_messages = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent_messages
        ]
    
    def clear(self):
        """Clear all messages"""
        self.messages = []
        self.updated_at = datetime.now().isoformat()