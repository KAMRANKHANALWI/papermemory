"""
Conversation memory management service
"""
from typing import Dict, Optional, List
from src.models.memory import ConversationMemory, ConversationMessage
import json
import os


class MemoryService:
    """Service for managing conversation memory across chat sessions"""
    
    def __init__(self, storage_path: str = "data/memory"):
        """
        Initialize memory service.
        
        Args:
            storage_path: Path to store conversation memories
        """
        self.storage_path = storage_path
        self.memories: Dict[str, ConversationMemory] = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing memories
        self._load_memories()
    
    def get_or_create_memory(self, chat_id: str) -> ConversationMemory:
        """
        Get existing memory or create new one for chat session.
        
        Args:
            chat_id: Unique chat identifier
            
        Returns:
            ConversationMemory instance
        """
        if chat_id not in self.memories:
            self.memories[chat_id] = ConversationMemory(chat_id=chat_id)
            self._save_memory(chat_id)
        
        return self.memories[chat_id]
    
    def add_message(
        self, 
        chat_id: str, 
        role: str, 
        content: str,
        collection_name: Optional[str] = None
    ) -> ConversationMemory:
        """
        Add a message to conversation memory.
        
        Args:
            chat_id: Chat identifier
            role: Message role (user or assistant)
            content: Message content
            collection_name: Associated collection name
            
        Returns:
            Updated ConversationMemory
        """
        memory = self.get_or_create_memory(chat_id)
        memory.add_message(role, content, collection_name)
        self._save_memory(chat_id)
        return memory
    
    def get_memory(self, chat_id: str) -> Optional[ConversationMemory]:
        """
        Get conversation memory for a chat session.
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            ConversationMemory or None if not found
        """
        return self.memories.get(chat_id)
    
    def get_formatted_history(
        self, 
        chat_id: str, 
        max_messages: int = 10
    ) -> List[Dict]:
        """
        Get formatted conversation history for LLM context.
        
        Args:
            chat_id: Chat identifier
            max_messages: Maximum number of recent messages
            
        Returns:
            List of formatted messages
        """
        memory = self.get_memory(chat_id)
        if not memory:
            return []
        
        return memory.get_formatted_history(max_messages)
    
    def clear_memory(self, chat_id: str) -> bool:
        """
        Clear all messages in a conversation.
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            True if cleared successfully
        """
        memory = self.get_memory(chat_id)
        if memory:
            memory.clear()
            self._save_memory(chat_id)
            return True
        return False
    
    def delete_memory(self, chat_id: str) -> bool:
        """
        Delete entire conversation memory.
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            True if deleted successfully
        """
        if chat_id in self.memories:
            del self.memories[chat_id]
            
            # Delete file
            file_path = os.path.join(self.storage_path, f"{chat_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return True
        return False
    
    def get_summary(self, chat_id: str) -> Optional[str]:
        """
        Get a summary of the conversation.
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            Summary string or None
        """
        memory = self.get_memory(chat_id)
        if not memory or not memory.messages:
            return None
        
        total_messages = len(memory.messages)
        user_messages = sum(1 for msg in memory.messages if msg.role == "user")
        assistant_messages = sum(1 for msg in memory.messages if msg.role == "assistant")
        
        collections = set(
            msg.collection_name 
            for msg in memory.messages 
            if msg.collection_name
        )
        
        summary = f"Conversation with {total_messages} messages "
        summary += f"({user_messages} from user, {assistant_messages} from assistant)"
        
        if collections:
            summary += f". Collections discussed: {', '.join(collections)}"
        
        return summary
    
    def _save_memory(self, chat_id: str):
        """Save memory to disk"""
        memory = self.memories.get(chat_id)
        if not memory:
            return
        
        file_path = os.path.join(self.storage_path, f"{chat_id}.json")
        with open(file_path, 'w') as f:
            json.dump(memory.dict(), f, indent=2)
    
    def _load_memories(self):
        """Load all memories from disk"""
        if not os.path.exists(self.storage_path):
            return
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                file_path = os.path.join(self.storage_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        memory = ConversationMemory(**data)
                        self.memories[memory.chat_id] = memory
                except Exception as e:
                    print(f"Error loading memory {filename}: {e}")