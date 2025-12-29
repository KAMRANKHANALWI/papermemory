"""
Smart collection naming service
"""
import re
import os
from typing import List, Tuple
from langchain_core.messages import HumanMessage
from src.utils.validators import validate_collection_name
import chromadb


class NamingService:
    """Service for generating and validating collection names"""
    
    def __init__(self, llm, chroma_client: chromadb.PersistentClient):
        """
        Initialize naming service.
        
        Args:
            llm: Language model for name generation
            chroma_client: ChromaDB client for checking duplicates
        """
        self.llm = llm
        self.chroma_client = chroma_client
    
    def generate_smart_collection_name(
        self, 
        filenames: List[str], 
        upload_type: str = "files"
    ) -> str:
        """
        Generate intelligent collection name from filenames using LLM.
        
        Args:
            filenames: List of PDF filenames
            upload_type: Type of upload (files, folder, etc.)
            
        Returns:
            Generated and validated collection name
        """
        try:
            if not filenames:
                return self._generate_default_name()
            
            # Clean filenames for analysis
            clean_names = [os.path.splitext(f)[0] for f in filenames]
            
            # If single file, use its name
            if len(clean_names) == 1:
                base_name = self._clean_filename(clean_names[0])
                return self._ensure_unique_name(base_name)
            
            # For multiple files, find common theme
            base_name = self._find_common_theme(clean_names)
            
            if not base_name or len(base_name) < 3:
                # Use LLM to generate name
                base_name = self._llm_generate_name(clean_names)
            
            # Validate and ensure uniqueness
            return self._ensure_unique_name(base_name)
            
        except Exception as e:
            print(f"Error generating collection name: {e}")
            return self._generate_default_name()
    
    def _clean_filename(self, filename: str) -> str:
        """Clean filename for use as collection name"""
        # Remove common prefixes/suffixes
        cleaned = filename.lower()
        
        # Remove common words
        remove_words = ['document', 'pdf', 'file', 'the', 'a', 'an']
        for word in remove_words:
            cleaned = re.sub(rf'\b{word}\b', '', cleaned, flags=re.IGNORECASE)
        
        # Remove special characters except spaces, hyphens, underscores
        cleaned = re.sub(r'[^a-zA-Z0-9\s_-]', '', cleaned)
        
        # Replace multiple spaces/hyphens with single
        cleaned = re.sub(r'[\s_-]+', '_', cleaned)
        
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        
        return cleaned
    
    def _find_common_theme(self, filenames: List[str]) -> str:
        """Find common theme in multiple filenames"""
        if not filenames:
            return ""
        
        # Clean all filenames
        cleaned = [self._clean_filename(f) for f in filenames]
        
        # Find common prefix
        if len(cleaned) > 1:
            # Get words from first filename
            first_words = set(cleaned[0].split('_'))
            
            # Find common words
            common_words = first_words
            for name in cleaned[1:]:
                name_words = set(name.split('_'))
                common_words = common_words.intersection(name_words)
            
            if common_words:
                # Use common words
                theme = '_'.join(sorted(common_words)[:3])
                return theme
        
        # No common theme, use first file's name
        return cleaned[0][:30]
    
    def _llm_generate_name(self, filenames: List[str]) -> str:
        """Use LLM to generate a concise collection name"""
        try:
            prompt = f"""Generate a short, descriptive collection name (2-4 words) for these PDF files:

Files:
{chr(10).join(f"- {name}" for name in filenames[:10])}

Requirements:
- 2-4 words maximum
- Describes the common theme or topic
- Use underscores instead of spaces
- Lowercase only
- No special characters

Collection name:"""

            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            # Clean LLM response
            name = response.content.strip().lower()
            name = self._clean_filename(name)
            
            return name if name else "documents"
            
        except Exception as e:
            print(f"Error in LLM name generation: {e}")
            return "documents"
    
    def _generate_default_name(self) -> str:
        """Generate a default timestamped collection name"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"collection_{timestamp}"
    
    def _ensure_unique_name(self, base_name: str) -> str:
        """
        Ensure collection name is unique by appending number if needed.
        
        Args:
            base_name: Base collection name
            
        Returns:
            Unique collection name
        """
        # Validate the base name first
        is_valid, cleaned_name, _ = validate_collection_name(base_name)
        
        if not is_valid:
            cleaned_name = "collection"
        
        # Check if name exists
        existing_collections = [col.name for col in self.chroma_client.list_collections()]
        
        if cleaned_name not in existing_collections:
            return cleaned_name
        
        # Name exists, append number
        counter = 1
        while f"{cleaned_name}_{counter}" in existing_collections:
            counter += 1
        
        return f"{cleaned_name}_{counter}"
    
    def validate_name(self, name: str) -> Tuple[bool, str, str]:
        """
        Validate collection name.
        
        Args:
            name: Collection name to validate
            
        Returns:
            Tuple of (is_valid, cleaned_name, message)
        """
        return validate_collection_name(name)