"""
LLM-based query classification service
"""
import re
from typing import Tuple, Optional
from langchain_core.messages import HumanMessage


class QueryClassifier:
    """Uses LLM to classify query intent with few-shot prompting."""
    
    CLASSIFICATION_PROMPT = """You are a query classifier for a document Q&A system. Classify the user's query into ONE of these categories:

CATEGORIES:
1. "list_pdfs" - User wants to see list of available documents/PDFs
2. "count_pdfs" - User wants to know how many documents are available
3. "list_collections" - User wants to see available collections (only in multi-collection mode)
4. "file_specific_search" - User asks about a SPECIFIC PDF file by name
5. "content_search" - User wants to search/ask about document content (default)

EXAMPLES:

User: "List all the PDFs you have"
Classification: list_pdfs

User: "What documents are available?"
Classification: list_pdfs

User: "Show me the files"
Classification: list_pdfs

User: "How many PDFs do you have?"
Classification: count_pdfs

User: "What collections are there?"
Classification: list_collections

User: "What does protein_interaction.pdf say about databases?"
Classification: file_specific_search

User: "Summarize research_paper.pdf"
Classification: file_specific_search

User: "Tell me about chapter 3 in machine_learning.pdf"
Classification: file_specific_search

User: "What is in climate_report.pdf?"
Classification: file_specific_search

User: "Search in document_analysis.pdf for methods"
Classification: file_specific_search

User: "What does the research paper say about climate change?"
Classification: content_search

User: "Tell me about machine learning"
Classification: content_search

User: "What is mentioned about quantum physics?"
Classification: content_search

User: "Explain the methodology used"
Classification: content_search

RULES:
- Only respond with ONE word: list_pdfs, count_pdfs, list_collections, file_specific_search, or content_search
- Use "file_specific_search" ONLY when user mentions a specific filename (with .pdf extension)
- When in doubt between file_specific_search and content_search, use "content_search"
- Focus on the USER'S INTENT, not exact keywords

Now classify this query:
User: "{query}"
Classification:"""

    def __init__(self, llm):
        """
        Initialize with a language model.
        
        Args:
            llm: Language model instance (Groq or Gemini)
        """
        self.llm = llm
    
    def classify_query(
        self, 
        query: str, 
        is_chatall_mode: bool = False
    ) -> Tuple[str, Optional[str]]:
        """
        Classify user query using LLM with few-shot prompting.
        
        Args:
            query: User's query text
            is_chatall_mode: Whether in ChatALL mode (enables list_collections)
        
        Returns:
            Tuple of (classification, optional_filename)
            - classification: One of the valid query types
            - optional_filename: Extracted filename if file_specific_search, else None
        """
        try:
            # Format prompt with user query
            prompt = self.CLASSIFICATION_PROMPT.format(query=query)
            
            # Get classification from LLM
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            # Extract classification
            classification = response.content.strip().lower()
            
            # Validate and normalize classification
            valid_types = [
                'list_pdfs', 
                'count_pdfs', 
                'list_collections', 
                'file_specific_search', 
                'content_search'
            ]
            
            # Handle variations in response
            if 'file' in classification and 'specific' in classification:
                classification = 'file_specific_search'
            elif 'list' in classification and 'collection' in classification:
                classification = 'list_collections'
            elif 'list' in classification or 'show' in classification:
                classification = 'list_pdfs'
            elif 'count' in classification or 'how many' in classification:
                classification = 'count_pdfs'
            
            # Final validation
            if classification not in valid_types:
                classification = 'content_search'
            
            # Disable list_collections if not in ChatALL mode
            if classification == 'list_collections' and not is_chatall_mode:
                classification = 'list_pdfs'
            
            # Extract filename if file_specific_search
            filename = None
            if classification == 'file_specific_search':
                filename = self._extract_filename_from_query(query)
                # If no filename found, fall back to content_search
                if not filename:
                    classification = 'content_search'
            
            return classification, filename
            
        except Exception as e:
            print(f"Classification error: {e}")
            # Default to content search on error
            return 'content_search', None
    
    def _extract_filename_from_query(self, query: str) -> Optional[str]:
        """
        Extract PDF filename from query.
        
        Args:
            query: User query
            
        Returns:
            Extracted filename or None
        """
        # Pattern to match .pdf filenames
        # Matches: filename.pdf, my_file.pdf, research-paper-2024.pdf, etc.
        pattern = r'([a-zA-Z0-9_\-]+\.pdf)'
        matches = re.findall(pattern, query, re.IGNORECASE)
        
        if matches:
            # Return first match
            return matches[0]
        
        return None