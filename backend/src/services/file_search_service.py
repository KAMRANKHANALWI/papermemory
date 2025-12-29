"""
File-specific search service
"""
from typing import Tuple, List, Dict, Optional, Any
from langchain_chroma import Chroma


class FileSearchService:
    """Service for searching within specific PDF files"""
    
    @staticmethod
    def search_specific_file(
        vectorstore: Chroma,
        filename: str,
        query: str,
        num_results: int = 25
    ) -> Tuple[str, List[Dict], bool]:
        """
        Search for content within a specific PDF file.
        
        Args:
            vectorstore: The vector store to search
            filename: Name of the specific PDF file
            query: User's query
            num_results: Number of chunks to retrieve
        
        Returns:
            Tuple of (context, search_results, file_found)
        """
        try:
            # Search with filename filter
            results = vectorstore.similarity_search_with_score(
                query,
                k=num_results,
                filter={"filename": filename}
            )
            
            if not results:
                return "", [], False
            
            context_parts = []
            search_results = []
            
            for doc, score in results:
                doc_filename = doc.metadata.get("filename", "unknown")
                page_numbers = doc.metadata.get("page_numbers", "[]")
                title = doc.metadata.get("title", "No Title")
                
                source_parts = [doc_filename]
                if page_numbers != "[]":
                    page_nums = (
                        page_numbers.strip("[]").replace("'", "").replace(" ", "").split(",")
                    )
                    if page_nums and page_nums[0]:
                        source_parts.append(f"p. {', '.join(page_nums)}")
                
                source = f"Source: {' - '.join(source_parts)}"
                if title != "No Title":
                    source += f"\nTitle: {title}"
                
                context_parts.append(f"{doc.page_content}\n{source}")
                search_results.append({
                    "content": doc.page_content,
                    "filename": doc_filename,
                    "title": title,
                    "pages": page_numbers,
                    "similarity": round(1 - score, 4),
                })
            
            context = "\n\n".join(context_parts)
            return context, search_results, True
            
        except Exception as e:
            print(f"Error in file-specific search: {e}")
            return "", [], False
    
    @staticmethod
    def search_specific_file_chatall(
        all_collections: Dict[str, Any],
        filename: str,
        query: str,
        num_results: int = 25
    ) -> Tuple[str, List[Dict], bool, Optional[str]]:
        """
        Search for content within a specific PDF file across all collections.
        
        Args:
            all_collections: Dict of collection_name -> vectorstore
            filename: Name of the PDF file
            query: Search query
            num_results: Number of results
        
        Returns:
            Tuple of (context, search_results, file_found, collection_name)
        """
        try:
            for collection_name, vectorstore in all_collections.items():
                # Try to find the file in this collection
                context, search_results, found = FileSearchService.search_specific_file(
                    vectorstore, filename, query, num_results
                )
                
                if found:
                    # Add collection info to results
                    for result in search_results:
                        result["collection"] = collection_name
                    return context, search_results, True, collection_name
            
            # File not found in any collection
            return "", [], False, None
            
        except Exception as e:
            print(f"Error in ChatALL file-specific search: {e}")
            return "", [], False, None
