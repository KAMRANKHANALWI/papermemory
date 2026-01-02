"""
PDF Selection Service - Manage selected PDFs across collections for targeted querying
"""
from typing import Dict, List, Set, Optional, Tuple
from langchain_chroma import Chroma
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class SelectedPDF:
    """Represents a selected PDF from a collection"""
    filename: str
    collection_name: str
    title: Optional[str] = None
    pages: int = 0
    chunks: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "filename": self.filename,
            "collection_name": self.collection_name,
            "title": self.title,
            "pages": self.pages,
            "chunks": self.chunks
        }


@dataclass
class PDFSelectionSession:
    """Manages a session of selected PDFs"""
    session_id: str
    selected_pdfs: List[SelectedPDF] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_pdf(self, pdf: SelectedPDF) -> bool:
        """Add a PDF to selection"""
        # Check if already selected
        if not self._is_selected(pdf.filename, pdf.collection_name):
            self.selected_pdfs.append(pdf)
            self.updated_at = datetime.now().isoformat()
            return True
        return False
    
    def remove_pdf(self, filename: str, collection_name: str) -> bool:
        """Remove a PDF from selection"""
        initial_length = len(self.selected_pdfs)
        self.selected_pdfs = [
            pdf for pdf in self.selected_pdfs 
            if not (pdf.filename == filename and pdf.collection_name == collection_name)
        ]
        if len(self.selected_pdfs) < initial_length:
            self.updated_at = datetime.now().isoformat()
            return True
        return False
    
    def clear_all(self):
        """Clear all selected PDFs"""
        self.selected_pdfs = []
        self.updated_at = datetime.now().isoformat()
    
    def _is_selected(self, filename: str, collection_name: str) -> bool:
        """Check if PDF is already selected"""
        return any(
            pdf.filename == filename and pdf.collection_name == collection_name
            for pdf in self.selected_pdfs
        )
    
    def get_selection_count(self) -> int:
        """Get total number of selected PDFs"""
        return len(self.selected_pdfs)
    
    def get_collections_involved(self) -> Set[str]:
        """Get set of collection names that have selected PDFs"""
        return {pdf.collection_name for pdf in self.selected_pdfs}
    
    def get_pdfs_by_collection(self) -> Dict[str, List[str]]:
        """Group selected PDFs by collection"""
        result = {}
        for pdf in self.selected_pdfs:
            if pdf.collection_name not in result:
                result[pdf.collection_name] = []
            result[pdf.collection_name].append(pdf.filename)
        return result
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "selected_pdfs": [pdf.to_dict() for pdf in self.selected_pdfs],
            "total_selected": self.get_selection_count(),
            "collections_involved": list(self.get_collections_involved()),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class PDFSelectionService:
    """Service for managing PDF selection across collections"""
    
    def __init__(self):
        """Initialize the PDF selection service"""
        self.sessions: Dict[str, PDFSelectionSession] = {}
    
    def create_session(self, session_id: str) -> PDFSelectionSession:
        """Create a new PDF selection session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = PDFSelectionSession(session_id=session_id)
        return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> Optional[PDFSelectionSession]:
        """Get an existing session"""
        return self.sessions.get(session_id)
    
    def get_or_create_session(self, session_id: str) -> PDFSelectionSession:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            return self.create_session(session_id)
        return self.sessions[session_id]
    
    def select_pdf(
        self,
        session_id: str,
        filename: str,
        collection_name: str,
        vectorstore: Chroma
    ) -> Tuple[bool, str]:
        """
        Add a PDF to selection with metadata extraction.
        
        Args:
            session_id: Session identifier
            filename: PDF filename
            collection_name: Collection name
            vectorstore: Vectorstore for the collection
            
        Returns:
            Tuple of (success, message)
        """
        try:
            session = self.get_or_create_session(session_id)
            
            # Extract PDF metadata from vectorstore
            collection = vectorstore._collection
            result = collection.get(
                where={"filename": filename},
                include=["metadatas"],
                limit=1000
            )
            
            if not result.get("metadatas"):
                return False, f"PDF '{filename}' not found in collection '{collection_name}'"
            
            # Gather metadata
            title = None
            pages_set = set()
            chunks = len(result["metadatas"])
            
            for metadata in result["metadatas"]:
                if not title and metadata.get("title"):
                    title = metadata["title"]
                
                pages_str = metadata.get("page_numbers", "[]")
                pages_list = (
                    pages_str.strip("[]")
                    .replace("'", "")
                    .replace(" ", "")
                    .split(",")
                )
                for page in pages_list:
                    if page and page.isdigit():
                        pages_set.add(int(page))
            
            # Create SelectedPDF object
            selected_pdf = SelectedPDF(
                filename=filename,
                collection_name=collection_name,
                title=title or "No Title",
                pages=len(pages_set),
                chunks=chunks
            )
            
            # Add to session
            if session.add_pdf(selected_pdf):
                return True, f"Added '{filename}' from '{collection_name}' to selection"
            else:
                return False, f"PDF '{filename}' is already selected"
            
        except Exception as e:
            return False, f"Error selecting PDF: {str(e)}"
    
    def deselect_pdf(
        self,
        session_id: str,
        filename: str,
        collection_name: str
    ) -> Tuple[bool, str]:
        """
        Remove a PDF from selection.
        
        Args:
            session_id: Session identifier
            filename: PDF filename
            collection_name: Collection name
            
        Returns:
            Tuple of (success, message)
        """
        session = self.get_session(session_id)
        if not session:
            return False, "Session not found"
        
        if session.remove_pdf(filename, collection_name):
            return True, f"Removed '{filename}' from '{collection_name}'"
        else:
            return False, f"PDF '{filename}' not found in selection"
    
    def clear_selection(self, session_id: str) -> Tuple[bool, str]:
        """Clear all selected PDFs from session"""
        session = self.get_session(session_id)
        if not session:
            return False, "Session not found"
        
        session.clear_all()
        return True, "Selection cleared"
    
    def get_selected_pdfs(self, session_id: str) -> Optional[Dict]:
        """Get all selected PDFs for a session"""
        session = self.get_session(session_id)
        if not session:
            return None
        return session.to_dict()
    
    def search_selected_pdfs(
        self,
        session_id: str,
        query: str,
        all_collections: Dict[str, Chroma],
        num_results: int = 25
    ) -> Tuple[str, List[Dict], int]:
        """
        Search only within selected PDFs across collections.
        
        Args:
            session_id: Session identifier
            query: Search query
            all_collections: Dict of collection_name -> vectorstore
            num_results: Number of results per PDF
            
        Returns:
            Tuple of (context, search_results, total_results)
        """
        session = self.get_session(session_id)
        if not session or session.get_selection_count() == 0:
            return "", [], 0
        
        all_results = []
        context_parts = []
        
        # Group selected PDFs by collection for efficient searching
        pdfs_by_collection = session.get_pdfs_by_collection()
        
        for collection_name, filenames in pdfs_by_collection.items():
            if collection_name not in all_collections:
                continue
            
            vectorstore = all_collections[collection_name]
            
            # Search each PDF in this collection
            for filename in filenames:
                try:
                    results = vectorstore.similarity_search_with_score(
                        query,
                        k=num_results,
                        filter={"filename": filename}
                    )
                    
                    for doc, score in results:
                        doc_filename = doc.metadata.get("filename", "unknown")
                        page_numbers = doc.metadata.get("page_numbers", "[]")
                        title = doc.metadata.get("title", "No Title")
                        
                        # Format source
                        source_parts = [f"{collection_name}/{doc_filename}"]
                        if page_numbers != "[]":
                            page_nums = (
                                page_numbers.strip("[]")
                                .replace("'", "")
                                .replace(" ", "")
                                .split(",")
                            )
                            if page_nums and page_nums[0]:
                                source_parts.append(f"p. {', '.join(page_nums)}")
                        
                        source = f"Source: {' - '.join(source_parts)}"
                        if title != "No Title":
                            source += f"\nTitle: {title}"
                        
                        context_parts.append(f"{doc.page_content}\n{source}")
                        
                        all_results.append({
                            "content": doc.page_content,
                            "filename": doc_filename,
                            "collection": collection_name,
                            "title": title,
                            "pages": page_numbers,
                            "similarity": round(1 - score, 4),
                        })
                
                except Exception as e:
                    print(f"Error searching {filename} in {collection_name}: {e}")
                    continue
        
        # Sort by similarity
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Limit total results
        all_results = all_results[:num_results * 2]
        
        context = "\n\n".join(context_parts)
        return context, all_results, len(all_results)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a selection session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def export_selection(self, session_id: str) -> Optional[str]:
        """Export selection as JSON string"""
        session = self.get_session(session_id)
        if not session:
            return None
        return json.dumps(session.to_dict(), indent=2)
    
    def import_selection(
        self,
        session_id: str,
        selection_json: str
    ) -> Tuple[bool, str]:
        """Import selection from JSON string"""
        try:
            data = json.loads(selection_json)
            session = self.get_or_create_session(session_id)
            
            for pdf_data in data.get("selected_pdfs", []):
                pdf = SelectedPDF(**pdf_data)
                session.add_pdf(pdf)
            
            return True, f"Imported {len(data.get('selected_pdfs', []))} PDFs"
        except Exception as e:
            return False, f"Error importing selection: {str(e)}"
