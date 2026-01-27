"""
Document processor with rich metadata extraction
"""
import os
import tempfile
import chromadb
from typing import List
from fastapi import UploadFile
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.utils.pdf_processor import process_pdf_with_pymupdf, chunk_text_content


class DocumentProcessor:
    def __init__(self, pdf_storage=None):  
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        self.pdf_storage = pdf_storage  # NEW: Store reference to pdf_storage
        
    async def process_files(self, files: List[UploadFile], collection_name: str):
        """Process uploaded PDF files with rich metadata"""
        all_documents = []
        processed_files = []
        
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                continue
            
            # NEW: Read file content once
            content = await file.read()
            
            # NEW: Save original PDF to disk (if pdf_storage is available)
            if self.pdf_storage:
                try:
                    self.pdf_storage.save_pdf(content, collection_name, file.filename)
                    print(f"✅ Saved PDF to storage: {file.filename}")
                except Exception as e:
                    print(f"⚠️ Failed to save PDF {file.filename}: {e}")
            
            # Save temporarily for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(content)  # Use content we already read
                tmp_path = tmp_file.name
            
            # Process PDF
            try:
                documents = self._process_single_pdf(tmp_path, file.filename)
                all_documents.extend(documents)
                processed_files.append(file.filename)
            finally:
                os.unlink(tmp_path)
        
        # Create vector store
        if all_documents:
            vectorstore = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embedding_model,
                client=self.chroma_client,
                collection_name=collection_name,
                persist_directory="data/chroma_db"
            )
            
        return {
            "files_processed": len(processed_files),
            "chunks_created": len(all_documents),
            "collection": collection_name,
            "processed_files": processed_files
        }
    
    def _process_single_pdf(self, file_path: str, filename: str) -> List[Document]:
        """
        Process a single PDF file into document chunks with rich metadata.
        
        This includes:
        - Page numbers for each chunk
        - Document title from headings
        - All extracted headings
        - Character count
        - Full file path
        """
        pages_data = process_pdf_with_pymupdf(file_path)
        if not pages_data:
            return []
        
        # Combine all text
        full_text = "\n\n".join([page["text"] for page in pages_data if page["text"]])
        
        # Extract overall headings
        all_headings = []
        for page in pages_data:
            all_headings.extend(page["headings"])
        
        # Get unique headings (preserve order)
        unique_headings = list(dict.fromkeys(all_headings))[:5]
        
        # Create chunks
        text_chunks = chunk_text_content(full_text)
        
        documents = []
        for chunk_id, chunk_text in enumerate(text_chunks):
            # Find which pages this chunk likely belongs to
            chunk_pages = self._find_chunk_pages(chunk_text, full_text, pages_data)
            
            # Create document with rich metadata
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "filename": filename,
                    "filepath": file_path,
                    "chunk_id": chunk_id,
                    "total_chunks": len(text_chunks),
                    "page_numbers": str(chunk_pages) if chunk_pages else "[]",
                    "title": unique_headings[0] if unique_headings else "No Title",
                    "all_headings": str(unique_headings),
                    "text_length": len(chunk_text),
                    "source_type": "pdf"
                }
            )
            documents.append(doc)
            
        return documents
    
    def _find_chunk_pages(
        self, 
        chunk_text: str, 
        full_text: str, 
        pages_data: List[dict]
    ) -> List[int]:
        """
        Determine which pages a chunk belongs to.
        
        Args:
            chunk_text: The text chunk
            full_text: Complete document text
            pages_data: List of page data dicts
            
        Returns:
            List of page numbers (1-indexed)
        """
        chunk_pages = []
        chunk_pos = full_text.find(chunk_text)
        
        if chunk_pos != -1:
            char_count = 0
            for page in pages_data:
                page_end = char_count + len(page["text"])
                
                # Check if chunk starts in this page
                if char_count <= chunk_pos < page_end:
                    chunk_pages.append(page["page_num"])
                    
                    # Check if chunk extends to next pages
                    chunk_end = chunk_pos + len(chunk_text)
                    if chunk_end > page_end:
                        # Add next pages until chunk ends
                        remaining_pages = [
                            p["page_num"] 
                            for p in pages_data 
                            if p["page_num"] > page["page_num"]
                        ]
                        for next_page_num in remaining_pages:
                            chunk_pages.append(next_page_num)
                            # Simple heuristic: assume chunk spans at most 3 pages
                            if len(chunk_pages) >= 3:
                                break
                    break
                
                char_count = page_end
        
        return chunk_pages if chunk_pages else [1]  # Default to page 1 if not found
