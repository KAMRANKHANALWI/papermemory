# ============================================================================
# PDF STORAGE UTILITY - Save and Serve Original PDFs
# File: backend/src/utils/pdf_storage.py
# ============================================================================

import os
import shutil
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class PDFStorageManager:
    """
    Manages storage of original PDF files on disk.
    PDFs are organized by collection: data/pdfs/{collection_name}/{filename.pdf}
    """
    
    def __init__(self, base_path: str = "data/pdfs"):
        """
        Initialize PDF storage manager.
        
        Args:
            base_path: Base directory for storing PDFs
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 PDF Storage initialized at: {self.base_path.absolute()}")
    
    def get_collection_path(self, collection_name: str) -> Path:
        """Get the directory path for a collection."""
        return self.base_path / collection_name
    
    def get_pdf_path(self, collection_name: str, filename: str) -> Path:
        """Get the full path to a specific PDF."""
        return self.get_collection_path(collection_name) / filename
    
    def save_pdf(self, collection_name: str, filename: str, file_data: bytes) -> bool:
        """
        Save a PDF file to disk.
        
        Args:
            collection_name: Name of the collection
            filename: Name of the PDF file
            file_data: Binary PDF data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create collection directory if it doesn't exist
            collection_path = self.get_collection_path(collection_name)
            collection_path.mkdir(parents=True, exist_ok=True)
            
            # Save PDF file
            pdf_path = self.get_pdf_path(collection_name, filename)
            
            with open(pdf_path, 'wb') as f:
                f.write(file_data)
            
            logger.info(f"✅ Saved PDF: {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save PDF {filename}: {e}")
            return False
    
    def pdf_exists(self, collection_name: str, filename: str) -> bool:
        """Check if a PDF file exists."""
        pdf_path = self.get_pdf_path(collection_name, filename)
        return pdf_path.exists()
    
    def get_pdf_data(self, collection_name: str, filename: str) -> Optional[bytes]:
        """
        Read PDF file data.
        
        Returns:
            PDF binary data or None if not found
        """
        try:
            pdf_path = self.get_pdf_path(collection_name, filename)
            
            if not pdf_path.exists():
                logger.warning(f"⚠️ PDF not found: {pdf_path}")
                return None
            
            with open(pdf_path, 'rb') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"❌ Failed to read PDF {filename}: {e}")
            return None
    
    def delete_pdf(self, collection_name: str, filename: str) -> bool:
        """
        Delete a specific PDF file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            pdf_path = self.get_pdf_path(collection_name, filename)
            
            if pdf_path.exists():
                pdf_path.unlink()
                logger.info(f"🗑️ Deleted PDF: {pdf_path}")
                return True
            else:
                logger.warning(f"⚠️ PDF not found: {pdf_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete PDF {filename}: {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete an entire collection folder and all its PDFs.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            collection_path = self.get_collection_path(collection_name)
            
            if collection_path.exists():
                shutil.rmtree(collection_path)
                logger.info(f"🗑️ Deleted collection folder: {collection_path}")
                return True
            else:
                logger.warning(f"⚠️ Collection folder not found: {collection_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete collection {collection_name}: {e}")
            return False
    
    def list_pdfs(self, collection_name: str) -> List[str]:
        """
        List all PDF files in a collection.
        
        Returns:
            List of PDF filenames
        """
        try:
            collection_path = self.get_collection_path(collection_name)
            
            if not collection_path.exists():
                return []
            
            # Get all PDF files
            pdf_files = [
                f.name for f in collection_path.iterdir() 
                if f.is_file() and f.suffix.lower() == '.pdf'
            ]
            
            return sorted(pdf_files)
            
        except Exception as e:
            logger.error(f"❌ Failed to list PDFs in {collection_name}: {e}")
            return []
    
    def get_pdf_size(self, collection_name: str, filename: str) -> int:
        """
        Get the size of a PDF file in bytes.
        
        Returns:
            File size in bytes, or 0 if not found
        """
        try:
            pdf_path = self.get_pdf_path(collection_name, filename)
            if pdf_path.exists():
                return pdf_path.stat().st_size
            return 0
        except:
            return 0
    
    def get_storage_stats(self) -> dict:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with stats about stored PDFs
        """
        try:
            total_pdfs = 0
            total_size = 0
            collections = {}
            
            for collection_dir in self.base_path.iterdir():
                if collection_dir.is_dir():
                    pdfs = self.list_pdfs(collection_dir.name)
                    count = len(pdfs)
                    size = sum(
                        self.get_pdf_size(collection_dir.name, pdf) 
                        for pdf in pdfs
                    )
                    
                    collections[collection_dir.name] = {
                        "pdf_count": count,
                        "total_size_mb": round(size / (1024 * 1024), 2)
                    }
                    
                    total_pdfs += count
                    total_size += size
            
            return {
                "total_collections": len(collections),
                "total_pdfs": total_pdfs,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "collections": collections
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get storage stats: {e}")
            return {
                "total_collections": 0,
                "total_pdfs": 0,
                "total_size_mb": 0,
                "collections": {}
            }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
"""
# In app.py, initialize the manager:
from src.utils.pdf_storage import PDFStorageManager

pdf_storage = PDFStorageManager(base_path="data/pdfs")

# When uploading PDFs:
for file in files:
    file_data = await file.read()
    
    # Save to disk
    pdf_storage.save_pdf(
        collection_name="research",
        filename=file.filename,
        file_data=file_data
    )
    
    # Also process for ChromaDB (existing code)
    # ...

# To serve a PDF:
@app.get("/api/pdfs/{collection_name}/{filename}")
async def get_pdf(collection_name: str, filename: str):
    pdf_data = pdf_storage.get_pdf_data(collection_name, filename)
    
    if not pdf_data:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return Response(
        content=pdf_data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"'
        }
    )

# To delete a PDF:
@app.delete("/api/pdfs/{collection_name}/{filename}")
async def delete_pdf(collection_name: str, filename: str):
    # Delete from disk
    pdf_storage.delete_pdf(collection_name, filename)
    
    # Delete from ChromaDB (existing code)
    # ...
    
    return {"message": "PDF deleted successfully"}

# When deleting a collection:
@app.delete("/api/collections/{collection_name}")
async def delete_collection(collection_name: str):
    # Delete collection folder from disk
    pdf_storage.delete_collection(collection_name)
    
    # Delete from ChromaDB (existing code)
    # ...
    
    return {"message": "Collection deleted"}
"""
