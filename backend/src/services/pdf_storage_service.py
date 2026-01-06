"""
PDF Storage Service - Stores original PDF files on disk
"""
import os
import shutil
from pathlib import Path
from typing import Optional
from fastapi import UploadFile


class PDFStorageService:
    """
    Service for storing and retrieving original PDF files.
    PDFs are stored in: data/pdfs/{collection_name}/{filename}.pdf
    """
    
    def __init__(self, base_path: str = "data/pdfs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ PDF Storage initialized at: {self.base_path}")
    
    def save_pdf(
        self, 
        pdf_content: bytes,
        collection_name: str, 
        filename: str
    ) -> str:
        """
        Save PDF bytes to disk.
        
        Args:
            pdf_content: PDF file content as bytes
            collection_name: Name of the collection
            filename: PDF filename
            
        Returns:
            str: Full path where PDF was saved
        """
        # Create collection directory
        collection_dir = self.base_path / collection_name
        collection_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure .pdf extension
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        # Full path to save
        file_path = collection_dir / filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(pdf_content)
        
        print(f"💾 Saved PDF: {file_path}")
        return str(file_path)
    
    def get_pdf_path(self, collection_name: str, filename: str) -> Optional[Path]:
        """
        Get path to a stored PDF.
        
        Returns:
            Path object if PDF exists, None otherwise
        """
        file_path = self.base_path / collection_name / filename
        
        if file_path.exists():
            return file_path
        
        return None
    
    def delete_pdf(self, collection_name: str, filename: str) -> bool:
        """
        Delete a PDF from storage.
        
        Returns:
            bool: True if deleted, False if not found
        """
        file_path = self.base_path / collection_name / filename
        
        if file_path.exists():
            file_path.unlink()
            print(f"🗑️ Deleted PDF: {file_path}")
            return True
        
        return False
    
    def rename_pdf(
        self, 
        collection_name: str, 
        old_filename: str, 
        new_filename: str
    ) -> bool:
        """
        Rename a PDF in storage.
        
        Returns:
            bool: True if renamed, False if not found
        """
        old_path = self.base_path / collection_name / old_filename
        new_path = self.base_path / collection_name / new_filename
        
        if old_path.exists():
            old_path.rename(new_path)
            print(f"✏️ Renamed PDF: {old_filename} → {new_filename}")
            return True
        
        return False
    
    def delete_collection_pdfs(self, collection_name: str) -> bool:
        """
        Delete all PDFs in a collection.
        
        Returns:
            bool: True if collection folder deleted, False if not found
        """
        collection_dir = self.base_path / collection_name
        
        if collection_dir.exists():
            shutil.rmtree(collection_dir)
            print(f"🗑️ Deleted all PDFs in collection: {collection_name}")
            return True
        
        return False
    
    def list_pdfs(self, collection_name: str) -> list:
        """
        List all PDFs in a collection.
        
        Returns:
            list: List of PDF filenames
        """
        collection_dir = self.base_path / collection_name
        
        if not collection_dir.exists():
            return []
        
        return [f.name for f in collection_dir.glob("*.pdf")]
