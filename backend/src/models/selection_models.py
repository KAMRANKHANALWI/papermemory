"""
Request and Response models for PDF Selection API
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict


# ==================== REQUESTS ====================

class SelectPDFRequest(BaseModel):
    """Request to select a PDF"""
    filename: str = Field(..., description="PDF filename to select")
    collection_name: str = Field(..., description="Collection containing the PDF")


class DeselectPDFRequest(BaseModel):
    """Request to deselect a PDF"""
    filename: str = Field(..., description="PDF filename to deselect")
    collection_name: str = Field(..., description="Collection containing the PDF")


class BatchSelectPDFsRequest(BaseModel):
    """Request to select multiple PDFs at once"""
    selections: List[Dict[str, str]] = Field(
        ..., 
        description="List of {filename, collection_name} pairs"
    )


class SelectedPDFsSearchRequest(BaseModel):
    """Request to search within selected PDFs"""
    query: str = Field(..., description="Search query")
    num_results: int = Field(25, description="Number of results to return")


# ==================== RESPONSES ====================

class SelectedPDFInfo(BaseModel):
    """Information about a selected PDF"""
    filename: str
    collection_name: str
    title: Optional[str] = None
    pages: int = 0
    chunks: int = 0


class PDFSelectionResponse(BaseModel):
    """Response for PDF selection operations"""
    success: bool
    message: str
    total_selected: int = 0
    selected_pdfs: List[SelectedPDFInfo] = []


class SelectionSessionResponse(BaseModel):
    """Response containing full selection session info"""
    session_id: str
    total_selected: int
    collections_involved: List[str]
    selected_pdfs: List[SelectedPDFInfo]
    created_at: str
    updated_at: str


class SelectedPDFsSearchResponse(BaseModel):
    """Response for searching within selected PDFs"""
    query: str
    total_results: int
    total_selected_pdfs: int
    collections_searched: List[str]
    results: List[Dict]  # Search results with similarity scores


class SelectionStatsResponse(BaseModel):
    """Statistics about current selection"""
    total_selected: int
    collections_involved: List[str]
    pdfs_by_collection: Dict[str, int]  # collection_name -> count
    total_chunks: int
    total_pages: int
