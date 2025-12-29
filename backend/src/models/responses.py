"""
Pydantic models for API responses
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class QueryClassificationResponse(BaseModel):
    """Response model for query classification"""
    classification: str = Field(..., description="Query type: list_pdfs, count_pdfs, list_collections, file_specific_search, content_search")
    filename: Optional[str] = Field(None, description="Extracted filename if file_specific_search")
    confidence: float = Field(1.0, description="Classification confidence")


class PDFDetail(BaseModel):
    """Model for PDF details"""
    filename: str
    title: str
    chunks: int
    pages: int
    page_range: str


class CollectionPDFsResponse(BaseModel):
    """Response model for listing PDFs in a collection"""
    collection_name: str
    total_pdfs: int
    total_chunks: int
    pdfs: List[PDFDetail]


class AllCollectionsPDFsResponse(BaseModel):
    """Response model for listing all PDFs across collections"""
    total_collections: int
    total_pdfs: int
    total_chunks: int
    collections: List[CollectionPDFsResponse]


class CollectionStatsResponse(BaseModel):
    """Response model for detailed collection statistics"""
    name: str
    total_pdfs: int
    total_chunks: int
    pdfs: List[PDFDetail]
    created_at: Optional[str] = None
    modified_at: Optional[str] = None


class GenerateNameResponse(BaseModel):
    """Response model for generated collection name"""
    suggested_name: str
    is_valid: bool
    validation_message: Optional[str] = None


class ValidateNameResponse(BaseModel):
    """Response model for name validation"""
    is_valid: bool
    validated_name: Optional[str] = None
    message: str


class SearchResult(BaseModel):
    """Model for search result"""
    content: str
    filename: str
    collection: Optional[str] = None
    similarity: float
    page_numbers: Optional[str] = None
    title: Optional[str] = None


class FileSearchResponse(BaseModel):
    """Response model for file-specific search"""
    filename: str
    collection_name: Optional[str] = None
    found: bool
    num_results: int
    results: List[SearchResult]


class MemoryMessage(BaseModel):
    """Model for conversation memory message"""
    role: str
    content: str
    timestamp: str
    collection_name: Optional[str] = None


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history"""
    chat_id: str
    message_count: int
    messages: List[MemoryMessage]


class MemorySummaryResponse(BaseModel):
    """Response model for conversation summary"""
    chat_id: str
    summary: str
    total_messages: int


class OperationResponse(BaseModel):
    """Generic response for operations"""
    status: str  # success, error
    message: str
    data: Optional[Dict[str, Any]] = None
