"""
Pydantic models for API requests
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class QueryClassificationRequest(BaseModel):
    """Request model for query classification"""
    query: str = Field(..., description="User query to classify")
    is_chatall_mode: bool = Field(False, description="Whether in ChatALL mode")


class GenerateNameRequest(BaseModel):
    """Request model for generating collection name"""
    filenames: List[str] = Field(..., description="List of PDF filenames")
    upload_type: str = Field("files", description="Type of upload: files, folder")


class ValidateNameRequest(BaseModel):
    """Request model for validating collection name"""
    name: str = Field(..., description="Collection name to validate")


class RenameCollectionRequest(BaseModel):
    """Request model for renaming collection"""
    old_name: str = Field(..., description="Current collection name")
    new_name: str = Field(..., description="New collection name")


class RenamePDFRequest(BaseModel):
    """Request model for renaming PDF in collection"""
    collection_name: str = Field(..., description="Collection name")
    old_filename: str = Field(..., description="Current PDF filename")
    new_filename: str = Field(..., description="New PDF filename")


class AddMemoryRequest(BaseModel):
    """Request model for adding message to conversation memory"""
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    collection_name: Optional[str] = Field(None, description="Associated collection")


class FileSearchRequest(BaseModel):
    """Request model for file-specific search"""
    filename: str = Field(..., description="PDF filename to search")
    query: str = Field(..., description="Search query")
    collection_name: Optional[str] = Field(None, description="Specific collection to search in")
    num_results: int = Field(25, description="Number of results to return")
