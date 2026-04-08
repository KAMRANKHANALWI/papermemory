"""
Models package initialization
"""
from .requests import (
    QueryClassificationRequest,
    GenerateNameRequest,
    ValidateNameRequest,
    RenameCollectionRequest,
    RenamePDFRequest,
    AddMemoryRequest,
    FileSearchRequest,
    ChatRequest,
)

from .responses import (
    QueryClassificationResponse,
    PDFDetail,
    CollectionPDFsResponse,
    AllCollectionsPDFsResponse,
    CollectionStatsResponse,
    GenerateNameResponse,
    ValidateNameResponse,
    SearchResult,
    FileSearchResponse,
    MemoryMessage,
    ConversationHistoryResponse,
    MemorySummaryResponse,
    OperationResponse,
)

from .memory import (
    ConversationMessage,
    ConversationMemory,
)

__all__ = [
    # Requests
    "QueryClassificationRequest",
    "GenerateNameRequest",
    "ValidateNameRequest",
    "RenameCollectionRequest",
    "RenamePDFRequest",
    "AddMemoryRequest",
    "FileSearchRequest",
    "ChatRequest",
    # Responses
    "QueryClassificationResponse",
    "PDFDetail",
    "CollectionPDFsResponse",
    "AllCollectionsPDFsResponse",
    "CollectionStatsResponse",
    "GenerateNameResponse",
    "ValidateNameResponse",
    "SearchResult",
    "FileSearchResponse",
    "MemoryMessage",
    "ConversationHistoryResponse",
    "MemorySummaryResponse",
    "OperationResponse",
    # Memory
    "ConversationMessage",
    "ConversationMemory",
]