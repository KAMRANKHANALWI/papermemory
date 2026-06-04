import chromadb
from src.config import AppConfig

from src.services.chat_service import ChatService
from src.services.memory_service import MemoryService
from src.services.metadata_service import MetadataService
from src.services.file_search_service import FileSearchService
from src.services.query_classifier import QueryClassifier
from src.services.reranker_factory import get_reranker
from src.services.collection_manager import CollectionManager
from src.services.pdf_selection_service import PDFSelectionService
from src.services.pdf_storage_service import PDFStorageService
from src.services.document_processor import DocumentProcessor

from src.orchestrators.chat_orchestrator import ChatOrchestrator

# ONE shared chroma client
chroma_client = chromadb.PersistentClient(path=AppConfig.CHROMA_DB_PATH)
chat_service = ChatService(chroma_client=chroma_client)
memory_service = MemoryService()
metadata_service = MetadataService()
file_search_service = FileSearchService()
reranker = get_reranker()
query_classifier = QueryClassifier(chat_service.llm)
pdf_storage = PDFStorageService(base_path="data/pdfs")
collection_manager = CollectionManager(chroma_client=chroma_client)
document_processor = DocumentProcessor(
    pdf_storage=pdf_storage, chroma_client=chroma_client
)
pdf_selection_service = PDFSelectionService()


orchestrator = ChatOrchestrator(
    chat_service=chat_service,
    memory_service=memory_service,
    metadata_service=metadata_service,
    file_search_service=file_search_service,
    pdf_selection_service=pdf_selection_service,
    collection_manager=collection_manager,
    query_classifier=query_classifier,
    reranker=reranker,
)
