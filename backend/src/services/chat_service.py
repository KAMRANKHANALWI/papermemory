from typing import Dict, Any
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from src.llm_factory import LLMFactory
import chromadb
from dotenv import load_dotenv
from src.config import AppConfig

load_dotenv()


class ChatService:
    def __init__(self, chroma_client=None):
        self.llm = LLMFactory.create()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=AppConfig.EMBEDDING_MODEL
        )
        self.chroma_client = chroma_client or chromadb.PersistentClient(
            path=AppConfig.CHROMA_DB_PATH
        )

    def get_model_info(self) -> Dict[str, Any]:
        if AppConfig.USE_LOCAL_LLM:
            return {
                "provider": "ollama",
                "model": AppConfig.OLLAMA_MODEL,
                "base_url": AppConfig.OLLAMA_BASE_URL,
                "is_local": True,
            }

        if AppConfig.DEFAULT_MODEL_PROVIDER == "gemini" and AppConfig.GOOGLE_API_KEY:
            return {
                "provider": "gemini",
                "model": AppConfig.GEMINI_MODEL,
                "is_local": False,
            }

        return {
            "provider": "groq",
            "model": AppConfig.GROQ_MODEL,
            "is_local": False,
        }
