# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class AppConfig:
    # Paths
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
    PDF_STORAGE_PATH: str = os.getenv("PDF_STORAGE_PATH", "data/pdfs")
    PAGES_STORE_PATH: str = os.getenv("PAGES_STORE_PATH", "data/pages_store")

    # Embedding
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "10"))
    TOP_K_CHATALL: int = int(os.getenv("TOP_K_CHATALL", "4"))

    # Memory
    MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "10"))

    # LLM
    USE_LOCAL_LLM: bool = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    DEFAULT_MODEL_PROVIDER: str = os.getenv("DEFAULT_MODEL_PROVIDER", "gemini")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")


class EvalConfig:
    EVAL_DIR = BASE_DIR / "eval"
    DATASETS_DIR = EVAL_DIR / "Eval_Dataset"
    OUTPUT_DIR = EVAL_DIR / "Eval_Results"
    OPEN_ENDED_CSV = DATASETS_DIR / "gut_microbiome_open_ended.csv"
    MCQ_CSV = DATASETS_DIR / "gut_microbiome_mcq.csv"
    DB_PATH = BASE_DIR.parent / AppConfig.CHROMA_DB_PATH
    COLLECTION_NAME = "gut_microbiome"
    EMBEDDING_MODEL = f"sentence-transformers/{AppConfig.EMBEDDING_MODEL}"
    TOP_K = AppConfig.TOP_K
    GROQ_API_KEY = AppConfig.GROQ_API_KEY
    GROQ_MODEL = AppConfig.GROQ_MODEL
    RAGAS_EMBEDDING_MODEL = f"sentence-transformers/{AppConfig.EMBEDDING_MODEL}"