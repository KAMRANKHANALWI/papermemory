"""
config.py - Central configuration for PaperMemory
All app settings read from .env via AppConfig.
Eval-specific settings in EvalConfig.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class AppConfig:
    # ----------------------------------------
    # Paths
    # ----------------------------------------
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
    PDF_STORAGE_PATH: str = os.getenv("PDF_STORAGE_PATH", "data/pdfs")

    # Pages store: data/pages_store/{collection}/{filename}.json
    PAGES_STORE_PATH: str = os.getenv("PAGES_STORE_PATH", "data/pages_store")

    # ----------------------------------------
    # Embedding
    # ----------------------------------------
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ----------------------------------------
    # Chunking
    # ----------------------------------------
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # ----------------------------------------
    # Retrieval
    # ----------------------------------------
    # Final pages/chunks sent to LLM as context
    TOP_K: int = int(os.getenv("TOP_K", "10"))

    # Per-collection results in chatall mode
    TOP_K_CHATALL: int = int(os.getenv("TOP_K_CHATALL", "4"))

    # How many chunks to pull before reranking
    RERANKING_SAMPLE_SIZE: int = int(os.getenv("RERANKING_SAMPLE_SIZE", "30"))

    # ----------------------------------------
    # Reranker
    # Options: "llm_local" | "llm_api" | "cross_encoder" | "none"
    #
    # "none"         → skip reranking, use top-K by similarity score
    # "cross_encoder"→ local cross-encoder model (no API cost)
    # "llm_local"    → uses your Ollama model to score chunks
    # "llm_api"      → uses Gemini or Groq API to score chunks
    # ----------------------------------------
    RERANKER_TYPE: str = os.getenv("RERANKER_TYPE", "llm_local")

    # Used when RERANKER_TYPE=llm_api
    # Options: "gemini" | "groq"
    RERANKER_API_PROVIDER: str = os.getenv("RERANKER_API_PROVIDER", "gemini")

    # Used when RERANKER_TYPE=cross_encoder
    CROSS_ENCODER_MODEL: str = os.getenv(
        "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # ----------------------------------------
    # Memory / Conversation history
    # ----------------------------------------
    MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "10"))

    # ----------------------------------------
    # LLM (main answering model) — read by LLMFactory
    # ----------------------------------------
    USE_LOCAL_LLM: bool = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    DEFAULT_MODEL_PROVIDER: str = os.getenv("DEFAULT_MODEL_PROVIDER", "gemini")

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")


class EvalConfig:
    # ----------------------------------------
    # Dataset paths
    # ----------------------------------------
    EVAL_DIR = BASE_DIR / "eval"
    DATASETS_DIR = EVAL_DIR / "datasets"
    OUTPUT_DIR = EVAL_DIR / "results"

    OPEN_ENDED_CSV = DATASETS_DIR / "full_open_ended.csv"
    MCQ_CSV = DATASETS_DIR / "full_mcq.csv"

    # ----------------------------------------
    # Vector DB — references AppConfig so they stay in sync
    # ----------------------------------------
    DB_PATH = BASE_DIR.parent / AppConfig.CHROMA_DB_PATH
    COLLECTION_NAME = "gut_microbiome"
    EMBEDDING_MODEL = f"sentence-transformers/{AppConfig.EMBEDDING_MODEL}"
    TOP_K = AppConfig.TOP_K

    # ----------------------------------------
    # RAGAS Eval LLM
    # ----------------------------------------
    GROQ_API_KEY = AppConfig.GROQ_API_KEY
    GROQ_MODEL = AppConfig.GROQ_MODEL

    # ----------------------------------------
    # RAGAS Embeddings
    # ----------------------------------------
    RAGAS_EMBEDDING_MODEL = f"sentence-transformers/{AppConfig.EMBEDDING_MODEL}"