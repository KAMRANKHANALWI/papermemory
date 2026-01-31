# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class EvalConfig:
    # --------------------
    # Dataset
    # --------------------
    OPEN_ENDED_CSV = "./sample_open_ended.csv"
    MCQ_CSV = "./sample_mcq.csv"
    OUTPUT_DIR = "evaluation_results"

    # --------------------
    # Vector DB
    # --------------------
    DB_PATH = "data_raw/chroma_db"
    COLLECTION_NAME = "gut_microbiome"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    TOP_K = 5

    # --------------------
    # RAG Generation LLM
    # --------------------
    USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    TEMPERATURE = 0.1

    # --------------------
    # RAGAS Eval LLM (Groq)
    # --------------------
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # --------------------
    # RAGAS Embeddings
    # --------------------
    RAGAS_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
