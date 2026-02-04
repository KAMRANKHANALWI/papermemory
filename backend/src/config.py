# config.py - Eval Config File
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Absolute path to backend/src
BASE_DIR = Path(__file__).resolve().parent

class EvalConfig:
    # --------------------
    # Dataset
    # --------------------
    EVAL_DIR = BASE_DIR / "eval"
    DATASETS_DIR = EVAL_DIR / "datasets"
    OUTPUT_DIR = EVAL_DIR / "results"

    OPEN_ENDED_CSV = DATASETS_DIR / "full_open_ended.csv"
    MCQ_CSV = DATASETS_DIR / "full_mcq.csv"

    # --------------------
    # Vector DB
    # --------------------
    DB_PATH = BASE_DIR.parent / "data" / "chroma_db"
    COLLECTION_NAME = "gut_microbiome"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    TOP_K = 10

    # --------------------
    # RAGAS Eval LLM
    # --------------------
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # --------------------
    # RAGAS Embeddings
    # --------------------
    RAGAS_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
