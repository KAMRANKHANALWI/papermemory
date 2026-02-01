import json
import ast
import pandas as pd
from pathlib import Path
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
)

from src.config import EvalConfig

# -------------------------------------------------
# Config & Paths
# -------------------------------------------------
config = EvalConfig()

INPUT_CSV = Path(config.OUTPUT_DIR) / "rag_outputs_open_ended.csv"
OUTPUT_CSV = Path(config.OUTPUT_DIR) / "open_ended_eval_results.csv"
CHECKPOINT = Path("src/eval/checkpoints/open_ended_eval_checkpoint.json")

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------
def load_checkpoint() -> int:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text()).get("last_index", -1)
    return -1


def save_checkpoint(idx: int):
    CHECKPOINT.write_text(json.dumps({"last_index": idx}))


def append_row(row: dict):
    df = pd.DataFrame([row])
    df.to_csv(
        OUTPUT_CSV,
        mode="a",
        header=not OUTPUT_CSV.exists(),
        index=False,
    )

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print(f"✅ Loaded {len(df)} evaluated RAG samples")

start_idx = load_checkpoint()
print(f"▶️ Resuming from index {start_idx + 1}")

# ---------------------------------------
# Ollama Local LLM
# ---------------------------------------

# local_client = AsyncOpenAI(
#     api_key="ollama",
#     base_url="http://localhost:11434/v1",
# )

# ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
# eval_llm = llm_factory(model=ollama_model, provider="openai", client=local_client)

# print(f"🖥️ Using Local Ollama {ollama_model} Model for Eval")

# -------------------------------------------------
# Groq Eval LLM
# -------------------------------------------------
groq_client = AsyncOpenAI(
    api_key=config.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

eval_llm = llm_factory(
    model=config.GROQ_MODEL,
    provider="openai",
    client=groq_client,
)

print(f"☁️ Using {config.GROQ_MODEL} via GROQ API for Eval")

# ---------------------------------------
# Embedding Model
# ---------------------------------------

embeddings = embedding_factory(
    "huggingface",
    model=config.RAGAS_EMBEDDING_MODEL,
)

# -------------------------------------------------
# Metrics
# -------------------------------------------------
faithfulness = Faithfulness(llm=eval_llm)
relevancy = AnswerRelevancy(llm=eval_llm, embeddings=embeddings)
correctness = AnswerCorrectness(llm=eval_llm, embeddings=embeddings)
ctx_precision = ContextPrecision(llm=eval_llm)
ctx_recall = ContextRecall(llm=eval_llm)

# -------------------------------------------------
# Evaluation Loop 
# -------------------------------------------------
for idx, row in df.iterrows():

    if idx <= start_idx:
        continue

    print(f"Evaluating {idx + 1}/{len(df)}")

    try:
        contexts = ast.literal_eval(row["contexts"])

        result = {
            "id": row.get("id"),
            "question": row["question"],
            "faithfulness": faithfulness.score(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=contexts,
            ).value,
            "answer_relevancy": relevancy.score(
                user_input=row["question"],
                response=row["answer"],
            ).value,
            "answer_correctness": correctness.score(
                user_input=row["question"],
                response=row["answer"],
                reference=row["reference"],
            ).value,
            "context_precision": ctx_precision.score(
                user_input=row["question"],
                reference=row["reference"],
                retrieved_contexts=contexts,
            ).value,
            "context_recall": ctx_recall.score(
                user_input=row["question"],
                reference=row["reference"],
                retrieved_contexts=contexts,
            ).value,
        }

        append_row(result)
        save_checkpoint(idx)

    except Exception as e:
        print(f"⚠️ Skipping row {idx}: {e}")
        break   # stop safely

# -------------------------------------------------
# Summary (optional)
# -------------------------------------------------
if OUTPUT_CSV.exists():
    out_df = pd.read_csv(OUTPUT_CSV)
    print("\n📊 Average Scores:")
    print(out_df.mean(numeric_only=True))

print("✅ Open-ended evaluation completed (safe mode)")
