import os
import json
import pandas as pd
import ast
import re
from pathlib import Path
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
)

from src.config import EvalConfig

config = EvalConfig()

# ---------------------------------------
# Paths
# ---------------------------------------
INPUT_CSV = Path(config.OUTPUT_DIR) / "rag_outputs_mcq.csv"
OUTPUT_CSV = Path(config.OUTPUT_DIR) / "mcq_ragas_eval_results.csv"
CHECKPOINT = Path("src/eval/checkpoints/eval_mcq_ragas_checkpoint.json")

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# Checkpoint helpers
# ---------------------------------------
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

# ---------------------------------------
# Helpers
# ---------------------------------------
def extract_explanation(text: str) -> str:
    """
    Extract explanation from MCQ answer:
    - Removes leading A/B/C/D
    - Keeps only explanation text
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()

    text = re.sub(
        r"^(ANSWER\s*[:\-]?\s*)?([ABCD])[\.\)\:\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


# ---------------------------------------
# Load data
# ---------------------------------------
df = pd.read_csv(INPUT_CSV)
print(f"✅ Loaded {len(df)} MCQ samples")

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

# ---------------------------------------
# Groq Eval LLM
# ---------------------------------------
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

# ---------------------------------------
# Metrics
# ---------------------------------------
faithfulness = Faithfulness(llm=eval_llm)
ctx_precision = ContextPrecision(llm=eval_llm)
ctx_recall = ContextRecall(llm=eval_llm)

# ---------------------------------------
# Evaluate 
# ---------------------------------------
for idx, row in df.iterrows():

    if idx <= start_idx:
        continue

    print(f"Evaluating MCQ explanation {idx + 1}/{len(df)}")

    try:
        contexts = ast.literal_eval(row["contexts"])

        explanation = extract_explanation(row["model_answer"])

        result_row = {
            "id": row.get("id", idx + 1),
            "question": row["question"],
            "faithfulness": faithfulness.score(
                user_input=row["question"],
                response=explanation,
                retrieved_contexts=contexts,
            ).value,
            "context_precision": ctx_precision.score(
                user_input=row["question"],
                reference=explanation,
                retrieved_contexts=contexts,
            ).value,
            "context_recall": ctx_recall.score(
                user_input=row["question"],
                reference=explanation,
                retrieved_contexts=contexts,
            ).value,
            "difficulty": row.get("difficulty"),
            "category": row.get("category"),
        }

        append_row(result_row)
        save_checkpoint(idx)

    except Exception as e:
        print(f"⚠️ Stopped safely at row {idx}: {e}")
        break

# ---------------------------------------
# Summary 
# ---------------------------------------
if OUTPUT_CSV.exists():
    out_df = pd.read_csv(OUTPUT_CSV)
    print("✅ MCQ RAGAS evaluation complete")
    print("\nAverage scores:")
    print(out_df.mean(numeric_only=True))
else:
    print("⚠️ No successful rows were evaluated (CSV not created)")
