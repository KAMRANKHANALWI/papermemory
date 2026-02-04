import os
import json
import pandas as pd
import numpy as np
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
        quoting=1,  # csv.QUOTE_ALL
        escapechar="\\",
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

local_client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    timeout=300.0,
    max_retries=5,
)

# ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1-70b-16k:latest")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
eval_llm = llm_factory(
    model=ollama_model,
    provider="openai",
    client=local_client,
    max_tokens=8192,  # Override RAGAS default of 1024
    temperature=0,
)

print(f"🖥️ Using Local Ollama {ollama_model} Model for Eval")

# ---------------------------------------
# Groq Eval LLM
# ---------------------------------------
# groq_client = AsyncOpenAI(
#     api_key=config.GROQ_API_KEY,
#     base_url="https://api.groq.com/openai/v1",
# )

# eval_llm = llm_factory(
#     model=config.GROQ_MODEL,
#     provider="openai",
#     client=groq_client,
# )

# print(f"☁️ Using {config.GROQ_MODEL} via GROQ API for Eval")

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

    errors = []
    contexts = None

    # -------- Safe context parsing --------
    try:
        raw_ctx = row.get("contexts")

        if pd.isna(raw_ctx):
            contexts = None
        else:
            raw_ctx = str(raw_ctx).strip()
            if raw_ctx == "" or raw_ctx == "[]":
                contexts = None
            else:
                parsed = ast.literal_eval(raw_ctx)
                if isinstance(parsed, list) and len(parsed) > 0:
                    contexts = parsed
                else:
                    contexts = None
    except Exception:
        contexts = None

    if contexts is None:
        errors.append("retrieved_contexts is missing")

    # -------- Explanation extraction --------
    explanation = extract_explanation(row.get("model_answer"))

    if not explanation.strip():
        errors.append("mcq_explanation is missing")

    # -------- Invalid sample → log & continue --------
    if errors:
        append_row(
            {
                "id": row.get("id", idx + 1),
                "question": row.get("question"),
                "faithfulness": np.nan,
                "context_precision": np.nan,
                "context_recall": np.nan,
                "difficulty": row.get("difficulty"),
                "category": row.get("category"),
                "error": "; ".join(errors),
            }
        )
        save_checkpoint(idx)
        print(f"⚠️ Row {idx}: {'; '.join(errors)}")
        continue

    # -------- Normal RAGAS evaluation --------
    try:
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
            "error": None,
        }

        append_row(result_row)
        save_checkpoint(idx)

    except Exception as e:
        append_row(
            {
                "id": row.get("id", idx + 1),
                "question": row.get("question"),
                "faithfulness": np.nan,
                "context_precision": np.nan,
                "context_recall": np.nan,
                "difficulty": row.get("difficulty"),
                "category": row.get("category"),
                "error": str(e),
            }
        )
        save_checkpoint(idx)
        print(f"⚠️ Row {idx}: {e}")
        continue

# ---------------------------------------
# Summary
# ---------------------------------------
if OUTPUT_CSV.exists():
    out_df = pd.read_csv(
        OUTPUT_CSV,
        engine="python",
        on_bad_lines="skip",
    )

    total = len(out_df)
    failed = out_df["error"].notna().sum()
    success = total - failed

    print("✅ MCQ RAGAS evaluation complete\n")

    print("📊 Evaluation Summary")
    print("---------------------")
    print(f"Total samples        : {total}")
    print(f"Successfully scored  : {success}")
    print(f"Failed samples       : {failed}")
    print(f"Success rate         : {(success / total) * 100:.2f}%")

    # Retrieval coverage
    retrieval_failures = (
        out_df["error"].str.contains("retrieved_contexts", na=False).sum()
    )

    print(f"Retrieval failures   : {retrieval_failures}")
    print(f"Retrieval coverage   : {((total - retrieval_failures) / total) * 100:.2f}%")

    # Average scores on valid rows only
    valid_df = out_df[out_df["error"].isna()]

    if not valid_df.empty:
        print("\n📈 Average Scores (valid samples only):")
        print(valid_df[["faithfulness", "context_precision", "context_recall"]].mean())
    else:
        print("\n⚠️ No valid samples available for averaging")

    # Optional: error breakdown
    print("\n🧯 Failure breakdown:")
    print(out_df["error"].value_counts(dropna=True))

else:
    print("⚠️ No evaluation output found (CSV not created)")
