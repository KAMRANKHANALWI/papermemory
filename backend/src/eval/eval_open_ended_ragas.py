import os
import json
import ast
import pandas as pd
import numpy as np
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

# -------------------------------------------------
# Groq Eval LLM
# -------------------------------------------------
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

    error_msg = None
    contexts = None
    answer = row.get("answer")

    # -------- Safe context parsing --------
    try:
        if pd.notna(row.get("contexts")) and str(row["contexts"]).strip():
            contexts = ast.literal_eval(row["contexts"])
            if not isinstance(contexts, list) or len(contexts) == 0:
                contexts = None
        else:
            contexts = None
    except Exception:
        contexts = None

    if contexts is None:
        error_msg = "retrieved_contexts is missing"

    if pd.isna(answer) or not str(answer).strip():
        error_msg = (
            f"{error_msg}; generated_answer is missing"
            if error_msg
            else "generated_answer is missing"
        )

    # -------- If invalid → append & continue --------
    if error_msg:
        append_row({
            "id": row.get("id"),
            "question": row.get("question"),
            "faithfulness": np.nan,
            "answer_relevancy": np.nan,
            "answer_correctness": np.nan,
            "context_precision": np.nan,
            "context_recall": np.nan,
            "error": error_msg,
        })
        save_checkpoint(idx)
        print(f"⚠️ Row {idx}: {error_msg}")
        continue

    # -------- RAGAS evaluation --------
    try:
        result = {
            "id": row.get("id"),
            "question": row["question"],
            "faithfulness": faithfulness.score(
                user_input=row["question"],
                response=answer,
                retrieved_contexts=contexts,
            ).value,
            "answer_relevancy": relevancy.score(
                user_input=row["question"],
                response=answer,
            ).value,
            "answer_correctness": correctness.score(
                user_input=row["question"],
                response=answer,
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
            "error": None,
        }

        append_row(result)
        save_checkpoint(idx)

    except Exception as e:
        append_row({
            "id": row.get("id"),
            "question": row.get("question"),
            "faithfulness": np.nan,
            "answer_relevancy": np.nan,
            "answer_correctness": np.nan,
            "context_precision": np.nan,
            "context_recall": np.nan,
            "error": str(e),
        })
        save_checkpoint(idx)
        print(f"⚠️ Row {idx}: {e}")
        continue

# -------------------------------------------------
# Summary 
# -------------------------------------------------
if OUTPUT_CSV.exists():
    out_df = pd.read_csv(OUTPUT_CSV)

    total = len(out_df)
    failed = out_df["error"].notna().sum()
    success = total - failed

    print("\n✅ Open-ended RAGAS evaluation complete\n")

    print("📊 Evaluation Summary")
    print("---------------------")
    print(f"Total samples        : {total}")
    print(f"Successfully scored  : {success}")
    print(f"Failed samples       : {failed}")
    print(f"Success rate         : {(success / total) * 100:.2f}%")

    # Retrieval failures
    retrieval_failures = out_df["error"].str.contains(
        "retrieved_contexts", na=False
    ).sum()

    print(f"Retrieval failures   : {retrieval_failures}")
    print(f"Retrieval coverage   : {((total - retrieval_failures) / total) * 100:.2f}%")

    # Valid samples only
    valid_df = out_df[out_df["error"].isna()]

    if not valid_df.empty:
        print("\n📈 Average Scores (valid samples only):")
        print(
            valid_df[
                [
                    "faithfulness",
                    "answer_relevancy",
                    "answer_correctness",
                    "context_precision",
                    "context_recall",
                ]
            ].mean()
        )
    else:
        print("\n⚠️ No valid samples available for averaging")

    # Failure breakdown
    print("\n🧯 Failure breakdown:")
    print(out_df["error"].value_counts(dropna=True))

else:
    print("⚠️ No evaluation output found (CSV not created)")
