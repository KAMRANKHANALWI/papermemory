# src/eval/run_rag_mcq.py

import asyncio
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.config import EvalConfig
from src.utils.response_generator import generate_chat_response_eval


# -------------------------------------------------
# Paths & config
# -------------------------------------------------
config = EvalConfig()

INPUT_CSV = Path(config.MCQ_CSV)
OUTPUT_CSV = Path(config.OUTPUT_DIR) / "rag_outputs_mcq.csv"
CHECKPOINT = Path("src/eval/checkpoints/rag_mcq_checkpoint.json")

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


def append_row(row: Dict[str, Any]):
    df = pd.DataFrame([row])
    df.to_csv(
        OUTPUT_CSV,
        mode="a",
        header=not OUTPUT_CSV.exists(),
        index=False,
    )


# -------------------------------------------------
# Build MCQ prompt
# -------------------------------------------------
def build_mcq_query(row: pd.Series) -> str:
    return f"""
Question:
{row['question']}

Options:
A. {row['option_a']}
B. {row['option_b']}
C. {row['option_c']}
D. {row['option_d']}

Answer format STRICTLY:
Option: A/B/C/D
Explanation: short justification based on context
""".strip()


# -------------------------------------------------
# Extract sources + similarity
# -------------------------------------------------
def extract_sources_and_similarity(sources):
    if not sources:
        return [], None

    filenames = set()
    similarities = []

    for src in sources:
        if src.get("filename"):
            filenames.add(src["filename"])
        if isinstance(src.get("similarity"), (int, float)):
            similarities.append(src["similarity"])

    return sorted(filenames), max(similarities) if similarities else None


# -------------------------------------------------
# Run MCQ through backend RAG Function
# -------------------------------------------------
async def run_single_mcq(
    row: pd.Series,
    collection_name: str,
) -> Dict[str, Any]:

    query = build_mcq_query(row)

    result = await generate_chat_response_eval(
        message=query,
        collection_name=collection_name,
        chat_mode="single",
    )

    answer_text = result.get("answer", "")
    contexts = result.get("contexts", [])
    sources_raw = result.get("sources", [])

    sources, similarity_score = extract_sources_and_similarity(sources_raw)

    return {
        "model_answer": answer_text,
        "contexts": contexts,
        "sources": sources,
        "similarity_score": similarity_score,
    }


async def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"✅ Loaded {len(df)} MCQ questions")

    start_idx = load_checkpoint()
    print(f"▶️ Resuming from index {start_idx + 1}")

    for idx, row in df.iterrows():

        if idx <= start_idx:
            continue

        print(f"Running MCQ {idx + 1}/{len(df)}")

        try:
            rag_result = await run_single_mcq(
                row=row,
                collection_name=config.COLLECTION_NAME,
            )

            output_row = {
                "id": row.get("id", idx + 1),
                "question": row["question"],
                "option_a": row["option_a"],
                "option_b": row["option_b"],
                "option_c": row["option_c"],
                "option_d": row["option_d"],
                "correct_option": row["correct_option"],
                "model_answer": rag_result["model_answer"],
                "contexts": rag_result["contexts"],
                "sources": rag_result["sources"],
                "similarity_score": rag_result["similarity_score"],
                "difficulty": row.get("difficulty"),
                "category": row.get("category"),
            }

            append_row(output_row)
            save_checkpoint(idx)

        except Exception as e:
            print(f"⚠️ Stopped safely at row {idx}: {e}")
            break

    print("MCQ RAG generation completed")
    print(f"📁 Output → {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
