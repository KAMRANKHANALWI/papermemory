import asyncio
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from src.config import EvalConfig
from src.utils.response_generator import generate_chat_response_eval

# -------------------------------------------------
# Paths & config
# -------------------------------------------------
config = EvalConfig()

INPUT_CSV = Path(config.OPEN_ENDED_CSV)
OUTPUT_CSV = Path(config.OUTPUT_DIR) / "rag_outputs_open_ended.csv"
CHECKPOINT = Path("src/eval/checkpoints/rag_open_ended_checkpoint.json")

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
# Contexts / sources helpers
# -------------------------------------------------
def extract_contexts(contexts: List[str]) -> List[str]:
    if not contexts:
        return []
    return [c.strip() for c in contexts if isinstance(c, str)]


def extract_sources_and_similarity(
    sources: List[Dict[str, Any]],
) -> (List[str], float | None):

    if not sources:
        return [], None

    filenames = set()
    similarities = []

    for src in sources:
        if src.get("filename"):
            filenames.add(src["filename"])
        if isinstance(src.get("similarity"), (int, float)):
            similarities.append(src["similarity"])

    top_similarity = max(similarities) if similarities else None
    return sorted(filenames), top_similarity

# -------------------------------------------------
# Run query via backend RAG Function
# -------------------------------------------------
async def run_single_query(
    question: str,
    collection_name: str,
) -> Dict[str, Any]:

    result = await generate_chat_response_eval(
        message=question,
        collection_name=collection_name,
        chat_mode="single",
    )

    answer = result.get("answer", "")
    contexts_raw = result.get("contexts", [])
    sources_raw = result.get("sources", [])

    contexts = extract_contexts(contexts_raw)
    sources, similarity_score = extract_sources_and_similarity(sources_raw)

    return {
        "answer": answer,
        "contexts": contexts,
        "sources": sources,
        "similarity_score": similarity_score,
    }

async def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"✅ Loaded {len(df)} open-ended questions")

    start_idx = load_checkpoint()
    print(f"▶️ Resuming from index {start_idx + 1}")

    for idx, row in df.iterrows():

        if idx <= start_idx:
            continue

        question = row["question"]
        print(f"Running question {idx + 1}/{len(df)}")

        try:
            rag_result = await run_single_query(
                question=question,
                collection_name=config.COLLECTION_NAME,
            )

            output_row = {
                "id": row.get("id", idx + 1),
                "question": question,
                "reference": row.get("reference"),
                "answer": rag_result["answer"],
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

    print("✅ Open-ended RAG generation completed safely")
    print(f"📁 Output → {OUTPUT_CSV}")

if __name__ == "__main__":
    asyncio.run(main())
