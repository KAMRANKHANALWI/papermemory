import asyncio
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

from src.config import EvalConfig
from src.utils.response_generator import generate_chat_response_eval


# -------------------------------------------------
# Lightweight entity extractor (eval-safe heuristic)
# -------------------------------------------------
def extract_entities(answer: str) -> List[str]:
    """
    Extract capitalized scientific / proper terms.
    Lightweight by design (no NER model).
    """
    if not answer:
        return []

    tokens = (
        answer.replace("*", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(".", "")
        .split()
    )

    entities = {tok for tok in tokens if tok.istitle() and len(tok) > 3}

    return sorted(entities)


# -------------------------------------------------
# Helpers: contexts, sources, similarity
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

    filenames = []
    similarities = []

    for src in sources:
        fname = src.get("filename")
        sim = src.get("similarity")

        if fname:
            filenames.append(fname)
        if isinstance(sim, (int, float)):
            similarities.append(sim)

    top_similarity = max(similarities) if similarities else None
    return sorted(set(filenames)), top_similarity


# -------------------------------------------------
# Run RAG for one query
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
    entities = extract_entities(answer)

    return {
        "answer": answer,
        "contexts": contexts,
        "sources": sources,
        "similarity_score": similarity_score,
        "entities": entities,
    }


# -------------------------------------------------
# MAIN
# -------------------------------------------------
async def main():
    config = EvalConfig()

    df = pd.read_csv(config.OPEN_ENDED_CSV)
    print(f"✅ Loaded {len(df)} open-ended questions")

    outputs = []

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Running RAG",
    ):
        question = row["question"]

        rag_result = await run_single_query(
            question=question,
            collection_name=config.COLLECTION_NAME,
        )

        outputs.append(
            {
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
        )

    out_df = pd.DataFrame(outputs)

    output_path = f"{config.OUTPUT_DIR}/rag_outputs_open_ended.csv"
    out_df.to_csv(output_path, index=False)

    print(f"✅ REAL RAG outputs saved → {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
