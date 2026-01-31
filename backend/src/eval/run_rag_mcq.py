# src/eval/run_rag_mcq_real.py

import asyncio
import pandas as pd
import ast
from tqdm import tqdm
from typing import Dict, Any

from src.config import EvalConfig
from src.utils.response_generator import generate_chat_response_eval


# --------------------------------
# Build MCQ prompt
# --------------------------------
def build_mcq_query(row) -> str:
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


# --------------------------------
# Run ONE MCQ through REAL RAG
# --------------------------------
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
    sources = result.get("sources", [])

    similarities = [
        src.get("similarity")
        for src in sources
        if isinstance(src.get("similarity"), (int, float))
    ]
    avg_similarity = round(sum(similarities) / len(similarities), 4) if similarities else None

    return {
        "model_answer": answer_text,
        "contexts": contexts,
        "sources": sources,
        "avg_similarity": avg_similarity,
    }


# --------------------------------
# MAIN
# --------------------------------
async def main():
    config = EvalConfig()

    df = pd.read_csv(config.MCQ_CSV)
    print(f"✅ Loaded {len(df)} MCQ questions")

    outputs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running REAL MCQ RAG"):
        rag_result = await run_single_mcq(
            row=row,
            collection_name=config.COLLECTION_NAME,
        )

        outputs.append({
            **row.to_dict(),
            "model_answer": rag_result["model_answer"],
            "contexts": rag_result["contexts"],
            "sources": rag_result["sources"],
            "avg_similarity": rag_result["avg_similarity"],
        })

    out_df = pd.DataFrame(outputs)

    output_path = f"{config.OUTPUT_DIR}/rag_outputs_mcq.csv"
    out_df.to_csv(output_path, index=False)

    print(f"✅ REAL MCQ RAG outputs saved → {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
