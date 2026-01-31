# src/eval/custom_eval_mcq_ragas.py

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
# Helpers
# ---------------------------------------

def extract_explanation(text: str) -> str:
    """
    Extract explanation from MCQ answer:
    - Removes leading A/B/C/D
    - Keeps only explanatory text
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()

    # Remove patterns like:
    # "A. ....", "B)", "Answer: C - ...."
    text = re.sub(
        r"^(ANSWER\s*[:\-]?\s*)?([ABCD])[\.\)\:\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


# ---------------------------------------
# Load MCQ RAG outputs
# ---------------------------------------
input_path = Path(config.OUTPUT_DIR) / "rag_outputs_mcq.csv"
output_path = Path(config.OUTPUT_DIR) / "mcq_ragas_eval_results.csv"

df = pd.read_csv(input_path)
print(f"✅ Loaded {len(df)} MCQ samples")

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

results = []

# ---------------------------------------
# Evaluate row by row
# ---------------------------------------
for i, row in df.iterrows():
    print(f"Evaluating MCQ explanation {i+1}/{len(df)}")

    # contexts are stored as stringified list
    contexts = ast.literal_eval(row["contexts"])

    explanation = extract_explanation(row["model_answer"])

    res = {
        "id": row.get("id"),
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

    results.append(res)

# ---------------------------------------
# Save results
# ---------------------------------------
out_df = pd.DataFrame(results)
out_df.to_csv(output_path, index=False)

print("✅ MCQ RAGAS evaluation complete")
print("\nAverage scores:")
print(out_df.mean(numeric_only=True))




