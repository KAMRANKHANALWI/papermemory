# src/eval/custom_eval_open_ended.py

import pandas as pd
import ast
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

config = EvalConfig()

# --------------------------------
# Load RAG outputs
# --------------------------------
df = pd.read_csv(
    f"{config.OUTPUT_DIR}/rag_outputs_open_ended.csv"
)

print(f"✅ Loaded {len(df)} evaluated RAG samples")

# Normalize ground truth column
if "reference" not in df.columns and "ground_truth" in df.columns:
    df["reference"] = df["ground_truth"]

# --------------------------------
# Groq Eval LLM
# --------------------------------
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

# --------------------------------
# Metrics (explicit instances)
# --------------------------------
faithfulness = Faithfulness(llm=eval_llm)
relevancy = AnswerRelevancy(llm=eval_llm, embeddings=embeddings)
correctness = AnswerCorrectness(llm=eval_llm, embeddings=embeddings)
ctx_precision = ContextPrecision(llm=eval_llm)
ctx_recall = ContextRecall(llm=eval_llm)

results = []

# --------------------------------
# Row-wise Evaluation (safe)
# --------------------------------
for i, row in df.iterrows():
    print(f"Evaluating {i + 1}/{len(df)}")

    try:
        contexts = ast.literal_eval(row["contexts"])
        if not isinstance(contexts, list):
            raise ValueError("Contexts is not a list")

        res = {
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

            # Metadata passthrough (for slicing later)
            "difficulty": row.get("difficulty"),
            "category": row.get("category"),
        }

        results.append(res)

    except Exception as e:
        print(f"⚠️ Skipping row {i}: {e}")

# --------------------------------
# Save Results
# --------------------------------
out_df = pd.DataFrame(results)

output_path = f"{config.OUTPUT_DIR}/open_ended_eval_results.csv"
out_df.to_csv(output_path, index=False)

print("✅ Custom evaluation complete")
print("\n📊 Average Scores:")
print(out_df.mean(numeric_only=True))
