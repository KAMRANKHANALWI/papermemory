# custom_eval_mcq_ragas.py
import pandas as pd
import ast
import re
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
)
from config import EvalConfig

config = EvalConfig()

# ---------------------------------------
# Helpers
# ---------------------------------------

def extract_explanation(text: str):
    """
    Remove option letter (A/B/C/D) and return explanation only
    """
    if not isinstance(text, str):
        return ""

    # Remove leading "A.", "B)", etc.
    text = re.sub(r"^[ABCD][\.\)\:]?\s*", "", text.strip(), flags=re.IGNORECASE)
    return text.strip()


# ---------------------------------------
# Load MCQ RAG outputs
# ---------------------------------------
df = pd.read_csv("rag_outputs_mcq.csv")
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

    contexts = ast.literal_eval(row["contexts"])
    explanation = extract_explanation(row["model_answer"])

    res = {
        "id": row["id"],
        "question": row["question"],
        "faithfulness": faithfulness.score(
            user_input=row["question"],
            response=explanation,
            retrieved_contexts=contexts,
        ).value,
        "context_precision": ctx_precision.score(
            user_input=row["question"],
            reference=row["explainations"],
            retrieved_contexts=contexts,
        ).value,
        "context_recall": ctx_recall.score(
            user_input=row["question"],
            reference=row["explainations"],
            retrieved_contexts=contexts,
        ).value,
        "difficulty": row["difficulty"],
        "category": row["category"],
    }

    results.append(res)

# ---------------------------------------
# Save results
# ---------------------------------------
out_df = pd.DataFrame(results)
out_df.to_csv("mcq_ragas_eval_results.csv", index=False)

print("✅ MCQ RAGAS evaluation complete")
print("\nAverage scores:")
print(out_df.mean(numeric_only=True))
