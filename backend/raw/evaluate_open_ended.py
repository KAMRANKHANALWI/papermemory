# evaluate_open_ended.py
from datasets import load_from_disk
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
)
from config import EvalConfig

config = EvalConfig()

# -----------------------------
# Load HF Dataset
# -----------------------------
dataset = load_from_disk("hf_datasets/open_ended_eval")
print("✅ Loaded HF Dataset")

# -----------------------------
# Groq Eval LLM
# -----------------------------
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

# -----------------------------
# Metrics (FULLY INITIALIZED)
# -----------------------------
metrics = [
    Faithfulness(llm=eval_llm),
    AnswerRelevancy(llm=eval_llm, embeddings=embeddings),
    AnswerCorrectness(llm=eval_llm, embeddings=embeddings),
    ContextPrecision(llm=eval_llm),
    ContextRecall(llm=eval_llm),
]

# -----------------------------
# Evaluate (NO llm here)
# -----------------------------
result = evaluate(
    dataset=dataset,
    metrics=metrics,
)

print(result)
result.to_pandas().to_csv("open_ended_eval_results.csv", index=False)
print("✅ Evaluation complete")

