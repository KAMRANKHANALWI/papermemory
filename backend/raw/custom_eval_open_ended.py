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
from config import EvalConfig

config = EvalConfig()

# --------------------------------
# Load data
# --------------------------------
df = pd.read_csv("rag_outputs_open_ended.csv")

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
# Metrics
# --------------------------------
faithfulness = Faithfulness(llm=eval_llm)
relevancy = AnswerRelevancy(llm=eval_llm, embeddings=embeddings)
correctness = AnswerCorrectness(llm=eval_llm, embeddings=embeddings)
ctx_precision = ContextPrecision(llm=eval_llm)
ctx_recall = ContextRecall(llm=eval_llm)

results = []

# --------------------------------
# Evaluate row by row
# --------------------------------
for i, row in df.iterrows():
    print(f"Evaluating {i+1}/{len(df)}")

    contexts = ast.literal_eval(row["contexts"])

    res = {
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
    }

    results.append(res)

# --------------------------------
# Save results
# --------------------------------
out_df = pd.DataFrame(results)
out_df.to_csv("open_ended_eval_results.csv", index=False)

print("✅ Custom evaluation complete")
print(out_df.mean(numeric_only=True))
