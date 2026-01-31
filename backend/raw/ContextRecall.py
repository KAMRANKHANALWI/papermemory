import os
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# Groq Eval LLM
# -------------------------------
groq_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

eval_llm = llm_factory(
    model=groq_model,
    provider="openai",
    client=groq_client,
)

# -------------------------------
# Metric
# -------------------------------
scorer = ContextRecall(llm=eval_llm)

# -------------------------------
# Evaluate (SYNC)
# -------------------------------
result = scorer.score(
    user_input="Where is the Eiffel Tower located?",
    retrieved_contexts=[
        "Paris is the capital of France."
    ],
    reference="The Eiffel Tower is located in Paris.",
)

print(f"Context Recall Score: {result.value}")
