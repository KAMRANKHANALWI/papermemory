# ----------------- groq ---------------------- 
import os
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerCorrectness
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# Groq Eval LLM (OpenAI compatible)
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
# Embeddings (local HF is fine)
# -------------------------------
embeddings = embedding_factory(
    "huggingface",
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Metric
# -------------------------------
scorer = AnswerCorrectness(
    llm=eval_llm,
    embeddings=embeddings,
)

# -------------------------------
# Evaluate
# -------------------------------
result = scorer.score(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    reference="The first superbowl was held on January 15, 1967",
)

print(f"Answer Correctness Score: {result.value}")


# ---------------- LOCAL OLLAMA -------------------
# import os
# from openai import AsyncOpenAI
# from ragas.llms import llm_factory
# from ragas.embeddings.base import embedding_factory
# from ragas.metrics.collections import AnswerCorrectness

# # Setup LLM and embeddings with ASYNC client
# client = AsyncOpenAI(
#     api_key="ollama",
#     base_url="http://localhost:11434/v1",
# )

# ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
# llm = llm_factory(model=ollama_model, provider="openai", client=client)

# embeddings = embedding_factory(
#     "huggingface", model="sentence-transformers/all-MiniLM-L6-v2"
# )

# # Create metric
# scorer = AnswerCorrectness(llm=llm, embeddings=embeddings)

# # Evaluate
# result = scorer.score(
#     user_input="When was the first super bowl?",
#     response="The first superbowl was held on Jan 15, 1967",
#     reference="The first superbowl was held on January 15, 1967",
# )
# print(f"Answer Correctness Score: {result.value}")