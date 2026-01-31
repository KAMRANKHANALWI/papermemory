import os
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
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
scorer = Faithfulness(llm=eval_llm)

# -------------------------------
# Evaluate
# -------------------------------
result = scorer.score(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL–NFL World Championship Game, later known as the Super Bowl was an American football game , "
        "was played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ],
)

print(f"Faithfulness Score: {result.value}")



# -------------- ASYNC VERSION ---------
import os
import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
from dotenv import load_dotenv

load_dotenv()

async def run_faithfulness():
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
    scorer = Faithfulness(llm=eval_llm)

    # -------------------------------
    # Evaluate
    # -------------------------------
    result = await scorer.ascore(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game "
            "played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ],
    )

    print(f"Faithfulness Score: {result.value}")

# Run
asyncio.run(run_faithfulness())
