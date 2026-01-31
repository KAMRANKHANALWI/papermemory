# # rag_evaluation_system.py

# import os
# import chromadb
# from dotenv import load_dotenv
# from datasets import Dataset
# from openai import OpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import ChatOllama
# from ragas import evaluate
# from ragas.run_config import RunConfig
# from ragas.llms import llm_factory
# from ragas.metrics import (
#     AnswerCorrectness,
#     AnswerRelevancy,
#     Faithfulness,
#     ContextPrecision,
#     ContextRecall,
# )

# load_dotenv()

# # Sample documents
# docs = [
#     "Paris is the capital and most populous city of France. The city is famed for the Eiffel Tower.",
#     "Jane Austen was an English novelist best known for 'Pride and Prejudice' and 'Sense and Sensibility'.",
#     "The Great Wall of China is a series of fortifications built to protect the ancient Chinese states.",
#     "Mount Everest, part of the Himalayas, is Earth's highest mountain above sea level.",
#     "Mike loves the color pink more than any other color.",
# ]

# # Initialize ChromaDB client
# chroma_client = chromadb.Client()

# # Initialize embedding model (always local)
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )


# # ============================================
# # ✅ NEW: LLM Configuration Functions
# # ============================================

# def get_generation_llm(use_local=True):
#     """
#     Get LLM for answer generation.

#     Args:
#         use_local: If True, use Ollama. If False, use Gemini.

#     Returns:
#         Configured LLM for generating answers
#     """
#     if use_local:
#         print("🏠 Using LOCAL Ollama for answer generation")
#         ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
#         return ChatOllama(
#             model=ollama_model,
#             temperature=0,
#             base_url="http://localhost:11434"
#         )
#     else:
#         print("☁️  Using GEMINI API for answer generation")
#         return ChatGoogleGenerativeAI(
#             model="gemini-2.0-flash-exp",
#             temperature=0,
#             google_api_key=os.getenv("GOOGLE_API_KEY")
#         )


# def get_evaluation_llm(use_local=True):
#     """
#     Get LLM for RAGAS evaluation.

#     Args:
#         use_local: If True, use Ollama. If False, use OpenAI.

#     Returns:
#         RAGAS-compatible LLM for evaluation
#     """
#     if use_local:
#         print("🏠 Using LOCAL Ollama for evaluation")
#         ollama_client = OpenAI(
#             api_key="ollama",
#             base_url="http://localhost:11434/v1"
#         )
#         ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
#         return llm_factory(
#             model=ollama_model,
#             provider="openai",
#             client=ollama_client
#         )
#     else:
#         print("☁️  Using OPENAI API for evaluation")
#         return None  # RAGAS will use default OpenAI


# # ============================================
# # Vector Store Setup
# # ============================================

# def setup_vector_store(collection_name="rag_collection"):
#     """Setup ChromaDB collection and add documents."""
#     collection = chroma_client.get_or_create_collection(
#         name=collection_name,
#         metadata={"hnsw:space": "cosine"}
#     )

#     collection.add(
#         documents=docs,
#         ids=[f"doc_{i}" for i in range(len(docs))],
#     )

#     return collection


# def retrieve(query, collection, k=5):
#     """Retrieve top-k relevant documents for a query."""
#     results = collection.query(
#         query_texts=[query],
#         n_results=min(k, len(docs))
#     )
#     return results["documents"][0]


# # ============================================
# # ✅ UPDATED: Now accepts LLM parameter properly
# # ============================================

# def generate_answer(question, contexts, llm):
#     """
#     Generate answer using LLM based on retrieved contexts.

#     Args:
#         question: User question
#         contexts: Retrieved context documents
#         llm: LLM to use (NO DEFAULT - must be provided!)

#     Returns:
#         Generated answer string
#     """
#     prompt = (
#         "Answer the user question **only** with facts found in the context.\n\n"
#         "Context:\n"
#         + "\n".join(f"- {c}" for c in contexts)
#         + f"\n\nQuestion: {question}\nAnswer:"
#     )

#     response = llm.invoke(prompt)
#     return response.content.strip()


# def create_evaluation_dataset(questions, ground_truths, collection, generation_llm, k=5):
#     """
#     Create evaluation dataset with questions, contexts, answers, and references.

#     Args:
#         questions: List of questions
#         ground_truths: List of ground truth answers
#         collection: ChromaDB collection
#         generation_llm: LLM to use for generating answers
#         k: Number of documents to retrieve

#     Returns:
#         Dataset for evaluation
#     """
#     rows = []

#     print("\n📝 Generating answers...")
#     for i, (question, ground_truth) in enumerate(zip(questions, ground_truths), 1):
#         print(f"  [{i}/{len(questions)}] Processing: {question}")

#         # Retrieve contexts
#         contexts = retrieve(question, collection, k=k)

#         # Generate answer using provided LLM
#         answer = generate_answer(question, contexts, llm=generation_llm)

#         rows.append({
#             "user_input": question,
#             "retrieved_contexts": contexts,
#             "response": answer,
#             "reference": ground_truth,
#         })

#     return Dataset.from_list(rows)


# def evaluate_rag_system(dataset, evaluation_llm=None, use_local_eval=True):
#     """
#     Evaluate RAG system using RAGAS metrics.

#     Args:
#         dataset: Evaluation dataset
#         evaluation_llm: LLM for evaluation (if None, uses default OpenAI)
#         use_local_eval: Whether to use local LLM for evaluation

#     Returns:
#         Evaluation scores
#     """
#     # Initialize metrics
#     metrics = [
#         AnswerCorrectness(),
#         AnswerRelevancy(),
#         Faithfulness(),
#         ContextPrecision(),
#         ContextRecall(),
#     ]

#     if use_local_eval and evaluation_llm is not None:
#         # Configure for longer timeout with local LLM
#         run_config = RunConfig(
#             timeout=180,
#             max_retries=2,
#             max_wait=300
#         )

#         scores = evaluate(
#             dataset,
#             metrics=metrics,
#             llm=evaluation_llm,
#             embeddings=embedding_model,
#             run_config=run_config
#         )
#     else:
#         # Use default OpenAI evaluation (fast)
#         scores = evaluate(
#             dataset,
#             metrics=metrics,
#         )

#     return scores


# def main():
#     """Main execution flow with configuration options."""
#     print("=" * 80)
#     print("🚀 RAG EVALUATION SYSTEM - FLEXIBLE CONFIGURATION")
#     print("=" * 80)

#     # ============================================
#     # ✅ CONFIGURATION - Change these!
#     # ============================================

#     USE_LOCAL_FOR_GENERATION = True   # ← Set to True for Ollama, False for Gemini
#     USE_LOCAL_FOR_EVALUATION = True  # ← Set to True for Ollama, False for OpenAI

#     print(f"\n⚙️  Configuration:")
#     print(f"   Answer Generation: {'LOCAL (Ollama)' if USE_LOCAL_FOR_GENERATION else 'API (Gemini)'}")
#     print(f"   RAG Evaluation: {'LOCAL (Ollama)' if USE_LOCAL_FOR_EVALUATION else 'API (OpenAI)'}")

#     # ============================================
#     # Setup
#     # ============================================

#     print("\n[1/5] Setting up vector store...")
#     collection = setup_vector_store()

#     print("[2/5] Initializing LLMs...")
#     generation_llm = get_generation_llm(use_local=USE_LOCAL_FOR_GENERATION)
#     evaluation_llm = get_evaluation_llm(use_local=USE_LOCAL_FOR_EVALUATION) if USE_LOCAL_FOR_EVALUATION else None

#     # Define test questions
#     questions = [
#         "What is the capital of France?",
#         "Who wrote Pride and Prejudice?",
#         "Where is Mount Everest located?",
#         "What is Mike's favorite color?",
#     ]

#     ground_truths = [
#         "Paris",
#         "Jane Austen",
#         "the Himalayas",
#         "Pink"
#     ]

#     # ============================================
#     # Generate Answers
#     # ============================================

#     print("\n[3/5] Generating answers using configured LLM...")
#     eval_dataset = create_evaluation_dataset(
#         questions,
#         ground_truths,
#         collection,
#         generation_llm=generation_llm,  # ← Pass the LLM!
#         k=5
#     )

#     # Print generated Q&A pairs
#     print("\n" + "=" * 80)
#     print("📋 GENERATED QUESTION-ANSWER PAIRS")
#     print("=" * 80)
#     for i, row in enumerate(eval_dataset, 1):
#         print(f"\n[{i}] Question: {row['user_input']}")
#         print(f"    Answer:    {row['response']}")
#         print(f"    Reference: {row['reference']}")

#     # ============================================
#     # Evaluate
#     # ============================================

#     print("\n" + "=" * 80)
#     print("[4/5] Evaluating RAG System...")
#     print("=" * 80)

#     try:
#         scores = evaluate_rag_system(
#             eval_dataset,
#             evaluation_llm=evaluation_llm,
#             use_local_eval=USE_LOCAL_FOR_EVALUATION
#         )

#         print("\n" + "=" * 80)
#         print("📊 EVALUATION RESULTS")
#         print("=" * 80)

#         for metric, score in scores.items():
#             if score >= 0.8:
#                 status = "✅"
#             elif score >= 0.5:
#                 status = "⚠️"
#             else:
#                 status = "❌"

#             print(f"{status} {metric:25s}: {score:.4f}")

#         print("=" * 80)
#         print("[5/5] ✅ Complete!")

#         return collection, eval_dataset, scores

#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         print("\n💡 Troubleshooting:")
#         if USE_LOCAL_FOR_GENERATION or USE_LOCAL_FOR_EVALUATION:
#             print("  • Make sure Ollama is running: ollama serve")
#             print(f"  • Check model exists: ollama list")
#             print(f"  • Pull if needed: ollama pull {os.getenv('OLLAMA_MODEL', 'llama3.1:latest')}")
#         else:
#             print("  • Check API keys in .env file")

#         return None, None, None


# if __name__ == "__main__":
#     collection, dataset, scores = main()

# -----------------------

# import os
# from openai import AsyncOpenAI
# from ragas.llms import llm_factory
# from ragas.embeddings.base import embedding_factory
# from ragas.metrics.collections import AnswerRelevancy

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
# scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)

# # Evaluate
# result = scorer.score(
#     user_input="When was the first super bowl?",
#     response="The first superbowl was held on Jan 15, 1967",
# )
# print(f"Answer Relevancy Score: {result.value}")

# ----------------  3  ----------------

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







