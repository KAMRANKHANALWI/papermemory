"""
RAG Evaluation Pipeline using Ragas
Evaluates open-ended and MCQ questions separately using your existing RAG system
"""

import os
import sys
import pandas as pd
from datasets import Dataset
from openai import AsyncOpenAI
import google.generativeai as genai
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    AnswerCorrectness,
    Faithfulness,
    ContextPrecision,
    ContextRecall,
)

# from ragas.metrics import (
#     answer_relevancy,
#     answer_correctness,
#     faithfulness,
#     context_precision,
#     context_recall
# )

# Import your RAG components
import chromadb
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================


class EvalConfig:
    """Configuration for evaluation pipeline"""

    # File paths
    OPEN_ENDED_CSV = "./sample_open_ended.csv"
    MCQ_CSV = "./sample_mcq.csv"
    OUTPUT_DIR = "evaluation_results"

    # Your RAG system config
    DB_PATH = "data_raw/chroma_db"
    COLLECTION_NAME = "gut_microbiome"  # Change this to your collection name
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    TOP_K = 5  # Number of chunks to retrieve

    # LLM for your RAG (answer generation)
    USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    LLM_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", "gemini")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    TEMPERATURE = 0.1

    # Ragas evaluation LLM (for metrics calculation)
    # Options: "ollama", "gemini", "openai"
    RAGAS_LLM_PROVIDER = os.getenv("RAGAS_LLM_PROVIDER", "gemini")

    # Model names for each provider
    RAGAS_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    RAGAS_GEMINI_MODEL = os.getenv("RAGAS_GEMINI_MODEL", "gemini-3-flash-preview")
    RAGAS_OPENAI_MODEL = os.getenv("RAGAS_OPENAI_MODEL", "gpt-4o-mini")

    # Ragas embedding model
    RAGAS_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================================
# YOUR RAG PIPELINE (Simplified from rag_query.py)
# ============================================================================


class SimpleRAGPipeline:
    """Simplified RAG pipeline for evaluation"""

    def __init__(self, config: EvalConfig):
        self.config = config

        # Initialize LLM for answer generation
        self.llm = self._initialize_llm()

        # Initialize embeddings
        self.embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=config.DB_PATH)

        # Get vectorstore
        self.vectorstore = self._get_vectorstore(config.COLLECTION_NAME)

        print(f"✅ RAG Pipeline initialized")
        print(f"   LLM: {self._get_llm_name()}")
        print(f"   Embeddings: {config.EMBEDDING_MODEL}")
        print(f"   Database: {config.DB_PATH}")
        print(f"   Collection: {config.COLLECTION_NAME}")

    def _initialize_llm(self):
        """Initialize LLM based on configuration"""
        if self.config.USE_LOCAL_LLM:
            print(f"🔧 Using Ollama: {self.config.OLLAMA_MODEL}")
            return ChatOllama(
                model=self.config.OLLAMA_MODEL,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=self.config.TEMPERATURE,
            )
        elif self.config.LLM_PROVIDER == "gemini" and os.getenv("GOOGLE_API_KEY"):
            print(f"🔧 Using Gemini: {self.config.GEMINI_MODEL}")
            return ChatGoogleGenerativeAI(
                model=self.config.GEMINI_MODEL,
                temperature=self.config.TEMPERATURE,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
        elif os.getenv("GROQ_API_KEY"):
            print(f"🔧 Using Groq: {self.config.GROQ_MODEL}")
            return ChatGroq(
                model=self.config.GROQ_MODEL,
                temperature=self.config.TEMPERATURE,
                api_key=os.getenv("GROQ_API_KEY"),
            )
        else:
            raise ValueError("No valid LLM configuration found. Set API keys in .env")

    def _get_llm_name(self) -> str:
        """Get current LLM name"""
        if self.config.USE_LOCAL_LLM:
            return f"Ollama ({self.config.OLLAMA_MODEL})"
        elif self.config.LLM_PROVIDER == "gemini":
            return f"Gemini ({self.config.GEMINI_MODEL})"
        else:
            return f"Groq ({self.config.GROQ_MODEL})"

    def _get_vectorstore(self, collection_name: str):
        """Get vectorstore for a collection"""
        try:
            return Chroma(
                client=self.chroma_client,
                collection_name=collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.config.DB_PATH,
            )
        except Exception as e:
            print(f"❌ Error loading collection '{collection_name}': {e}")
            print(
                f"Available collections: {[c.name for c in self.chroma_client.list_collections()]}"
            )
            raise

    def retrieve(self, query: str, top_k: int = None) -> list[str]:
        """
        Retrieve relevant contexts for a query

        Args:
            query: User question
            top_k: Number of chunks to retrieve

        Returns:
            List of retrieved context strings
        """
        top_k = top_k or self.config.TOP_K

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)

            # Extract just the text content for Ragas
            contexts = []
            for doc, score in results:
                contexts.append(doc.page_content)

            return contexts

        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []

    def generate(self, query: str, contexts: list[str]) -> str:
        """
        Generate answer using retrieved contexts

        Args:
            query: User question
            contexts: Retrieved context chunks

        Returns:
            Generated answer string
        """
        # Build context string
        context_str = "\n\n".join(
            [f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)]
        )

        # Create prompt
        system_prompt = f"""You are a knowledgeable assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly.

Context from documents:
{context_str}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {e}"


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================


def load_dataset(csv_path: str, sample_size: int = None) -> pd.DataFrame:
    """Load questions dataset from CSV"""
    df = pd.read_csv(csv_path)

    if sample_size:
        df = df.head(sample_size)

    print(f"✅ Loaded {len(df)} questions from {csv_path}")
    return df


def generate_rag_responses(
    df: pd.DataFrame, rag_pipeline: SimpleRAGPipeline
) -> pd.DataFrame:
    """Generate RAG responses for all questions"""
    responses = []
    contexts_list = []

    print(f"\n{'='*70}")
    print(f"Generating RAG responses for {len(df)} questions...")
    print(f"{'='*70}")

    for idx, row in df.iterrows():
        query = row["question"]

        print(f"\n[{idx+1}/{len(df)}] Processing: {query[:60]}...")

        # Retrieve contexts
        contexts = rag_pipeline.retrieve(query, top_k=5)
        print(f"   Retrieved {len(contexts)} chunks")

        # Generate answer
        answer = rag_pipeline.generate(query, contexts)
        print(f"   Generated answer: {answer[:80]}...")

        responses.append(answer)
        contexts_list.append(contexts)

    df["response"] = responses
    df["retrieved_contexts"] = contexts_list

    print(f"\n✅ Generated all {len(df)} responses")
    return df


def run_ragas_evaluation(
    df: pd.DataFrame, output_name: str, ragas_llm, ragas_embeddings
) -> pd.DataFrame:
    """Run Ragas evaluation on dataset"""

    print(f"\n{'='*70}")
    print(f"Running Ragas evaluation on {len(df)} samples...")
    print(f"{'='*70}")

    # Prepare dataset for Ragas
    ragas_data = df[["question", "response", "reference", "retrieved_contexts"]].rename(
        columns={"question": "user_input"}
    )

    ragas_dataset = Dataset.from_pandas(ragas_data)

    # Define metrics - Initialize WITH llm and embeddings
    metrics = [
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
        Faithfulness(llm=ragas_llm),  # Only needs llm
        ContextPrecision(llm=ragas_llm),  # Only needs llm
        ContextRecall(llm=ragas_llm),  # Only needs llm
    ]

    print(f"\nEvaluating with metrics:")
    print(f"  - Answer Relevancy")
    print(f"  - Answer Correctness")
    print(f"  - Faithfulness")
    print(f"  - Context Precision")
    print(f"  - Context Recall")

    # Run evaluation - Do NOT pass llm/embeddings again
    try:
        results = evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
        )

        # Convert to DataFrame
        results_df = results.to_pandas()

        # Add back original columns
        results_df["id"] = df["id"].values
        results_df["difficulty"] = df["difficulty"].values
        results_df["category"] = df["category"].values

        # Reorder columns
        cols = [
            "id",
            "user_input",
            "reference",
            "response",
            "difficulty",
            "category",
            "answer_relevancy",
            "answer_correctness",
            "faithfulness",
            "context_precision",
            "context_recall",
        ]
        results_df = results_df[cols]

        # Save to CSV
        os.makedirs(EvalConfig.OUTPUT_DIR, exist_ok=True)
        output_path = f"{EvalConfig.OUTPUT_DIR}/{output_name}_evaluation.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved detailed results to: {output_path}")

        return results_df

    except Exception as e:
        print(f"\n❌ Evaluation error: {e}")
        import traceback

        traceback.print_exc()
        raise


def print_summary(results_df: pd.DataFrame, question_type: str):
    """Print evaluation summary"""

    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY - {question_type.upper()}")
    print(f"{'='*70}")
    print(f"Total Questions: {len(results_df)}")

    print(f"\nOverall Scores (Mean):")
    print(f"  Answer Relevancy:   {results_df['answer_relevancy'].mean():.4f}")
    print(f"  Answer Correctness: {results_df['answer_correctness'].mean():.4f}")
    print(f"  Faithfulness:       {results_df['faithfulness'].mean():.4f}")
    print(f"  Context Precision:  {results_df['context_precision'].mean():.4f}")
    print(f"  Context Recall:     {results_df['context_recall'].mean():.4f}")

    print(f"\nScores by Difficulty:")
    for difficulty in sorted(results_df["difficulty"].unique()):
        subset = results_df[results_df["difficulty"] == difficulty]
        print(f"\n  {difficulty.upper()} ({len(subset)} questions):")
        print(f"    Relevancy:   {subset['answer_relevancy'].mean():.4f}")
        print(f"    Correctness: {subset['answer_correctness'].mean():.4f}")
        print(f"    Faithfulness: {subset['faithfulness'].mean():.4f}")

    print(f"\nScores by Category:")
    for category in sorted(results_df["category"].unique()):
        subset = results_df[results_df["category"] == category]
        print(f"\n  {category.upper()} ({len(subset)} questions):")
        print(f"    Correctness: {subset['answer_correctness'].mean():.4f}")
        print(f"    Faithfulness: {subset['faithfulness'].mean():.4f}")

    print(f"\n{'='*70}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main evaluation pipeline"""

    print("=" * 70)
    print("RAG EVALUATION PIPELINE WITH RAGAS")
    print("=" * 70)

    config = EvalConfig()

    # Initialize Ragas LLM and embeddings (for evaluation)
    print(f"\n1️⃣  Initializing Ragas evaluation components...")
    print(f"   Provider: {config.RAGAS_LLM_PROVIDER}")

    # Initialize Ragas LLM based on provider
    if config.RAGAS_LLM_PROVIDER == "ollama":
        print(f"   Using Ollama (local): {config.RAGAS_OLLAMA_MODEL}")
        print(f"   ⚠️  Note: Local LLM is slower but free")

        ragas_client = AsyncOpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        )
        ragas_llm = llm_factory(
            model=config.RAGAS_OLLAMA_MODEL, provider="openai", client=ragas_client
        )

    elif config.RAGAS_LLM_PROVIDER == "gemini":
        print(f"   Using Gemini: {config.RAGAS_GEMINI_MODEL}")
        print(f"   ⚡ Note: Gemini is fast and cost-effective")

        if not os.getenv("GOOGLE_API_KEY"):
            print(f"\n❌ Error: GOOGLE_API_KEY not found in .env")
            print(f"   Add: GOOGLE_API_KEY=your_key_here")
            return

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        gemini_client = genai.GenerativeModel(config.RAGAS_GEMINI_MODEL)

        ragas_llm = llm_factory(
            model=config.RAGAS_GEMINI_MODEL, provider="google", client=gemini_client
        )

    elif config.RAGAS_LLM_PROVIDER == "openai":
        print(f"   Using OpenAI: {config.RAGAS_OPENAI_MODEL}")
        print(f"   ⚡ Note: OpenAI is fast but more expensive")

        if not os.getenv("OPENAI_API_KEY"):
            print(f"\n❌ Error: OPENAI_API_KEY not found in .env")
            print(f"   Add: OPENAI_API_KEY=your_key_here")
            return

        from openai import OpenAI

        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        ragas_llm = llm_factory(
            model=config.RAGAS_OPENAI_MODEL, provider="openai", client=openai_client
        )
    else:
        print(f"\n❌ Error: Invalid RAGAS_LLM_PROVIDER: {config.RAGAS_LLM_PROVIDER}")
        print(f"   Valid options: 'ollama', 'gemini', 'openai'")
        return

    # Initialize Ragas embeddings
    ragas_embeddings = embedding_factory(
        "huggingface", model=config.RAGAS_EMBEDDING_MODEL
    )

    print(f"✅ Ragas components initialized")
    print(f"   Eval Embeddings: {config.RAGAS_EMBEDDING_MODEL}")

    # Initialize your RAG pipeline (for answer generation)
    print(f"\n2️⃣  Initializing your RAG pipeline...")
    try:
        rag_pipeline = SimpleRAGPipeline(config)
    except Exception as e:
        print(f"\n❌ Failed to initialize RAG pipeline: {e}")
        print(f"\n💡 Make sure:")
        print(f"   - ChromaDB is at: {config.DB_PATH}")
        print(f"   - Collection '{config.COLLECTION_NAME}' exists")
        print(f"   - LLM API keys are set in .env")
        return

    # ========================================================================
    # EVALUATE OPEN-ENDED QUESTIONS
    # ========================================================================
    print(f"\n" + "=" * 70)
    print(f"3️⃣  EVALUATING OPEN-ENDED QUESTIONS")
    print("=" * 70)

    # Load data (5 samples for testing)
    open_ended_df = load_dataset(config.OPEN_ENDED_CSV, sample_size=5)

    # Generate responses
    open_ended_df = generate_rag_responses(open_ended_df, rag_pipeline)

    # Run evaluation
    open_ended_results = run_ragas_evaluation(
        open_ended_df, "open_ended", ragas_llm, ragas_embeddings
    )

    # Print summary
    print_summary(open_ended_results, "Open-Ended Questions")

    # ========================================================================
    # EVALUATE MCQ QUESTIONS
    # ========================================================================
    print(f"\n" + "=" * 70)
    print(f"4️⃣  EVALUATING MCQ QUESTIONS")
    print("=" * 70)

    # Load data (5 samples for testing)
    mcq_df = load_dataset(config.MCQ_CSV, sample_size=5)

    # Generate responses
    mcq_df = generate_rag_responses(mcq_df, rag_pipeline)

    # Run evaluation
    mcq_results = run_ragas_evaluation(mcq_df, "mcq", ragas_llm, ragas_embeddings)

    # Print summary
    print_summary(mcq_results, "MCQ Questions")

    # ========================================================================
    # COMBINED SUMMARY
    # ========================================================================
    print(f"\n" + "=" * 70)
    print(f"5️⃣  COMBINED SUMMARY")
    print("=" * 70)

    print(f"\n📊 Overall Performance:")
    print(f"\nOpen-Ended Questions:")
    print(f"  Avg Relevancy:   {open_ended_results['answer_relevancy'].mean():.4f}")
    print(f"  Avg Correctness: {open_ended_results['answer_correctness'].mean():.4f}")
    print(f"  Avg Faithfulness: {open_ended_results['faithfulness'].mean():.4f}")

    print(f"\nMCQ Questions:")
    print(f"  Avg Relevancy:   {mcq_results['answer_relevancy'].mean():.4f}")
    print(f"  Avg Correctness: {mcq_results['answer_correctness'].mean():.4f}")
    print(f"  Avg Faithfulness: {mcq_results['faithfulness'].mean():.4f}")

    print(f"\n📁 Results saved to '{config.OUTPUT_DIR}/':")
    print(f"   - {config.OUTPUT_DIR}/open_ended_evaluation.csv")
    print(f"   - {config.OUTPUT_DIR}/mcq_evaluation.csv")

    print(f"\n{'='*70}")
    print(f"✅ EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
