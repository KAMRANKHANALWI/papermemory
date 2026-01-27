# #!/usr/bin/env python3
# """
# Ragas Evaluation Script for Gut Microbiome RAG System
# Evaluates both open-ended and MCQ questions separately

# Usage:
#     # Evaluate open-ended questions
#     python evaluate_rag_with_ragas.py --type open_ended --collection your_collection --sample 50
    
#     # Evaluate MCQ questions
#     python evaluate_rag_with_ragas.py --type mcq --collection your_collection --sample 50
    
#     # Evaluate full dataset
#     python evaluate_rag_with_ragas.py --type open_ended --collection your_collection
    
#     # Use ChatALL mode (search across all collections)
#     python evaluate_rag_with_ragas.py --type open_ended --chatall --sample 100
# """

# import os
# import sys
# import argparse
# import pandas as pd
# from datasets import Dataset

# # Try to import Ragas metrics with proper version compatibility
# try:
#     # Try Ragas v1.0+ import first
#     from ragas.metrics import AnswerCorrectness, AnswerRelevancy, Faithfulness, ContextPrecision, ContextRecall
#     RAGAS_V1 = True
#     print("✓ Using Ragas v1.0+ metrics")
# except ImportError:
#     try:
#         # Try old collections import
#         from ragas.metrics.collections import (
#             answer_correctness,
#             answer_relevancy,
#             faithfulness,
#             context_precision,
#             context_recall,
#         )
#         RAGAS_V1 = False
#         print("✓ Using Ragas v0.x metrics (collections)")
#     except ImportError:
#         # Fallback to oldest import
#         from ragas.metrics import (
#             answer_correctness,
#             answer_relevancy,
#             faithfulness,
#             context_precision,
#             context_recall,
#         )
#         RAGAS_V1 = False
#         print("✓ Using Ragas v0.x metrics (direct)")

# from ragas import evaluate
# from typing import List, Dict, Optional
# from tqdm import tqdm
# import json
# from datetime import datetime
# import time

# # Import your RAG pipeline
# # Make sure rag_query.py is in the same directory or in Python path
# try:
#     from rag_query import RAGPipeline, RAGConfig
# except ImportError:
#     print("❌ Error: Cannot import rag_query.py")
#     print("   Make sure rag_query.py is in the same directory")
#     sys.exit(1)


# class RAGEvaluator:
#     """Ragas-based evaluator for RAG system"""
    
#     def __init__(
#         self,
#         rag_pipeline: RAGPipeline,
#         collection_name: Optional[str] = None,
#         chatall: bool = False,
#         delay: float = 15.0
#     ):
#         self.rag_pipeline = rag_pipeline
#         self.collection_name = collection_name
#         self.chatall = chatall
#         self.delay = delay  # Delay between questions to avoid rate limits
        
#         # Ragas metrics - version-compatible initialization
#         if RAGAS_V1:
#             # Ragas v1.0+ uses class constructors
#             self.metrics = [
#                 AnswerCorrectness(),
#                 AnswerRelevancy(),
#                 Faithfulness(),
#                 ContextPrecision(),
#                 ContextRecall(),
#             ]
#         else:
#             # Ragas v0.x uses direct objects
#             self.metrics = [
#                 answer_correctness,
#                 answer_relevancy,
#                 faithfulness,
#                 context_precision,
#                 context_recall,
#             ]
    
#     def load_dataset(self, dataset_type: str = "open_ended") -> pd.DataFrame:
#         """
#         Load evaluation dataset
        
#         Args:
#             dataset_type: "open_ended", "mcq", or "full"
#         """
#         dataset_files = {
#             "open_ended": "Processed_Eval_Data/gut_microbiome_open_ended.csv",
#             "mcq": "Processed_Eval_Data/gut_microbiome_mcq.csv",
#             "full": "Processed_Eval_Data/gut_microbiome_full_dataset.csv"
#         }
        
#         if dataset_type not in dataset_files:
#             raise ValueError(f"Invalid dataset_type. Choose from: {list(dataset_files.keys())}")
        
#         filepath = dataset_files[dataset_type]
        
#         if not os.path.exists(filepath):
#             raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
#         df = pd.read_csv(filepath)
#         print(f"✅ Loaded {len(df)} questions from {filepath}")
        
#         return df
    
#     def retrieve_contexts(self, question: str, k: int = 10) -> List[str]:
#         """
#         Retrieve contexts using your RAG pipeline
        
#         This is the wrapper for your retrieve function
#         """
#         try:
#             # Use your RAG pipeline's query function
#             result = self.rag_pipeline.query(
#                 question,
#                 collection_name=self.collection_name,
#                 chatall=self.chatall,
#                 classify=False  # Skip classification for evaluation
#             )
            
#             # Extract contexts from sources
#             if "sources" in result and result["sources"]:
#                 contexts = [source["content"] for source in result["sources"]]
#                 return contexts[:k]  # Limit to top k
#             else:
#                 return []
        
#         except Exception as e:
#             print(f"⚠️  Error retrieving contexts: {e}")
#             return []
    
#     def generate_answer(self, question: str, contexts: List[str]) -> str:
#         """
#         Generate answer using your RAG pipeline
        
#         This is the wrapper for your generate function
#         """
#         try:
#             # Build context string
#             context_text = "\n\n".join(contexts)
            
#             # Use your RAG pipeline's generate function
#             answer = self.rag_pipeline.generate_answer(question, context_text)
            
#             return answer
        
#         except Exception as e:
#             print(f"⚠️  Error generating answer: {e}")
#             return f"Error: {str(e)}"
    
#     def evaluate_single_question(
#         self,
#         question: str,
#         reference: str,
#         k: int = 10
#     ) -> Dict:
#         """
#         Evaluate a single question
        
#         Returns dict with question, contexts, answer, reference
#         """
#         # Step 1: Retrieve contexts
#         contexts = self.retrieve_contexts(question, k=k)
        
#         # Step 2: Generate answer
#         answer = self.generate_answer(question, contexts)
        
#         return {
#             "question": question,
#             "contexts": contexts,
#             "answer": answer,
#             "reference": reference
#         }
    
#     def evaluate_dataset(
#         self,
#         df: pd.DataFrame,
#         k: int = 10,
#         sample_size: Optional[int] = None,
#         output_dir: str = "./evaluation_results"
#     ) -> Dict:
#         """
#         Evaluate entire dataset with Ragas
        
#         Args:
#             df: DataFrame with questions and references
#             k: Number of contexts to retrieve
#             sample_size: If set, randomly sample this many questions
#             output_dir: Directory to save results
        
#         Returns:
#             Dict with scores and detailed results
#         """
#         # Sample if requested
#         if sample_size and sample_size < len(df):
#             df = df.sample(n=sample_size, random_state=42)
#             print(f"📊 Evaluating {sample_size} sampled questions")
#         else:
#             print(f"📊 Evaluating all {len(df)} questions")
        
#         # Create output directory
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Evaluate each question
#         rows = []
#         print(f"\n{'='*60}")
#         print(f"🔄 Processing questions...")
#         print(f"{'='*60}\n")
        
#         for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
#             result = self.evaluate_single_question(
#                 question=row['question'],
#                 reference=row['reference'],
#                 k=k
#             )
            
#             # Add metadata
#             result['id'] = row.get('id', f'q_{idx}')
#             result['difficulty'] = row.get('difficulty', 'unknown')
#             result['category'] = row.get('category', 'unknown')
#             result['question_type'] = row.get('question_type', 'unknown')
#             result['source_document'] = row.get('source_document', 'unknown')
            
#             rows.append(result)
            
#             # Add delay to avoid rate limits
#             if idx < len(df) - 1 and self.delay > 0:  # Don't delay after last question
#                 time.sleep(self.delay)
        
#         # Create HuggingFace Dataset for Ragas
#         print(f"\n{'='*60}")
#         print(f"🔬 Running Ragas evaluation...")
#         print(f"{'='*60}\n")
        
#         evaluation_dataset = Dataset.from_list(rows)
        
#         # Run Ragas evaluation
#         try:
#             scores = evaluate(
#                 evaluation_dataset,
#                 metrics=self.metrics,
#             )
            
#             # Save results
#             self._save_results(rows, scores, df, output_dir)
            
#             return {
#                 'scores': scores,
#                 'detailed_results': rows,
#                 'num_questions': len(rows)
#             }
        
#         except Exception as e:
#             print(f"❌ Ragas evaluation error: {e}")
#             print(f"   Saving partial results...")
#             self._save_partial_results(rows, output_dir)
#             raise
    
#     def _save_results(
#         self,
#         rows: List[Dict],
#         scores: Dict,
#         original_df: pd.DataFrame,
#         output_dir: str
#     ):
#         """Save evaluation results"""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # Save detailed results
#         results_df = pd.DataFrame(rows)
#         results_path = os.path.join(output_dir, f"detailed_results_{timestamp}.csv")
#         results_df.to_csv(results_path, index=False)
#         print(f"✅ Saved detailed results: {results_path}")
        
#         # Save Ragas scores
#         scores_dict = dict(scores)
#         scores_path = os.path.join(output_dir, f"ragas_scores_{timestamp}.json")
#         with open(scores_path, 'w') as f:
#             json.dump(scores_dict, f, indent=2)
#         print(f"✅ Saved Ragas scores: {scores_path}")
        
#         # Save summary report
#         report_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
#         self._create_summary_report(scores_dict, results_df, original_df, report_path)
#         print(f"✅ Saved summary report: {report_path}")
        
#         # Save score breakdown by metadata
#         self._save_score_breakdowns(results_df, output_dir, timestamp)
    
#     def _save_partial_results(self, rows: List[Dict], output_dir: str):
#         """Save partial results if Ragas fails"""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         results_df = pd.DataFrame(rows)
#         results_path = os.path.join(output_dir, f"partial_results_{timestamp}.csv")
#         results_df.to_csv(results_path, index=False)
#         print(f"✅ Saved partial results: {results_path}")
    
#     def _create_summary_report(
#         self,
#         scores: Dict,
#         results_df: pd.DataFrame,
#         original_df: pd.DataFrame,
#         report_path: str
#     ):
#         """Create text summary report"""
#         report = []
#         report.append("="*80)
#         report.append("RAG EVALUATION REPORT - RAGAS METRICS")
#         report.append("="*80)
#         report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         report.append(f"Total Questions Evaluated: {len(results_df)}")
#         report.append("")
        
#         # Overall Ragas scores
#         report.append("RAGAS SCORES")
#         report.append("-"*80)
#         for metric, score in scores.items():
#             report.append(f"{metric:.<50} {score:.4f}")
#         report.append("")
        
#         # Dataset statistics
#         report.append("DATASET STATISTICS")
#         report.append("-"*80)
#         report.append(f"Total questions in dataset: {len(original_df)}")
#         report.append(f"Questions evaluated: {len(results_df)}")
#         report.append("")
        
#         # By difficulty
#         report.append("BY DIFFICULTY:")
#         for diff in ['easy', 'medium', 'hard']:
#             count = len(results_df[results_df['difficulty'] == diff])
#             if count > 0:
#                 report.append(f"  {diff.capitalize():.<20} {count}")
#         report.append("")
        
#         # By category
#         report.append("BY CATEGORY:")
#         for cat in results_df['category'].unique():
#             count = len(results_df[results_df['category'] == cat])
#             report.append(f"  {cat:.<20} {count}")
#         report.append("")
        
#         # Context statistics
#         report.append("RETRIEVAL STATISTICS")
#         report.append("-"*80)
#         contexts_retrieved = [len(r) for r in results_df['contexts'] if isinstance(r, list)]
#         if contexts_retrieved:
#             report.append(f"Average contexts retrieved: {sum(contexts_retrieved)/len(contexts_retrieved):.2f}")
#             report.append(f"Min contexts: {min(contexts_retrieved)}")
#             report.append(f"Max contexts: {max(contexts_retrieved)}")
#         report.append("")
        
#         report.append("="*80)
        
#         # Write report
#         with open(report_path, 'w') as f:
#             f.write("\n".join(report))
    
#     def _save_score_breakdowns(self, results_df: pd.DataFrame, output_dir: str, timestamp: str):
#         """Save score breakdowns by difficulty and category"""
#         # Note: Individual metric scores per question are not directly available
#         # from Ragas. We can only get overall scores.
#         # This is a placeholder for future enhancement if needed.
#         pass
    
#     def evaluate_by_difficulty(
#         self,
#         df: pd.DataFrame,
#         k: int = 10,
#         output_dir: str = "./evaluation_results"
#     ) -> Dict[str, Dict]:
#         """
#         Evaluate separately by difficulty level
        
#         Returns dict with results for each difficulty level
#         """
#         results_by_difficulty = {}
        
#         for difficulty in ['easy', 'medium', 'hard']:
#             diff_df = df[df['difficulty'] == difficulty]
            
#             if len(diff_df) == 0:
#                 continue
            
#             print(f"\n{'='*60}")
#             print(f"📊 Evaluating {difficulty.upper()} questions ({len(diff_df)} total)")
#             print(f"{'='*60}")
            
#             diff_output_dir = os.path.join(output_dir, f"difficulty_{difficulty}")
            
#             result = self.evaluate_dataset(
#                 diff_df,
#                 k=k,
#                 output_dir=diff_output_dir
#             )
            
#             results_by_difficulty[difficulty] = result
            
#             # Print scores
#             print(f"\n📈 Scores for {difficulty} questions:")
#             for metric, score in result['scores'].items():
#                 print(f"  {metric}: {score:.4f}")
        
#         return results_by_difficulty


# def main():
#     parser = argparse.ArgumentParser(
#         description="Evaluate RAG system with Ragas on gut microbiome dataset",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Evaluate 50 open-ended questions
#   python evaluate_rag_with_ragas.py --type open_ended -c my_collection --sample 50
  
#   # Evaluate all MCQ questions
#   python evaluate_rag_with_ragas.py --type mcq -c my_collection
  
#   # Evaluate with ChatALL mode (all collections)
#   python evaluate_rag_with_ragas.py --type open_ended --chatall --sample 100
  
#   # Evaluate by difficulty
#   python evaluate_rag_with_ragas.py --type open_ended -c my_collection --by-difficulty
  
#   # Custom top-k for retrieval
#   python evaluate_rag_with_ragas.py --type open_ended -c my_collection --top-k 5
#         """
#     )
    
#     # Required arguments
#     parser.add_argument(
#         "--type",
#         required=True,
#         choices=["open_ended", "mcq", "full"],
#         help="Type of questions to evaluate"
#     )
    
#     # Collection arguments (same as rag_query.py)
#     parser.add_argument("-c", "--collection", help="Collection name to search")
#     parser.add_argument("--chatall", action="store_true", help="Search across all collections")
    
#     # Evaluation arguments
#     parser.add_argument("--sample", type=int, help="Number of questions to sample (for testing)")
#     parser.add_argument("--top-k", type=int, default=10, help="Number of contexts to retrieve (default: 10)")
#     parser.add_argument("--by-difficulty", action="store_true", help="Evaluate separately by difficulty")
#     parser.add_argument("--output-dir", default="./evaluation_results", help="Output directory")
#     parser.add_argument("--delay", type=float, default=15.0, help="Delay between questions in seconds (default: 15, use 0 for Groq)")
    
#     # RAG Configuration (same as rag_query.py)
#     parser.add_argument("--use-local", action="store_true", help="Use local Ollama LLM")
#     parser.add_argument("--llm-provider", choices=["gemini", "groq"], help="LLM provider")
#     parser.add_argument("--ollama-model", help="Ollama model name")
#     parser.add_argument("--gemini-model", help="Gemini model name")
#     parser.add_argument("--groq-model", help="Groq model name")
#     parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
#     parser.add_argument("--embedding-model", help="HuggingFace embedding model")
#     parser.add_argument("--db-path", default="data_raw/chroma_db", help="Path to ChromaDB database")
#     parser.add_argument("--top-k-chatall", type=int, default=1, help="Chunks per collection in ChatALL mode")
#     parser.add_argument("--max-history", type=int, default=5, help="Max conversation history")
    
#     args = parser.parse_args()
    
#     # Validate collection requirement
#     if not args.collection and not args.chatall:
#         print("❌ Error: Must specify --collection or use --chatall")
#         parser.print_help()
#         sys.exit(1)
    
#     print(f"\n{'='*60}")
#     print(f"🚀 RAG EVALUATION WITH RAGAS")
#     print(f"{'='*60}\n")
    
#     # Initialize RAG pipeline
#     print("1️⃣ Initializing RAG pipeline...")
#     try:
#         from rag_query import RAGConfig
#         config = RAGConfig(args)
#         rag_pipeline = RAGPipeline(config)
#     except Exception as e:
#         print(f"❌ Failed to initialize RAG pipeline: {e}")
#         sys.exit(1)
    
#     # Initialize evaluator
#     print("2️⃣ Initializing Ragas evaluator...")
#     evaluator = RAGEvaluator(
#         rag_pipeline=rag_pipeline,
#         collection_name=args.collection,
#         chatall=args.chatall,
#         delay=args.delay
#     )
    
#     # Load dataset
#     print(f"3️⃣ Loading {args.type} dataset...")
#     try:
#         df = evaluator.load_dataset(args.type)
#     except Exception as e:
#         print(f"❌ Failed to load dataset: {e}")
#         sys.exit(1)
    
#     # Run evaluation
#     print(f"4️⃣ Starting evaluation...")
    
#     if args.by_difficulty:
#         # Evaluate by difficulty
#         results = evaluator.evaluate_by_difficulty(
#             df,
#             k=args.top_k,
#             output_dir=args.output_dir
#         )
        
#         # Print summary
#         print(f"\n{'='*60}")
#         print(f"📊 EVALUATION COMPLETE - SUMMARY BY DIFFICULTY")
#         print(f"{'='*60}\n")
        
#         for difficulty, result in results.items():
#             print(f"{difficulty.upper()} ({result['num_questions']} questions):")
#             for metric, score in result['scores'].items():
#                 print(f"  {metric}: {score:.4f}")
#             print()
    
#     else:
#         # Standard evaluation
#         results = evaluator.evaluate_dataset(
#             df,
#             k=args.top_k,
#             sample_size=args.sample,
#             output_dir=args.output_dir
#         )
        
#         # Print summary
#         print(f"\n{'='*60}")
#         print(f"📊 EVALUATION COMPLETE")
#         print(f"{'='*60}\n")
        
#         print(f"Questions evaluated: {results['num_questions']}")
#         print(f"\nRAGAS SCORES:")
#         for metric, score in results['scores'].items():
#             print(f"  {metric}: {score:.4f}")
#         print()
    
#     print(f"✅ All results saved in: {args.output_dir}\n")


# if __name__ == "__main__":
#     main()

# ====================== 2 ============================
#!/usr/bin/env python3
"""
Production RAG Evaluation with Ragas
Fully integrated with .env configuration - supports Ollama, Groq, and Gemini

Usage:
    # Evaluate with settings from .env
    python evaluate_rag.py --type mcq --collection gut_microbiome --sample 10
    
    # Override .env settings
    python evaluate_rag.py --type mcq --collection gut_microbiome --use-local --sample 10
    
    # Full evaluation
    python evaluate_rag.py --type mcq --collection gut_microbiome
"""

import os
import sys
import argparse
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from typing import List, Dict, Optional
from tqdm import tqdm
import json
from datetime import datetime
import time

# Load environment variables FIRST
load_dotenv()

# Import Ragas metrics with version compatibility
try:
    from ragas.metrics import AnswerCorrectness, AnswerRelevancy, Faithfulness, ContextPrecision, ContextRecall
    RAGAS_V1 = True
except ImportError:
    try:
        from ragas.metrics.collections import (
            answer_correctness,
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        )
        RAGAS_V1 = False
    except ImportError:
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        )
        RAGAS_V1 = False

from ragas import evaluate

# Import after dotenv
import chromadb
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


class ConfigManager:
    """Manages configuration from .env and command-line args"""
    
    def __init__(self, args):
        """Initialize config with priority: CLI args > .env > defaults"""
        
        # LLM Configuration
        self.use_local_llm = self._get_bool(args.use_local, "USE_LOCAL_LLM", False)
        
        if self.use_local_llm:
            self.llm_provider = "ollama"
            self.ollama_model = args.ollama_model or os.getenv("OLLAMA_MODEL", "llama3.1:latest")
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        else:
            self.llm_provider = args.llm_provider or os.getenv("DEFAULT_MODEL_PROVIDER", "gemini")
            self.gemini_model = args.gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
            self.groq_model = args.groq_model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        
        # Embedding Configuration
        self.embedding_model = args.embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # Retrieval Configuration
        self.top_k = args.top_k
        self.top_k_chatall = args.top_k_chatall or int(os.getenv("TOP_K_CHATALL", "1"))
        
        # Database Configuration
        self.db_path = args.db_path or os.getenv("CHROMA_DB_PATH", "data_raw/chroma_db")
        
        # Evaluation Configuration
        self.temperature = args.temperature
        self.delay = args.delay
        self.max_history = args.max_history or int(os.getenv("MAX_HISTORY", "5"))
    
    def _get_bool(self, arg_value, env_var, default):
        """Get boolean from args or env"""
        if arg_value is not None:
            return arg_value
        env_value = os.getenv(env_var, str(default)).lower()
        return env_value in ('true', '1', 'yes', 'on')
    
    def print_config(self):
        """Print current configuration"""
        print(f"{'='*60}")
        print(f"CONFIGURATION")
        print(f"{'='*60}")
        if self.use_local_llm:
            print(f"LLM: Ollama ({self.ollama_model})")
            print(f"Ollama URL: {self.ollama_base_url}")
        elif self.llm_provider == "gemini":
            print(f"LLM: Gemini ({self.gemini_model})")
        else:
            print(f"LLM: Groq ({self.groq_model})")
        print(f"Embeddings: {self.embedding_model}")
        print(f"Database: {self.db_path}")
        print(f"Top-K: {self.top_k}")
        print(f"Temperature: {self.temperature}")
        print(f"Delay: {self.delay}s")
        print(f"{'='*60}\n")


class RAGPipeline:
    """RAG pipeline for document querying"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.llm = self._initialize_llm()
        self.embedding_model = HuggingFaceEmbeddings(model_name=config.embedding_model)
        self.chroma_client = chromadb.PersistentClient(path=config.db_path)
        
        print(f"✅ RAG Pipeline initialized")
    
    def _initialize_llm(self):
        """Initialize LLM based on configuration"""
        if self.config.use_local_llm:
            print(f"🔧 Initializing Ollama: {self.config.ollama_model}")
            return ChatOllama(
                model=self.config.ollama_model,
                base_url=self.config.ollama_base_url,
                temperature=self.config.temperature,
            )
        elif self.config.llm_provider == "gemini" and os.getenv("GOOGLE_API_KEY"):
            print(f"🔧 Initializing Gemini: {self.config.gemini_model}")
            return ChatGoogleGenerativeAI(
                model=self.config.gemini_model,
                temperature=self.config.temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
        elif os.getenv("GROQ_API_KEY"):
            print(f"🔧 Initializing Groq: {self.config.groq_model}")
            return ChatGroq(
                model=self.config.groq_model,
                temperature=self.config.temperature,
                api_key=os.getenv("GROQ_API_KEY"),
            )
        else:
            raise ValueError("No valid LLM configuration found. Check your .env file.")
    
    def get_vectorstore(self, collection_name: str) -> Optional[Chroma]:
        """Get vectorstore for a collection"""
        try:
            return Chroma(
                client=self.chroma_client,
                collection_name=collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.config.db_path,
            )
        except Exception as e:
            print(f"❌ Error loading collection '{collection_name}': {e}")
            return None
    
    def retrieve_contexts(self, query: str, collection_name: str, k: int) -> List[str]:
        """Retrieve contexts from collection"""
        vectorstore = self.get_vectorstore(collection_name)
        if not vectorstore:
            return []
        
        try:
            results = vectorstore.similarity_search_with_score(query, k=k)
            contexts = [doc.page_content for doc, score in results]
            return contexts
        except Exception as e:
            print(f"⚠️  Retrieval error: {e}")
            return []
    
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """Generate answer using LLM"""
        if not contexts:
            return "No relevant context found."
        
        context_text = "\n\n".join(contexts)
        
        system_prompt = f"""You are a knowledgeable scientific assistant. Answer questions based only on the provided context.

Context from documents:
{context_text}
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


class RAGEvaluator:
    """Ragas-based evaluator for RAG system"""
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        collection_name: str,
        delay: float = 0.0
    ):
        self.rag_pipeline = rag_pipeline
        self.collection_name = collection_name
        self.delay = delay
        
        # Initialize Ragas metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize Ragas metrics with custom LLM and embeddings"""
        if RAGAS_V1:
            ragas_llm = self.rag_pipeline.llm
            ragas_embeddings = self.rag_pipeline.embedding_model
            
            try:
                self.metrics = [
                    AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
                    AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
                    Faithfulness(llm=ragas_llm),
                    ContextPrecision(llm=ragas_llm),
                    ContextRecall(llm=ragas_llm),
                ]
                print(f"✓ Ragas metrics configured with custom LLM")
            except Exception as e:
                print(f"⚠️  Trying LLM-only configuration: {e}")
                try:
                    self.metrics = [
                        AnswerCorrectness(llm=ragas_llm),
                        AnswerRelevancy(llm=ragas_llm),
                        Faithfulness(llm=ragas_llm),
                        ContextPrecision(llm=ragas_llm),
                        ContextRecall(llm=ragas_llm),
                    ]
                    print(f"✓ Ragas metrics configured (LLM only)")
                except Exception as e2:
                    print(f"⚠️  Using default Ragas configuration")
                    self.metrics = [
                        AnswerCorrectness(),
                        AnswerRelevancy(),
                        Faithfulness(),
                        ContextPrecision(),
                        ContextRecall(),
                    ]
        else:
            self.metrics = [
                answer_correctness,
                answer_relevancy,
                faithfulness,
                context_precision,
                context_recall,
            ]
    
    def load_dataset(self, dataset_type: str = "open_ended") -> pd.DataFrame:
        """Load evaluation dataset"""
        dataset_files = {
            "open_ended": "Processed_Eval_Data/gut_microbiome_open_ended.csv",
            "mcq": "Processed_Eval_Data/gut_microbiome_mcq.csv",
            "full": "Processed_Eval_Data/gut_microbiome_full_dataset.csv"
        }
        
        filepath = dataset_files[dataset_type]
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} questions from {filepath}")
        return df
    
    def evaluate_single_question(self, question: str, reference: str, k: int) -> Dict:
        """Evaluate a single question"""
        contexts = self.rag_pipeline.retrieve_contexts(question, self.collection_name, k)
        answer = self.rag_pipeline.generate_answer(question, contexts)
        
        return {
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "reference": reference
        }
    
    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        k: int,
        sample_size: Optional[int] = None,
        output_dir: str = "./evaluation_results"
    ) -> Dict:
        """Evaluate entire dataset with Ragas"""
        
        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"📊 Evaluating {sample_size} sampled questions")
        else:
            print(f"📊 Evaluating all {len(df)} questions")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate each question
        rows = []
        print(f"\n{'='*60}")
        print(f"🔄 Processing questions...")
        print(f"{'='*60}\n")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            result = self.evaluate_single_question(
                question=row['question'],
                reference=row['reference'],
                k=k
            )
            
            # Add metadata
            result['id'] = row.get('id', f'q_{idx}')
            result['difficulty'] = row.get('difficulty', 'unknown')
            result['category'] = row.get('category', 'unknown')
            result['question_type'] = row.get('question_type', 'unknown')
            result['source_document'] = row.get('source_document', 'unknown')
            
            rows.append(result)
            
            # Delay between questions if needed
            if idx < len(df) - 1 and self.delay > 0:
                time.sleep(self.delay)
        
        # Create HuggingFace Dataset for Ragas
        print(f"\n{'='*60}")
        print(f"🔬 Running Ragas evaluation...")
        print(f"{'='*60}\n")
        
        evaluation_dataset = Dataset.from_list(rows)
        
        try:
            scores = evaluate(
                evaluation_dataset,
                metrics=self.metrics,
                llm=self.rag_pipeline.llm,
                embeddings=self.rag_pipeline.embedding_model,
            )
            
            # Print scores
            print(f"\n{'='*60}")
            print(f"📊 RAGAS SCORES")
            print(f"{'='*60}")
            try:
                if hasattr(scores, '_scores_dict'):
                    for metric, score in scores._scores_dict.items():
                        if isinstance(score, (list, tuple)):
                            avg_score = sum(s for s in score if s == s) / len([s for s in score if s == s])  # Ignore NaN
                            print(f"  {metric}: {avg_score:.4f}")
                        else:
                            print(f"  {metric}: {score:.4f}")
                else:
                    print(scores)
            except Exception as e:
                print(f"  Raw scores: {scores}")
            print(f"{'='*60}\n")
            
            # Save results
            self._save_results(rows, scores, df, output_dir)
            
            return {
                'scores': scores,
                'detailed_results': rows,
                'num_questions': len(rows)
            }
        
        except Exception as e:
            print(f"❌ Ragas evaluation error: {e}")
            print(f"   Saving partial results...")
            self._save_partial_results(rows, output_dir)
            
            return {
                'scores': {},
                'detailed_results': rows,
                'num_questions': len(rows),
                'error': str(e)
            }
    
    def _save_results(self, rows: List[Dict], scores: Dict, original_df: pd.DataFrame, output_dir: str):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_df = pd.DataFrame(rows)
        results_path = os.path.join(output_dir, f"detailed_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"✅ Saved detailed results: {results_path}")
        
        # Save Ragas scores
        try:
            if hasattr(scores, '_scores_dict'):
                scores_dict = {}
                for k, v in scores._scores_dict.items():
                    if isinstance(v, (list, tuple)):
                        scores_dict[k] = [float(x) if x == x else None for x in v]  # Convert NaN to None
                    else:
                        scores_dict[k] = float(v) if v == v else None
            else:
                scores_dict = dict(scores) if scores else {}
            
            scores_path = os.path.join(output_dir, f"ragas_scores_{timestamp}.json")
            with open(scores_path, 'w') as f:
                json.dump(scores_dict, f, indent=2)
            print(f"✅ Saved Ragas scores: {scores_path}")
        except Exception as e:
            print(f"⚠️  Could not save scores: {e}")
    
    def _save_partial_results(self, rows: List[Dict], output_dir: str):
        """Save partial results if Ragas fails"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df = pd.DataFrame(rows)
        results_path = os.path.join(output_dir, f"partial_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"✅ Saved partial results: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Production RAG Evaluation with Ragas - Fully .env integrated",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use settings from .env
  python evaluate_rag.py --type mcq --collection gut_microbiome --sample 10
  
  # Override to use local Ollama
  python evaluate_rag.py --type mcq --collection gut_microbiome --use-local
  
  # Full evaluation with custom settings
  python evaluate_rag.py --type mcq --collection gut_microbiome --top-k 15
        """
    )
    
    # Required arguments
    parser.add_argument("--type", required=True, choices=["open_ended", "mcq", "full"],
                        help="Type of questions to evaluate")
    parser.add_argument("-c", "--collection", required=True,
                        help="Collection name to search")
    
    # Evaluation arguments
    parser.add_argument("--sample", type=int, help="Number of questions to sample")
    parser.add_argument("--top-k", type=int, default=15, help="Number of contexts (default: 15)")
    parser.add_argument("--output-dir", default="./evaluation_results", help="Output directory")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between questions (default: 0)")
    
    # LLM Configuration (overrides .env)
    parser.add_argument("--use-local", action="store_true", help="Use local Ollama (overrides .env)")
    parser.add_argument("--llm-provider", choices=["gemini", "groq"], help="LLM provider")
    parser.add_argument("--ollama-model", help="Ollama model name")
    parser.add_argument("--gemini-model", help="Gemini model name")
    parser.add_argument("--groq-model", help="Groq model name")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    
    # Other config
    parser.add_argument("--embedding-model", help="HuggingFace embedding model")
    parser.add_argument("--db-path", help="Path to ChromaDB database")
    parser.add_argument("--top-k-chatall", type=int, help="Chunks per collection in ChatALL")
    parser.add_argument("--max-history", type=int, help="Max conversation history")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"🚀 RAG EVALUATION WITH RAGAS")
    print(f"{'='*60}\n")
    
    # Initialize configuration
    print("1️⃣ Loading configuration from .env...")
    config = ConfigManager(args)
    config.print_config()
    
    # Initialize RAG pipeline
    print("2️⃣ Initializing RAG pipeline...")
    try:
        rag_pipeline = RAGPipeline(config)
    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        sys.exit(1)
    
    # Initialize evaluator
    print("3️⃣ Initializing Ragas evaluator...")
    evaluator = RAGEvaluator(
        rag_pipeline=rag_pipeline,
        collection_name=args.collection,
        delay=args.delay
    )
    
    # Load dataset
    print(f"4️⃣ Loading {args.type} dataset...")
    try:
        df = evaluator.load_dataset(args.type)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        sys.exit(1)
    
    # Run evaluation
    print(f"5️⃣ Starting evaluation...")
    results = evaluator.evaluate_dataset(
        df,
        k=args.top_k,
        sample_size=args.sample,
        output_dir=args.output_dir
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION COMPLETE")
    print(f"{'='*60}\n")
    
    print(f"Questions evaluated: {results['num_questions']}")
    
    if 'error' in results:
        print(f"\n⚠️  Warning: Evaluation completed with errors")
        print(f"Error: {results['error']}")
    elif results.get('scores'):
        print(f"\n✅ Evaluation successful!")
    
    print(f"\n✅ All results saved in: {args.output_dir}\n")


if __name__ == "__main__":
    main()