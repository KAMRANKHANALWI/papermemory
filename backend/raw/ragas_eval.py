#!/usr/bin/env python3
"""
Enhanced RAGAS Evaluation Script with Full Metrics
Supports: Recall@K, Precision@K, MRR, Faithfulness, Answer Relevancy

Usage:
    python ragas_eval_enhanced.py -c collection -t test_queries.json
    python ragas_eval_enhanced.py --compare configs.json -t tests.json
"""

import os
import argparse
import json
import time
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Import RAG pipeline
from rag_query import RAGPipeline, RAGConfig

# For semantic similarity
from sentence_transformers import SentenceTransformer
import numpy as np


class EnhancedRAGASEvaluator:
    """Enhanced evaluator with full RAGAS metrics"""
    
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.results = []
        
        # Load semantic similarity model for answer relevancy
        print("🔧 Loading semantic similarity model...")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Similarity model loaded")
    
    def load_test_queries(self, filepath: str) -> List[Dict]:
        """Load test queries from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        test_cases = data.get("test_cases", data)
        print(f"📂 Loaded {len(test_cases)} test cases")
        return test_cases
    
    def calculate_recall_at_k(
        self, 
        retrieved_docs: List[str],
        expected_docs: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K
        
        Recall@K = (Number of expected docs in top-K) / (Total expected docs)
        
        Args:
            retrieved_docs: List of retrieved document filenames
            expected_docs: List of expected document filenames
            k: Cut-off position
        
        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not expected_docs:
            return 0.0
        
        # Normalize filenames (remove paths, lowercase)
        retrieved_set = set([os.path.basename(doc).lower() for doc in retrieved_docs[:k]])
        expected_set = set([os.path.basename(doc).lower() for doc in expected_docs])
        
        # Count how many expected docs are in retrieved
        found = len(retrieved_set.intersection(expected_set))
        
        return found / len(expected_set)
    
    def calculate_precision_at_k(
        self,
        retrieved_docs: List[str],
        expected_docs: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K
        
        Precision@K = (Number of relevant docs in top-K) / K
        
        Args:
            retrieved_docs: List of retrieved document filenames
            expected_docs: List of expected document filenames
            k: Cut-off position
        
        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if k == 0 or not retrieved_docs:
            return 0.0
        
        # Normalize filenames
        retrieved_set = set([os.path.basename(doc).lower() for doc in retrieved_docs[:k]])
        expected_set = set([os.path.basename(doc).lower() for doc in expected_docs])
        
        # Count relevant docs in top-K
        relevant = len(retrieved_set.intersection(expected_set))
        
        return relevant / min(k, len(retrieved_docs))
    
    def calculate_mrr(
        self,
        retrieved_docs: List[str],
        expected_docs: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        MRR = 1 / (position of first relevant document)
        
        Args:
            retrieved_docs: List of retrieved document filenames (ordered)
            expected_docs: List of expected document filenames
        
        Returns:
            MRR score (0.0 to 1.0)
        """
        if not expected_docs or not retrieved_docs:
            return 0.0
        
        # Normalize filenames
        expected_set = set([os.path.basename(doc).lower() for doc in expected_docs])
        
        # Find position of first relevant document
        for i, doc in enumerate(retrieved_docs, start=1):
            normalized = os.path.basename(doc).lower()
            if normalized in expected_set:
                return 1.0 / i
        
        return 0.0  # No relevant document found
    
    def calculate_faithfulness(
        self,
        answer: str,
        context: str,
        query: str
    ) -> Dict:
        """
        Calculate faithfulness - is the answer grounded in the context?
        
        Uses LLM to verify if answer claims are supported by context
        
        Returns:
            Dict with faithfulness score and details
        """
        try:
            faithfulness_prompt = f"""You are an evaluator assessing whether an answer is faithful to the given context.

Context:
{context[:2000]}  

Answer:
{answer}

Task: Determine if the answer is fully supported by the context.
- Score 1.0: All claims in answer are in context
- Score 0.5: Some claims are in context, some are not
- Score 0.0: Answer contradicts or is unrelated to context

Respond with ONLY a number between 0.0 and 1.0, nothing else.

Faithfulness score:"""

            messages = [{"role": "user", "content": faithfulness_prompt}]
            response = self.pipeline.llm.invoke(messages)
            
            # Extract score
            score_text = response.content.strip()
            
            # Try to parse the score
            try:
                score = float(score_text)
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except:
                # If LLM doesn't return a number, extract from text
                import re
                numbers = re.findall(r'0\.\d+|1\.0|0|1', score_text)
                score = float(numbers[0]) if numbers else 0.5
            
            return {
                "score": score,
                "assessment": "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"
            }
        
        except Exception as e:
            print(f"⚠️  Faithfulness calculation error: {e}")
            return {"score": 0.5, "assessment": "unknown"}
    
    def calculate_answer_relevancy(
        self,
        query: str,
        answer: str
    ) -> float:
        """
        Calculate answer relevancy using semantic similarity
        
        Measures how relevant the answer is to the query
        
        Returns:
            Relevancy score (0.0 to 1.0)
        """
        try:
            # Encode query and answer
            query_embedding = self.similarity_model.encode(query)
            answer_embedding = self.similarity_model.encode(answer)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, answer_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(answer_embedding)
            )
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            relevancy = (similarity + 1) / 2
            
            return float(relevancy)
        
        except Exception as e:
            print(f"⚠️  Answer relevancy error: {e}")
            return 0.5
    
    def calculate_context_precision(
        self,
        retrieved_chunks: List[Dict],
        ground_truth: str
    ) -> float:
        """
        Calculate context precision - are retrieved chunks relevant?
        
        Checks if ground truth concepts appear in retrieved chunks
        
        Returns:
            Context precision score (0.0 to 1.0)
        """
        if not retrieved_chunks or not ground_truth:
            return 0.0
        
        # Extract key terms from ground truth (simple approach)
        ground_truth_words = set(ground_truth.lower().split())
        
        # Remove common words
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'])
        ground_truth_words = ground_truth_words - stop_words
        
        # Count how many chunks contain ground truth concepts
        relevant_chunks = 0
        for chunk in retrieved_chunks:
            chunk_text = chunk.get('content', '').lower()
            chunk_words = set(chunk_text.split())
            
            # If chunk shares words with ground truth, it's relevant
            if len(ground_truth_words.intersection(chunk_words)) >= 2:
                relevant_chunks += 1
        
        return relevant_chunks / len(retrieved_chunks)
    
    def evaluate_single_query(
        self,
        test_case: Dict,
        collection_name: str,
        k_values: List[int] = [5, 10, 15, 20]
    ) -> Dict:
        """
        Evaluate a single test query with full metrics
        
        Args:
            test_case: Test case dict with query, ground_truth, expected_docs
            collection_name: Collection to search
            k_values: List of K values for Recall@K and Precision@K
        
        Returns:
            Comprehensive evaluation result
        """
        query = test_case["query"]
        ground_truth = test_case.get("ground_truth_answer", "")
        expected_docs = test_case.get("expected_documents", [])
        
        print(f"\n{'='*60}")
        print(f"📝 Query: {query[:80]}...")
        print(f"{'='*60}")
        
        # Start timing
        start_time = time.time()
        
        # Run RAG query
        result = self.pipeline.query(
            query,
            collection_name=collection_name,
            classify=False
        )
        
        latency = time.time() - start_time
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return {
                "test_id": test_case.get("id", "unknown"),
                "query": query,
                "error": result["error"],
                "latency": latency,
                "success": False
            }
        
        # Extract data
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        
        # Get retrieved document filenames (ordered by similarity)
        retrieved_docs = [s["filename"] for s in sources]
        
        # Calculate metrics
        print(f"📊 Calculating metrics...")
        
        # 1. Retrieval Metrics
        recall_scores = {}
        precision_scores = {}
        
        for k in k_values:
            recall_scores[f"recall@{k}"] = self.calculate_recall_at_k(
                retrieved_docs, expected_docs, k
            )
            precision_scores[f"precision@{k}"] = self.calculate_precision_at_k(
                retrieved_docs, expected_docs, k
            )
        
        mrr_score = self.calculate_mrr(retrieved_docs, expected_docs)
        
        # 2. Generation Metrics
        faithfulness = self.calculate_faithfulness(
            answer,
            "\n".join([s["content"] for s in sources[:5]]),
            query
        )
        
        answer_relevancy = self.calculate_answer_relevancy(query, answer)
        
        context_precision = self.calculate_context_precision(sources, ground_truth)
        
        # 3. Basic Metrics
        avg_similarity = sum([s["similarity"] for s in sources]) / len(sources) if sources else 0.0
        
        # Compile results
        eval_result = {
            "test_id": test_case.get("id", "unknown"),
            "query": query,
            "ground_truth": ground_truth,
            "answer": answer,
            "difficulty": test_case.get("difficulty", "unknown"),
            "category": test_case.get("category", "unknown"),
            
            # Retrieval Metrics
            "retrieval_metrics": {
                **recall_scores,
                **precision_scores,
                "mrr": mrr_score,
                "num_sources": len(sources),
                "avg_similarity": avg_similarity
            },
            
            # Generation Metrics
            "generation_metrics": {
                "faithfulness": faithfulness["score"],
                "faithfulness_assessment": faithfulness["assessment"],
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision
            },
            
            # Performance Metrics
            "performance": {
                "latency": latency,
                "success": True
            },
            
            # Retrieved documents
            "retrieved_documents": retrieved_docs[:10],
            "expected_documents": expected_docs,
            
            # Top sources
            "top_sources": [
                {
                    "filename": s["filename"],
                    "similarity": s["similarity"],
                    "content_preview": s["content"][:150]
                }
                for s in sources[:3]
            ]
        }
        
        # Print summary
        print(f"\n✅ Evaluation Complete:")
        print(f"   Recall@5: {recall_scores.get('recall@5', 0):.3f}")
        print(f"   Recall@10: {recall_scores.get('recall@10', 0):.3f}")
        print(f"   Precision@5: {precision_scores.get('precision@5', 0):.3f}")
        print(f"   MRR: {mrr_score:.3f}")
        print(f"   Faithfulness: {faithfulness['score']:.3f} ({faithfulness['assessment']})")
        print(f"   Answer Relevancy: {answer_relevancy:.3f}")
        print(f"   Latency: {latency:.2f}s")
        
        return eval_result
    
    def evaluate_test_set(
        self,
        test_cases: List[Dict],
        collection_name: str,
        k_values: List[int] = [5, 10, 15, 20]
    ) -> Dict:
        """Evaluate all test cases"""
        print(f"\n{'='*70}")
        print(f"🔬 ENHANCED RAGAS EVALUATION")
        print(f"{'='*70}")
        print(f"Collection: {collection_name}")
        print(f"Test Cases: {len(test_cases)}")
        print(f"K Values: {k_values}")
        print(f"{'='*70}\n")
        
        results = []
        successful = 0
        failed = 0
        
        # Aggregate metrics
        total_latency = 0
        recall_sums = defaultdict(float)
        precision_sums = defaultdict(float)
        mrr_sum = 0
        faithfulness_sum = 0
        relevancy_sum = 0
        context_precision_sum = 0
        
        # Track by difficulty and category
        by_difficulty = defaultdict(list)
        by_category = defaultdict(list)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Evaluating test case: {test_case.get('id', 'unknown')}")
            
            eval_result = self.evaluate_single_query(
                test_case,
                collection_name,
                k_values
            )
            
            results.append(eval_result)
            
            if eval_result.get("success", False):
                successful += 1
                
                # Accumulate metrics
                total_latency += eval_result["performance"]["latency"]
                
                for k in k_values:
                    recall_sums[f"recall@{k}"] += eval_result["retrieval_metrics"][f"recall@{k}"]
                    precision_sums[f"precision@{k}"] += eval_result["retrieval_metrics"][f"precision@{k}"]
                
                mrr_sum += eval_result["retrieval_metrics"]["mrr"]
                faithfulness_sum += eval_result["generation_metrics"]["faithfulness"]
                relevancy_sum += eval_result["generation_metrics"]["answer_relevancy"]
                context_precision_sum += eval_result["generation_metrics"]["context_precision"]
                
                # Group by difficulty and category
                difficulty = eval_result.get("difficulty", "unknown")
                category = eval_result.get("category", "unknown")
                
                by_difficulty[difficulty].append(eval_result)
                by_category[category].append(eval_result)
            else:
                failed += 1
        
        # Calculate averages
        avg_metrics = {}
        if successful > 0:
            avg_metrics = {
                "avg_latency": total_latency / successful,
                **{f"avg_{k}": recall_sums[k] / successful for k in recall_sums.keys()},
                **{f"avg_{k}": precision_sums[k] / successful for k in precision_sums.keys()},
                "avg_mrr": mrr_sum / successful,
                "avg_faithfulness": faithfulness_sum / successful,
                "avg_answer_relevancy": relevancy_sum / successful,
                "avg_context_precision": context_precision_sum / successful
            }
        
        # Compile summary
        summary = {
            "evaluation_info": {
                "collection": collection_name,
                "total_test_cases": len(test_cases),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(test_cases) if test_cases else 0,
                "timestamp": datetime.now().isoformat()
            },
            
            "aggregate_metrics": avg_metrics,
            
            "metrics_by_difficulty": {
                difficulty: self._calculate_group_metrics(cases)
                for difficulty, cases in by_difficulty.items()
            },
            
            "metrics_by_category": {
                category: self._calculate_group_metrics(cases)
                for category, cases in by_category.items()
            },
            
            "detailed_results": results
        }
        
        return summary
    
    def _calculate_group_metrics(self, results: List[Dict]) -> Dict:
        """Calculate average metrics for a group of results"""
        if not results:
            return {}
        
        successful = [r for r in results if r.get("success", False)]
        if not successful:
            return {"count": 0}
        
        avg_recall_5 = sum(r["retrieval_metrics"]["recall@5"] for r in successful) / len(successful)
        avg_precision_5 = sum(r["retrieval_metrics"]["precision@5"] for r in successful) / len(successful)
        avg_faithfulness = sum(r["generation_metrics"]["faithfulness"] for r in successful) / len(successful)
        avg_relevancy = sum(r["generation_metrics"]["answer_relevancy"] for r in successful) / len(successful)
        
        return {
            "count": len(results),
            "avg_recall@5": avg_recall_5,
            "avg_precision@5": avg_precision_5,
            "avg_faithfulness": avg_faithfulness,
            "avg_answer_relevancy": avg_relevancy
        }
    
    def compare_configurations(
        self,
        configs: List[Dict],
        test_cases: List[Dict]
    ) -> Dict:
        """Compare multiple RAG configurations"""
        print(f"\n{'='*70}")
        print(f"🔬 CONFIGURATION COMPARISON")
        print(f"{'='*70}")
        print(f"Configurations: {len(configs)}")
        print(f"Test Cases: {len(test_cases)}")
        print(f"{'='*70}\n")
        
        comparisons = []
        
        for i, config_dict in enumerate(configs, 1):
            config_name = config_dict.get('name', f'Config {i}')
            print(f"\n{'='*70}")
            print(f"[{i}/{len(configs)}] Evaluating: {config_name}")
            print(f"{'='*70}")
            
            # Create config
            class Args:
                pass
            args = Args()
            for key, value in config_dict.items():
                setattr(args, key, value)
            
            # Set defaults
            for attr in ['use_local', 'llm_provider', 'ollama_model', 'gemini_model',
                        'groq_model', 'embedding_model', 'top_k', 'top_k_chatall',
                        'db_path', 'temperature', 'max_history']:
                if not hasattr(args, attr):
                    setattr(args, attr, None)
            
            try:
                config = RAGConfig(args)
                pipeline = RAGPipeline(config)
                
                # Update evaluator's pipeline
                self.pipeline = pipeline
                
                # Evaluate
                collection_name = config_dict.get('collection_name', 'documents')
                results = self.evaluate_test_set(test_cases, collection_name)
                
                results['config_name'] = config_name
                results['config'] = config_dict
                
                comparisons.append(results)
                
            except Exception as e:
                print(f"❌ Failed to evaluate config: {e}")
                comparisons.append({
                    'config_name': config_name,
                    'error': str(e)
                })
        
        return {
            "comparison_date": datetime.now().isoformat(),
            "num_configs": len(configs),
            "num_test_cases": len(test_cases),
            "comparisons": comparisons
        }
    
    def save_results(self, results: Dict, output_file: str):
        """Save results to JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def print_summary(self, results: Dict):
        """Print beautiful summary"""
        print(f"\n{'='*70}")
        print(f"📊 ENHANCED EVALUATION SUMMARY")
        print(f"{'='*70}\n")
        
        if "comparisons" in results:
            # Comparison mode
            self._print_comparison_summary(results)
        else:
            # Single evaluation mode
            self._print_single_summary(results)
    
    def _print_single_summary(self, results: Dict):
        """Print single evaluation summary"""
        info = results["evaluation_info"]
        metrics = results["aggregate_metrics"]
        
        print(f"Collection: {info['collection']}")
        print(f"Test Cases: {info['total_test_cases']}")
        print(f"Success Rate: {info['success_rate']*100:.1f}% ({info['successful']}/{info['total_test_cases']})")
        print(f"\n{'─'*70}")
        print(f"📈 RETRIEVAL METRICS")
        print(f"{'─'*70}")
        print(f"Recall@5:  {metrics.get('avg_recall@5', 0):.3f}")
        print(f"Recall@10: {metrics.get('avg_recall@10', 0):.3f}")
        print(f"Recall@15: {metrics.get('avg_recall@15', 0):.3f}")
        print(f"Recall@20: {metrics.get('avg_recall@20', 0):.3f}")
        print(f"\nPrecision@5:  {metrics.get('avg_precision@5', 0):.3f}")
        print(f"Precision@10: {metrics.get('avg_precision@10', 0):.3f}")
        print(f"\nMRR: {metrics.get('avg_mrr', 0):.3f}")
        
        print(f"\n{'─'*70}")
        print(f"💡 GENERATION METRICS")
        print(f"{'─'*70}")
        print(f"Faithfulness:      {metrics.get('avg_faithfulness', 0):.3f}")
        print(f"Answer Relevancy:  {metrics.get('avg_answer_relevancy', 0):.3f}")
        print(f"Context Precision: {metrics.get('avg_context_precision', 0):.3f}")
        
        print(f"\n{'─'*70}")
        print(f"⚡ PERFORMANCE")
        print(f"{'─'*70}")
        print(f"Avg Latency: {metrics.get('avg_latency', 0):.2f}s")
        
        # Breakdown by difficulty
        if results.get("metrics_by_difficulty"):
            print(f"\n{'─'*70}")
            print(f"📊 BY DIFFICULTY")
            print(f"{'─'*70}")
            for difficulty, metrics in results["metrics_by_difficulty"].items():
                if metrics.get("count", 0) > 0:
                    print(f"\n{difficulty.upper()} ({metrics['count']} cases):")
                    print(f"  Recall@5: {metrics.get('avg_recall@5', 0):.3f}")
                    print(f"  Faithfulness: {metrics.get('avg_faithfulness', 0):.3f}")
        
        # Breakdown by category
        if results.get("metrics_by_category"):
            print(f"\n{'─'*70}")
            print(f"📊 BY CATEGORY")
            print(f"{'─'*70}")
            for category, metrics in results["metrics_by_category"].items():
                if metrics.get("count", 0) > 0:
                    print(f"\n{category.upper()} ({metrics['count']} cases):")
                    print(f"  Recall@5: {metrics.get('avg_recall@5', 0):.3f}")
                    print(f"  Faithfulness: {metrics.get('avg_faithfulness', 0):.3f}")
        
        print(f"\n{'='*70}\n")
    
    def _print_comparison_summary(self, results: Dict):
        """Print comparison summary"""
        print(f"Configurations Compared: {results['num_configs']}")
        print(f"Test Cases: {results['num_test_cases']}")
        print(f"\n{'='*70}")
        
        for comp in results["comparisons"]:
            if "error" in comp:
                print(f"\n❌ {comp['config_name']}: {comp['error']}")
                continue
            
            info = comp["evaluation_info"]
            metrics = comp["aggregate_metrics"]
            
            print(f"\n🔧 {comp['config_name']}")
            print(f"{'─'*70}")
            print(f"Success Rate: {info['success_rate']*100:.1f}%")
            print(f"Recall@5:  {metrics.get('avg_recall@5', 0):.3f}")
            print(f"Recall@10: {metrics.get('avg_recall@10', 0):.3f}")
            print(f"Precision@5: {metrics.get('avg_precision@5', 0):.3f}")
            print(f"MRR: {metrics.get('avg_mrr', 0):.3f}")
            print(f"Faithfulness: {metrics.get('avg_faithfulness', 0):.3f}")
            print(f"Answer Relevancy: {metrics.get('avg_answer_relevancy', 0):.3f}")
            print(f"Avg Latency: {metrics.get('avg_latency', 0):.2f}s")
        
        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced RAGAS Evaluation with Full Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python ragas_eval_enhanced.py -c gut_microbiome_reviews -t test_queries.json
  
  # Save detailed results
  python ragas_eval_enhanced.py -c collection -t tests.json -o results/eval.json
  
  # Compare configurations
  python ragas_eval_enhanced.py --compare configs.json -t tests.json
  
  # Custom K values for Recall@K
  python ragas_eval_enhanced.py -c collection -t tests.json --k-values 5 10 20
        """
    )
    
    # Input arguments
    parser.add_argument("-c", "--collection", help="Collection name to evaluate")
    parser.add_argument("-t", "--test-queries", required=True, help="Path to test queries JSON")
    parser.add_argument("--compare", help="Path to configurations JSON for comparison")
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 15, 20], help="K values for Recall@K and Precision@K")
    
    # Output arguments
    parser.add_argument("-o", "--output", help="Output file for results (JSON)")
    
    # RAG Configuration
    parser.add_argument("--use-local", action="store_true")
    parser.add_argument("--llm-provider", choices=["gemini", "groq"])
    parser.add_argument("--ollama-model")
    parser.add_argument("--gemini-model")
    parser.add_argument("--groq-model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--embedding-model")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-k-chatall", type=int)
    parser.add_argument("--db-path")
    parser.add_argument("--max-history", type=int)
    
    args = parser.parse_args()
    
    # Load test queries
    if not os.path.exists(args.test_queries):
        print(f"❌ Test queries file not found: {args.test_queries}")
        return
    
    # Handle comparison mode
    if args.compare:
        if not os.path.exists(args.compare):
            print(f"❌ Configs file not found: {args.compare}")
            return
        
        with open(args.compare, 'r') as f:
            data = json.load(f)
        
        configs = data.get("configs", [])
        
        # Initialize with first config to load test queries
        config = RAGConfig(args)
        pipeline = RAGPipeline(config)
        evaluator = EnhancedRAGASEvaluator(pipeline)
        
        test_queries = evaluator.load_test_queries(args.test_queries)
        
        # Run comparison
        results = evaluator.compare_configurations(configs, test_queries)
        evaluator.print_summary(results)
        
        if args.output:
            evaluator.save_results(results, args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            evaluator.save_results(results, f"eval_comparison_{timestamp}.json")
        
        return
    
    # Single evaluation mode
    if not args.collection:
        print("❌ Error: Collection name required (-c)")
        parser.print_help()
        return
    
    # Initialize
    config = RAGConfig(args)
    pipeline = RAGPipeline(config)
    evaluator = EnhancedRAGASEvaluator(pipeline)
    
    # Load test queries
    test_queries = evaluator.load_test_queries(args.test_queries)
    
    # Run evaluation
    results = evaluator.evaluate_test_set(
        test_queries,
        args.collection,
        k_values=args.k_values
    )
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    if args.output:
        evaluator.save_results(results, args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluator.save_results(results, f"eval_results_{timestamp}.json")


if __name__ == "__main__":
    main()
