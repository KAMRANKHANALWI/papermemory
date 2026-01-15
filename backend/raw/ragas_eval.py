#!/usr/bin/env python3
"""
RAGAS Evaluation Script for RAG Pipeline
Test different configurations and measure performance

Usage:
    python ragas_eval.py -c my_collection -t test_queries.json
    python ragas_eval.py -c my_collection --generate-test-set
    python ragas_eval.py --compare-configs configs.yaml
"""

import argparse
import json
import time
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime

# Import RAG pipeline
from rag_query import RAGPipeline, RAGConfig

# Try to import RAGAS (optional)
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠️  RAGAS not installed. Install with: pip install ragas")


class RAGASEvaluator:
    """Evaluate RAG pipeline using RAGAS metrics"""
    
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.results = []
    
    def load_test_queries(self, filepath: str) -> List[Dict]:
        """Load test queries from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Expected format: [{"query": "...", "ground_truth": "...", "context": "..."}]
        return data.get("test_cases", data)
    
    def generate_test_set(self, collection_name: str, num_queries: int = 10) -> List[Dict]:
        """Generate test queries from documents"""
        print(f"🔧 Generating {num_queries} test queries from collection...")
        
        # Get PDFs from collection
        pdf_info = self.pipeline.list_pdfs_in_collection(collection_name)
        
        if pdf_info.get("error"):
            print(f"❌ Error: {pdf_info['error']}")
            return []
        
        # Generate queries using LLM
        test_queries = []
        pdfs = pdf_info.get("pdfs", [])[:5]  # Use first 5 PDFs
        
        for pdf in pdfs:
            # Search for sample content
            vectorstore = self.pipeline.get_vectorstore(collection_name)
            if not vectorstore:
                continue
            
            try:
                # Get some chunks from this PDF
                results = vectorstore.similarity_search(
                    "summary overview",  # Generic query
                    k=2,
                    filter={"filename": pdf["filename"]}
                )
                
                for doc in results[:2]:
                    # Generate question from content
                    question_prompt = f"""Based on this text, generate a specific factual question that can be answered from this content:

Text: {doc.page_content[:500]}

Generate ONE specific question:"""
                    
                    try:
                        messages = [{"role": "user", "content": question_prompt}]
                        response = self.pipeline.llm.invoke(messages)
                        question = response.content.strip()
                        
                        test_queries.append({
                            "query": question,
                            "ground_truth": doc.page_content[:300],
                            "source_file": pdf["filename"],
                            "collection": collection_name
                        })
                        
                        if len(test_queries) >= num_queries:
                            break
                    except Exception as e:
                        print(f"⚠️  Error generating question: {e}")
                        continue
            
            except Exception as e:
                print(f"⚠️  Error processing {pdf['filename']}: {e}")
                continue
            
            if len(test_queries) >= num_queries:
                break
        
        print(f"✅ Generated {len(test_queries)} test queries")
        return test_queries
    
    def evaluate_query(
        self, 
        query: str, 
        collection_name: str,
        ground_truth: str = None
    ) -> Dict:
        """Evaluate a single query"""
        start_time = time.time()
        
        # Run query
        result = self.pipeline.query(
            query,
            collection_name=collection_name,
            classify=False  # Skip classification for eval
        )
        
        latency = time.time() - start_time
        
        if "error" in result:
            return {
                "query": query,
                "error": result["error"],
                "latency": latency
            }
        
        # Basic metrics
        eval_result = {
            "query": query,
            "answer": result.get("answer", ""),
            "num_sources": result.get("num_sources", 0),
            "latency": latency,
            "classification": result.get("classification", "content_search")
        }
        
        # Add sources
        if result.get("sources"):
            eval_result["sources"] = [
                {
                    "filename": s["filename"],
                    "similarity": s["similarity"],
                    "content_preview": s["content"][:200]
                }
                for s in result["sources"][:5]
            ]
            
            # Calculate average similarity
            similarities = [s["similarity"] for s in result["sources"]]
            eval_result["avg_similarity"] = sum(similarities) / len(similarities)
            eval_result["max_similarity"] = max(similarities)
        
        # Add ground truth if provided
        if ground_truth:
            eval_result["ground_truth"] = ground_truth
        
        return eval_result
    
    def evaluate_test_set(
        self, 
        test_queries: List[Dict],
        collection_name: str
    ) -> Dict:
        """Evaluate all test queries"""
        print(f"\n{'='*60}")
        print(f"📊 Evaluating {len(test_queries)} queries")
        print(f"{'='*60}\n")
        
        results = []
        successful = 0
        failed = 0
        total_latency = 0
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"[{i}/{len(test_queries)}] Evaluating: {test_case['query'][:60]}...")
            
            eval_result = self.evaluate_query(
                test_case["query"],
                collection_name,
                test_case.get("ground_truth")
            )
            
            results.append(eval_result)
            
            if "error" in eval_result:
                failed += 1
                print(f"   ❌ Failed: {eval_result['error']}")
            else:
                successful += 1
                total_latency += eval_result["latency"]
                print(f"   ✅ Success (latency: {eval_result['latency']:.2f}s, sources: {eval_result['num_sources']})")
        
        # Calculate aggregate metrics
        avg_latency = total_latency / successful if successful > 0 else 0
        
        # Calculate average similarities
        similarities = [r.get("avg_similarity", 0) for r in results if "avg_similarity" in r]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        summary = {
            "total_queries": len(test_queries),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(test_queries) if test_queries else 0,
            "avg_latency": avg_latency,
            "avg_similarity": avg_similarity,
            "results": results
        }
        
        return summary
    
    def compare_configurations(self, configs: List[Dict], test_queries: List[Dict]) -> Dict:
        """Compare different RAG configurations"""
        print(f"\n{'='*60}")
        print(f"🔬 Comparing {len(configs)} configurations")
        print(f"{'='*60}\n")
        
        comparisons = []
        
        for i, config_dict in enumerate(configs, 1):
            print(f"\n[Config {i}] {config_dict.get('name', f'Config {i}')}")
            print(f"{'='*40}")
            
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
            
            config = RAGConfig(args)
            
            # Create pipeline
            try:
                pipeline = RAGPipeline(config)
                self.pipeline = pipeline
                
                # Evaluate
                collection_name = config_dict.get('collection_name', 'documents')
                results = self.evaluate_test_set(test_queries, collection_name)
                
                results['config_name'] = config_dict.get('name', f'Config {i}')
                results['config'] = config_dict
                
                comparisons.append(results)
                
            except Exception as e:
                print(f"❌ Failed to initialize config: {e}")
                comparisons.append({
                    'config_name': config_dict.get('name', f'Config {i}'),
                    'error': str(e)
                })
        
        return {
            "comparison_date": datetime.now().isoformat(),
            "num_configs": len(configs),
            "num_test_queries": len(test_queries),
            "comparisons": comparisons
        }
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print(f"\n{'='*60}")
        print(f"📊 EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        if "comparisons" in results:
            # Comparison results
            print(f"\nConfiguration Comparison:")
            print(f"{'='*60}")
            
            for comp in results["comparisons"]:
                if "error" in comp:
                    print(f"\n❌ {comp['config_name']}: {comp['error']}")
                    continue
                
                print(f"\n🔧 {comp['config_name']}")
                print(f"   Success Rate: {comp['success_rate']*100:.1f}%")
                print(f"   Avg Latency: {comp['avg_latency']:.2f}s")
                print(f"   Avg Similarity: {comp['avg_similarity']:.3f}")
                print(f"   Successful: {comp['successful']}/{comp['total_queries']}")
        else:
            # Single evaluation results
            print(f"\nTotal Queries: {results['total_queries']}")
            print(f"Successful: {results['successful']}")
            print(f"Failed: {results['failed']}")
            print(f"Success Rate: {results['success_rate']*100:.1f}%")
            print(f"Avg Latency: {results['avg_latency']:.2f}s")
            print(f"Avg Similarity: {results['avg_similarity']:.3f}")
        
        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation for RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with test queries file
  python ragas_eval.py -c my_collection -t test_queries.json
  
  # Generate test queries and evaluate
  python ragas_eval.py -c my_collection --generate-test-set --num-queries 20
  
  # Save results
  python ragas_eval.py -c my_collection -t tests.json -o results/eval_20250110.json
  
  # Compare configurations
  python ragas_eval.py --compare configs.json -t tests.json
  
Test queries JSON format:
{
  "test_cases": [
    {
      "query": "What is machine learning?",
      "ground_truth": "Expected answer or relevant context",
      "source_file": "optional_source.pdf"
    }
  ]
}

Configs JSON format:
{
  "configs": [
    {
      "name": "Config 1",
      "collection_name": "documents",
      "top_k": 10,
      "embedding_model": "all-MiniLM-L6-v2"
    }
  ]
}
        """
    )
    
    # Input arguments
    parser.add_argument("-c", "--collection", help="Collection name to evaluate")
    parser.add_argument("-t", "--test-queries", help="Path to test queries JSON file")
    parser.add_argument("--generate-test-set", action="store_true", help="Generate test queries from collection")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of queries to generate")
    
    # Configuration
    parser.add_argument("--compare", help="Path to configurations JSON for comparison")
    
    # Output arguments
    parser.add_argument("-o", "--output", help="Output file for results (JSON)")
    parser.add_argument("--save-test-queries", help="Save generated test queries to file")
    
    # RAG Configuration (same as rag_query.py)
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
    
    # Initialize configuration
    config = RAGConfig(args)
    
    # Initialize pipeline
    try:
        pipeline = RAGPipeline(config)
        evaluator = RAGASEvaluator(pipeline)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Handle comparison mode
    if args.compare:
        with open(args.compare, 'r') as f:
            data = json.load(f)
        
        configs = data.get("configs", [])
        
        if not args.test_queries:
            print("❌ Error: --test-queries required for comparison mode")
            return
        
        test_queries = evaluator.load_test_queries(args.test_queries)
        
        results = evaluator.compare_configurations(configs, test_queries)
        evaluator.print_summary(results)
        
        if args.output:
            evaluator.save_results(results, args.output)
        
        return
    
    # Validate inputs
    if not args.collection:
        print("❌ Error: Collection name required (-c)")
        parser.print_help()
        return
    
    # Load or generate test queries
    if args.generate_test_set:
        test_queries = evaluator.generate_test_set(args.collection, args.num_queries)
        
        if args.save_test_queries:
            with open(args.save_test_queries, 'w') as f:
                json.dump({"test_cases": test_queries}, f, indent=2)
            print(f"💾 Test queries saved to: {args.save_test_queries}")
        
        if not test_queries:
            print("❌ No test queries generated")
            return
    
    elif args.test_queries:
        test_queries = evaluator.load_test_queries(args.test_queries)
        print(f"📂 Loaded {len(test_queries)} test queries")
    
    else:
        print("❌ Error: Either --test-queries or --generate-test-set required")
        parser.print_help()
        return
    
    # Run evaluation
    results = evaluator.evaluate_test_set(test_queries, args.collection)
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    if args.output:
        evaluator.save_results(results, args.output)
    else:
        # Auto-save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"eval_results_{timestamp}.json"
        evaluator.save_results(results, output_file)


if __name__ == "__main__":
    main()
