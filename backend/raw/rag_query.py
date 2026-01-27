#!/usr/bin/env python3
"""
Minimalistic RAG Query Pipeline - Command Line Interface
For testing, evaluation, and experimentation with different configurations

Usage:
    python rag_query.py -q "What is machine learning?" -c my_collection
    python rag_query.py -q "Explain neural networks" --chatall
    python rag_query.py -q "Summarize research.pdf" -c documents
    python rag_query.py --interactive -c my_collection
"""

import os
import argparse
import chromadb
from typing import List, Dict, Tuple, Optional
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import re

# Load environment
load_dotenv()

# ==================== CONFIGURATION ====================

class RAGConfig:
    """Configuration for RAG pipeline"""
    def __init__(self, args):
        # LLM Configuration
        self.use_local_llm = args.use_local or os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        self.llm_provider = args.llm_provider or os.getenv("DEFAULT_MODEL_PROVIDER", "gemini")
        self.ollama_model = args.ollama_model or os.getenv("OLLAMA_MODEL", "llama3.1:latest")
        self.gemini_model = args.gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.groq_model = args.groq_model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        
        # Embedding Configuration
        self.embedding_model = args.embedding_model or "all-MiniLM-L6-v2"
        
        # Retrieval Configuration
        self.top_k = args.top_k or 10
        # self.top_k = args.top_k or 25
        self.top_k_chatall = args.top_k_chatall or 1
        
        # Database Configuration
        self.db_path = args.db_path or "data_raw/chroma_db"
        
        # Query Configuration
        self.temperature = args.temperature or 0.1
        self.max_history = args.max_history or 5


# ==================== QUERY CLASSIFIER ====================

class QueryClassifier:
    """LLM-based query classifier"""
    
    CLASSIFICATION_PROMPT = """You are a query classifier for a document Q&A system. Classify the user's query into ONE of these categories:

CATEGORIES:
1. "list_pdfs" - User wants to see list of available documents/PDFs
2. "count_pdfs" - User wants to know how many documents are available
3. "file_specific_search" - User asks about a SPECIFIC PDF file by name
4. "content_search" - User wants to search/ask about document content (default)

EXAMPLES:

User: "List all the PDFs you have"
Classification: list_pdfs

User: "What documents are available?"
Classification: list_pdfs

User: "How many PDFs do you have?"
Classification: count_pdfs

User: "What does protein_interaction.pdf say about databases?"
Classification: file_specific_search

User: "Summarize research_paper.pdf"
Classification: file_specific_search

User: "What is mentioned about quantum physics?"
Classification: content_search

User: "Explain the methodology used"
Classification: content_search

RULES:
- Only respond with ONE word: list_pdfs, count_pdfs, file_specific_search, or content_search
- Use "file_specific_search" ONLY when user mentions a specific filename (with .pdf extension)
- When in doubt, use "content_search"

Now classify this query:
User: "{query}"
Classification:"""

    def __init__(self, llm):
        self.llm = llm
    
    def classify(self, query: str) -> Tuple[str, Optional[str]]:
        """Classify query and extract filename if needed"""
        try:
            prompt = self.CLASSIFICATION_PROMPT.format(query=query)
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            classification = response.content.strip().lower()
            
            # Normalize classification
            if 'file' in classification and 'specific' in classification:
                classification = 'file_specific_search'
            elif 'list' in classification:
                classification = 'list_pdfs'
            elif 'count' in classification:
                classification = 'count_pdfs'
            elif classification not in ['list_pdfs', 'count_pdfs', 'file_specific_search', 'content_search']:
                classification = 'content_search'
            
            # Extract filename
            filename = None
            if classification == 'file_specific_search':
                filename = self._extract_filename(query)
                if not filename:
                    classification = 'content_search'
            
            return classification, filename
        
        except Exception as e:
            print(f"⚠️  Classification error: {e}")
            return 'content_search', None
    
    def _extract_filename(self, query: str) -> Optional[str]:
        """Extract PDF filename from query"""
        pattern = r'([a-zA-Z0-9_\-]+\.pdf)'
        matches = re.findall(pattern, query, re.IGNORECASE)
        return matches[0] if matches else None


# ==================== RAG PIPELINE ====================

class RAGPipeline:
    """Core RAG pipeline for querying documents"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = self._initialize_llm()
        self.embedding_model = HuggingFaceEmbeddings(model_name=config.embedding_model)
        self.chroma_client = chromadb.PersistentClient(path=config.db_path)
        self.classifier = QueryClassifier(self.llm)
        
        print(f"   RAG Pipeline initialized")
        print(f"   LLM: {self._get_llm_name()}")
        print(f"   Embeddings: {config.embedding_model}")
        print(f"   Database: {config.db_path}")
    
    def _initialize_llm(self):
        """Initialize LLM based on configuration"""
        if self.config.use_local_llm:
            print(f"🔧 Initializing Ollama: {self.config.ollama_model}")
            return ChatOllama(
                model=self.config.ollama_model,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
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
            raise ValueError("No valid LLM configuration found. Set API keys in .env")
    
    def _get_llm_name(self) -> str:
        """Get current LLM name"""
        if self.config.use_local_llm:
            return f"Ollama ({self.config.ollama_model})"
        elif self.config.llm_provider == "gemini":
            return f"Gemini ({self.config.gemini_model})"
        else:
            return f"Groq ({self.config.groq_model})"
    
    def get_collections(self) -> List[str]:
        """Get all available collections"""
        try:
            collections = self.chroma_client.list_collections()
            return [col.name for col in collections]
        except:
            return []
    
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
    
    def list_pdfs_in_collection(self, collection_name: str) -> Dict:
        """List all PDFs in a collection"""
        vectorstore = self.get_vectorstore(collection_name)
        if not vectorstore:
            return {"error": "Collection not found"}
        
        try:
            collection = vectorstore._collection
            doc_count = collection.count()
            result = collection.get(include=["metadatas"], limit=doc_count)
            
            # Extract unique filenames
            pdf_info = {}
            for metadata in result.get("metadatas", []):
                if metadata and metadata.get("filename"):
                    filename = metadata["filename"]
                    if filename not in pdf_info:
                        pdf_info[filename] = {
                            "filename": filename,
                            "chunks": 0,
                            "title": metadata.get("title", "No Title"),
                        }
                    pdf_info[filename]["chunks"] += 1
            
            return {
                "collection": collection_name,
                "total_pdfs": len(pdf_info),
                "total_chunks": doc_count,
                "pdfs": list(pdf_info.values())
            }
        except Exception as e:
            return {"error": str(e)}
    
    def search_collection(
        self, 
        query: str, 
        collection_name: str, 
        k: int = None
    ) -> Tuple[str, List[Dict]]:
        """Search in a single collection"""
        k = k or self.config.top_k
        vectorstore = self.get_vectorstore(collection_name)
        
        if not vectorstore:
            return "", []
        
        try:
            results = vectorstore.similarity_search_with_score(query, k=k)
            return self._format_results(results, collection_name)
        except Exception as e:
            print(f"❌ Search error: {e}")
            return "", []
    
    def search_all_collections(self, query: str) -> Tuple[str, List[Dict]]:
        """Search across all collections (ChatALL mode)"""
        collections = self.get_collections()
        all_results = []
        
        for collection_name in collections:
            try:
                vectorstore = self.get_vectorstore(collection_name)
                if not vectorstore:
                    continue
                
                results = vectorstore.similarity_search_with_score(
                    query, 
                    k=self.config.top_k_chatall
                )
                
                for doc, score in results:
                    all_results.append({
                        "content": doc.page_content,
                        "filename": doc.metadata.get("filename", "unknown"),
                        "collection": collection_name,
                        "similarity": round(1 - score, 4),
                        "page_numbers": doc.metadata.get("page_numbers", "[]"),
                        "title": doc.metadata.get("title", "No Title"),
                    })
            except:
                continue
        
        # Sort by similarity
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Take top results
        top_results = all_results[:15]
        
        # Build context
        context = self._build_context_from_results(top_results)
        
        return context, top_results
    
    def search_specific_file(
        self, 
        query: str, 
        filename: str, 
        collection_name: Optional[str] = None
    ) -> Tuple[str, List[Dict], bool]:
        """Search within a specific PDF file"""
        collections_to_search = [collection_name] if collection_name else self.get_collections()
        
        for coll_name in collections_to_search:
            vectorstore = self.get_vectorstore(coll_name)
            if not vectorstore:
                continue
            
            try:
                results = vectorstore.similarity_search_with_score(
                    query,
                    k=self.config.top_k,
                    filter={"filename": filename}
                )
                
                if results:
                    context, search_results = self._format_results(results, coll_name)
                    return context, search_results, True
            except:
                continue
        
        return "", [], False
    
    def _format_results(self, results, collection_name: str) -> Tuple[str, List[Dict]]:
        """Format search results"""
        context_parts = []
        search_results = []
        
        for doc, score in results:
            filename = doc.metadata.get("filename", "unknown")
            page_numbers = doc.metadata.get("page_numbers", "[]")
            title = doc.metadata.get("title", "No Title")
            similarity = round(1 - score, 4)
            
            source_info = f"Source: {filename}"
            if collection_name:
                source_info += f" (Collection: {collection_name})"
            if page_numbers != "[]":
                source_info += f" - Pages: {page_numbers}"
            
            context_parts.append(f"{doc.page_content}\n{source_info}")
            
            search_results.append({
                "content": doc.page_content,
                "filename": filename,
                "collection": collection_name,
                "similarity": similarity,
                "page_numbers": page_numbers,
                "title": title,
            })
        
        context = "\n\n".join(context_parts)
        return context, search_results
    
    def _build_context_from_results(self, results: List[Dict]) -> str:
        """Build context from results"""
        context_parts = []
        for result in results:
            source_info = f"Source: {result['filename']} (Collection: {result['collection']})"
            if result.get('page_numbers') != "[]":
                source_info += f" - Pages: {result['page_numbers']}"
            context_parts.append(f"{result['content']}\n{source_info}")
        return "\n\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM"""
        system_prompt = f"""You are a knowledgeable document assistant. Answer questions based only on the provided context.

Context from documents:
{context}
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
    
    def query(
        self, 
        query: str, 
        collection_name: Optional[str] = None,
        chatall: bool = False,
        classify: bool = True
    ) -> Dict:
        """
        Main query function - handles classification and routing
        
        Returns:
            Dict with query results including answer, sources, classification
        """
        print(f"\n{'='*60}")
        print(f"📝 Query: {query}")
        print(f"{'='*60}")
        
        # Step 1: Classify query
        if classify:
            classification, filename = self.classifier.classify(query)
            print(f"🔍 Classification: {classification}")
            if filename:
                print(f"📄 Extracted filename: {filename}")
        else:
            classification = "content_search"
            filename = None
        
        # Step 2: Handle based on classification
        if classification == "list_pdfs":
            if chatall:
                # List PDFs from all collections
                collections = self.get_collections()
                all_pdfs = []
                for coll in collections:
                    pdf_list = self.list_pdfs_in_collection(coll)
                    all_pdfs.append(pdf_list)
                return {
                    "classification": classification,
                    "answer": "Here are all available PDFs across collections",
                    "data": all_pdfs,
                    "sources": []
                }
            else:
                # List PDFs from single collection
                if not collection_name:
                    return {"error": "No collection specified"}
                pdf_list = self.list_pdfs_in_collection(collection_name)
                return {
                    "classification": classification,
                    "answer": f"PDFs in collection '{collection_name}'",
                    "data": pdf_list,
                    "sources": []
                }
        
        elif classification == "count_pdfs":
            if chatall:
                collections = self.get_collections()
                total = sum(
                    self.list_pdfs_in_collection(c).get("total_pdfs", 0)
                    for c in collections
                )
                return {
                    "classification": classification,
                    "answer": f"Total PDFs across all collections: {total}",
                    "data": {"total": total},
                    "sources": []
                }
            else:
                if not collection_name:
                    return {"error": "No collection specified"}
                pdf_list = self.list_pdfs_in_collection(collection_name)
                count = pdf_list.get("total_pdfs", 0)
                return {
                    "classification": classification,
                    "answer": f"Total PDFs in '{collection_name}': {count}",
                    "data": pdf_list,
                    "sources": []
                }
        
        elif classification == "file_specific_search":
            print(f"🔎 Searching in file: {filename}")
            context, sources, found = self.search_specific_file(
                query, filename, collection_name
            )
            
            if not found:
                return {
                    "classification": classification,
                    "error": f"File '{filename}' not found",
                    "sources": []
                }
            
            print(f"✅ Found {len(sources)} relevant chunks")
            answer = self.generate_answer(query, context)
            
            return {
                "classification": classification,
                "query": query,
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources)
            }
        
        else:  # content_search
            if chatall:
                print(f"🌐 Searching across all collections")
                context, sources = self.search_all_collections(query)
            else:
                if not collection_name:
                    return {"error": "No collection specified"}
                print(f"🔎 Searching in collection: {collection_name}")
                context, sources = self.search_collection(query, collection_name)
            
            if not sources:
                return {
                    "classification": classification,
                    "error": "No relevant documents found",
                    "sources": []
                }
            
            print(f"✅ Found {len(sources)} relevant chunks")
            answer = self.generate_answer(query, context)
            
            return {
                "classification": classification,
                "query": query,
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources)
            }


# ==================== CLI INTERFACE ====================

def print_result(result: Dict, verbose: bool = False):
    """Pretty print query result"""
    print(f"\n{'='*60}")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    if "answer" in result:
        print(f"💡 Answer:")
        print(f"{result['answer']}")
    
    if verbose and "sources" in result and result["sources"]:
        print(f"\n📚 Sources ({result.get('num_sources', 0)}):")
        for i, source in enumerate(result["sources"][:5], 1):
            print(f"\n{i}. {source['filename']} (Similarity: {source['similarity']:.3f})")
            if source.get('page_numbers') != "[]":
                print(f"   Pages: {source['page_numbers']}")
            print(f"   {source['content'][:200]}...")
    
    print(f"{'='*60}\n")


def interactive_mode(pipeline: RAGPipeline, collection_name: Optional[str], chatall: bool):
    """Interactive query mode"""
    print(f"\n{'='*60}")
    print(f"🤖 Interactive RAG Query Mode")
    print(f"{'='*60}")
    
    if chatall:
        print(f"Mode: ChatALL (searching across all collections)")
        collections = pipeline.get_collections()
        print(f"Available collections: {', '.join(collections)}")
    elif collection_name:
        print(f"Collection: {collection_name}")
    else:
        print("❌ Error: Must specify collection with -c or use --chatall")
        return
    
    print(f"\nCommands:")
    print(f"  - Type your query and press Enter")
    print(f"  - 'list' - List all PDFs")
    print(f"  - 'count' - Count PDFs")
    print(f"  - 'exit' or 'quit' - Exit interactive mode")
    print(f"{'='*60}\n")
    
    while True:
        try:
            query = input("Query> ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'list':
                query = "list all pdfs"
            elif query.lower() == 'count':
                query = "how many pdfs are there"
            
            result = pipeline.query(
                query, 
                collection_name=collection_name,
                chatall=chatall
            )
            
            print_result(result, verbose=True)
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Minimalistic RAG Query Pipeline for Testing and Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query a single collection
  python rag_query.py -q "What is machine learning?" -c my_collection
  
  # Search across all collections (ChatALL mode)
  python rag_query.py -q "Explain neural networks" --chatall
  
  # Search in specific file
  python rag_query.py -q "Summarize introduction" -c docs --file research.pdf
  
  # Interactive mode
  python rag_query.py --interactive -c my_collection
  python rag_query.py -i --chatall
  
  # List available collections
  python rag_query.py --list-collections
  
  # Custom configuration
  python rag_query.py -q "query" -c coll --top-k 20 --temperature 0.3
  python rag_query.py -q "query" -c coll --use-local --ollama-model llama2
  python rag_query.py -q "query" -c coll --embedding-model all-mpnet-base-v2
        """
    )
    
    # Query arguments
    parser.add_argument("-q", "--query", help="Query text")
    parser.add_argument("-c", "--collection", help="Collection name to search")
    parser.add_argument("--chatall", action="store_true", help="Search across all collections")
    parser.add_argument("--file", help="Search in specific PDF file")
    
    # Mode arguments
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive query mode")
    parser.add_argument("--list-collections", action="store_true", help="List all collections")
    
    # LLM Configuration
    parser.add_argument("--use-local", action="store_true", help="Use local Ollama LLM")
    parser.add_argument("--llm-provider", choices=["gemini", "groq"], help="LLM provider")
    parser.add_argument("--ollama-model", help="Ollama model name")
    parser.add_argument("--gemini-model", help="Gemini model name")
    parser.add_argument("--groq-model", help="Groq model name")
    parser.add_argument("--temperature", type=float, help="LLM temperature")
    
    # Embedding Configuration
    parser.add_argument("--embedding-model", help="HuggingFace embedding model")
    
    # Retrieval Configuration
    parser.add_argument("--top-k", type=int, help="Number of chunks to retrieve")
    parser.add_argument("--top-k-chatall", type=int, help="Chunks per collection in ChatALL")
    
    # Database Configuration
    parser.add_argument("--db-path", help="Path to ChromaDB database")
    
    # Output Configuration
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed sources")
    parser.add_argument("--no-classify", action="store_true", help="Skip query classification")
    parser.add_argument("--max-history", type=int, help="Max conversation history")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = RAGConfig(args)
    
    # Initialize pipeline
    try:
        pipeline = RAGPipeline(config)
    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        return
    
    # Handle list-collections
    if args.list_collections:
        collections = pipeline.get_collections()
        print(f"\n📚 Available Collections ({len(collections)}):")
        for i, coll in enumerate(collections, 1):
            info = pipeline.list_pdfs_in_collection(coll)
            print(f"{i}. {coll}")
            print(f"   PDFs: {info.get('total_pdfs', 0)}, Chunks: {info.get('total_chunks', 0)}")
        return
    
    # Handle interactive mode
    if args.interactive:
        interactive_mode(pipeline, args.collection, args.chatall)
        return
    
    # Handle single query
    if args.query:
        result = pipeline.query(
            args.query,
            collection_name=args.collection,
            chatall=args.chatall,
            classify=not args.no_classify
        )
        print_result(result, verbose=args.verbose)
        return
    
    # No action specified
    parser.print_help()


if __name__ == "__main__":
    main()
