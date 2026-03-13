from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from src.llm_factory import LLMFactory
import chromadb
import os
from dotenv import load_dotenv
from src.config import AppConfig

load_dotenv()


class ChatService:
    def __init__(self):
        self.llm = LLMFactory.create()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=AppConfig.EMBEDDING_MODEL
        )
        self.chroma_client = chromadb.PersistentClient(
            path=AppConfig.CHROMA_DB_PATH
        )

    def search_single_collection(self, query: str, collection_name: str, k: int = None):
        """Search in a single collection"""
        k = k or AppConfig.TOP_K
        try:
            vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=collection_name,
                embedding_function=self.embedding_model,
                persist_directory=AppConfig.CHROMA_DB_PATH,
            )
            results = vectorstore.similarity_search_with_score(query, k=k)
            return self._format_search_results(results, collection_name)
        except Exception as e:
            return {"error": str(e)}, []

    def search_all_collections(self, query: str, k_per_collection: int = None):
        """Search across all collections"""
        k_per_collection = k_per_collection or AppConfig.TOP_K_CHATALL
        all_collections = self._get_available_collections()
        all_results = []

        for collection_name in all_collections:
            try:
                vectorstore = Chroma(
                    client=self.chroma_client,
                    collection_name=collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=AppConfig.CHROMA_DB_PATH,
                )
                results = vectorstore.similarity_search_with_score(
                    query, k=k_per_collection
                )
                formatted_results = self._format_search_results(
                    results, collection_name
                )
                all_results.extend(formatted_results[1])
            except Exception:
                continue

        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        context = self._build_context_from_results(all_results[:1])
        return context, all_results[:15]

    def _format_search_results(self, results, collection_name):
        """Format search results"""
        context_parts = []
        search_results = []

        for doc, score in results:
            filename = doc.metadata.get("filename", "unknown")
            similarity = round(1 - score, 4)

            source_info = f"Source: {filename}"
            if collection_name:
                source_info += f" (Collection: {collection_name})"

            context_parts.append(f"{doc.page_content}\n{source_info}")
            search_results.append(
                {
                    "content": doc.page_content,
                    "filename": filename,
                    "collection": collection_name,
                    "similarity": similarity,
                }
            )

        context = "\n\n".join(context_parts)
        return context, search_results

    def _build_context_from_results(self, results):
        """Build context from multiple collection results"""
        context_parts = []
        for result in results:
            source_info = (
                f"Source: {result['filename']} (Collection: {result['collection']})"
            )
            context_parts.append(f"{result['content']}\n{source_info}")
        return "\n\n".join(context_parts)

    def _get_available_collections(self):
        """Get list of available collection names"""
        try:
            collections = self.chroma_client.list_collections()
            return [col.name for col in collections]
        except Exception:
            return []

    async def generate_response(self, query: str, context: str):
        """Generate AI response using LLM"""
        system_prompt = f"""You are a knowledgeable document assistant. Answer questions based only on the provided context.

Context from documents:
{context}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response_stream = self.llm.astream(messages)
        async for chunk in response_stream:
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

    def get_model_info(self) -> Dict[str, Any]:
        if AppConfig.USE_LOCAL_LLM:
            return {
                "provider": "ollama",
                "model": AppConfig.OLLAMA_MODEL,
                "base_url": AppConfig.OLLAMA_BASE_URL,
                "is_local": True,
            }

        if AppConfig.DEFAULT_MODEL_PROVIDER == "gemini" and AppConfig.GOOGLE_API_KEY:
            return {
                "provider": "gemini",
                "model": AppConfig.GEMINI_MODEL,
                "is_local": False,
            }

        return {
            "provider": "groq",
            "model": AppConfig.GROQ_MODEL,
            "is_local": False,
        }