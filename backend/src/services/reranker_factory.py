"""
reranker_factory.py
Returns the correct reranker based on RERANKER_TYPE in .env / AppConfig.

Supported types:
    "none"          → NoReranker (pass-through, keeps top-K by similarity score)
    "cross_encoder" → CrossEncoderReranker (local, no API cost)
    "llm_local"     → LLMReranker using Ollama
    "llm_api"       → LLMReranker using Gemini or Groq API
"""

from __future__ import annotations
import logging
from typing import List, Dict, Tuple
from src.config import AppConfig

logger = logging.getLogger(__name__)


# ============================================================
# Base interface
# ============================================================

class BaseReranker:
    """
    All rerankers take a list of chunk dicts and a query,
    and return the same list sorted by relevance (descending).

    Each chunk dict has at minimum:
        {
            "content": str,
            "filename": str,
            "similarity": float,   ← original vector similarity score
            "page_numbers": str,
            ...
        }
    After reranking a "rerank_score" key is added.
    """

    def rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        raise NotImplementedError


# ============================================================
# 1. NoReranker — just return top-K by similarity score
# ============================================================

class NoReranker(BaseReranker):
    def rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        sorted_chunks = sorted(chunks, key=lambda x: x["similarity"], reverse=True)
        for chunk in sorted_chunks:
            chunk["rerank_score"] = chunk["similarity"]
        return sorted_chunks[:top_k]


# ============================================================
# 2. CrossEncoderReranker — local, no API cost
# ============================================================

class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = AppConfig.CROSS_ENCODER_MODEL):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            logger.info(f"CrossEncoder loaded: {model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    def rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        if not chunks:
            return []

        pairs = [(query, chunk["content"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_chunks[:top_k]


# ============================================================
# 3. LLMReranker — works with Ollama (local) or API (Gemini/Groq)
# ============================================================

RERANK_SYSTEM_PROMPT = """You are a relevance scoring assistant.
Given a query and a text passage, score how relevant the passage is to the query.

Scoring scale (0.0 to 1.0 in steps of 0.1):
0.0 = Completely irrelevant
0.1 = Virtually irrelevant
0.2 = Very slight connection
0.3 = Slightly relevant, minimal detail
0.4 = Somewhat relevant, partial info
0.5 = Moderately relevant, limited scope
0.6 = Fairly relevant, lacks depth
0.7 = Relevant, substantive but incomplete
0.8 = Very relevant, strong match
0.9 = Highly relevant, nearly complete answer
1.0 = Perfectly relevant, direct complete answer

Respond with ONLY a JSON object: {"score": <float>}
No explanation, no markdown, just the JSON."""


class LLMReranker(BaseReranker):
    """
    Uses an LLM to score chunk relevance.
    Supports: Ollama (local), Gemini API, Groq API.
    """

    def __init__(self, provider: str = "local"):
        """
        Args:
            provider: "local" (Ollama) | "gemini" | "groq"
        """
        self.provider = provider
        self.llm = self._build_llm(provider)

    def _build_llm(self, provider: str):
        if provider == "local":
            from langchain_ollama import ChatOllama
            logger.info(f"LLMReranker using Ollama: {AppConfig.OLLAMA_MODEL}")
            return ChatOllama(
                model=AppConfig.OLLAMA_MODEL,
                base_url=AppConfig.OLLAMA_BASE_URL,
                temperature=0,
            )

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            logger.info(f"LLMReranker using Gemini: {AppConfig.GEMINI_MODEL}")
            return ChatGoogleGenerativeAI(
                model=AppConfig.GEMINI_MODEL,
                temperature=0,
                google_api_key=AppConfig.GOOGLE_API_KEY,
            )

        if provider == "groq":
            from langchain_groq import ChatGroq
            logger.info(f"LLMReranker using Groq: {AppConfig.GROQ_MODEL}")
            return ChatGroq(
                model=AppConfig.GROQ_MODEL,
                temperature=0,
                api_key=AppConfig.GROQ_API_KEY,
            )

        raise ValueError(f"Unknown LLM reranker provider: {provider}")

    def _score_chunk(self, query: str, content: str) -> float:
        """Score a single chunk. Returns float 0-1."""
        import json as _json

        user_msg = f'Query: "{query}"\n\nPassage:\n"""\n{content[:1500]}\n"""'
        messages = [
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        try:
            response = self.llm.invoke(messages)
            raw = response.content.strip()

            # Strip markdown fences if model wraps JSON
            if raw.startswith("```"):
                raw = raw.strip("`").strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()

            data = _json.loads(raw)
            score = float(data.get("score", 0.5))
            return max(0.0, min(1.0, score))  # clamp to [0, 1]

        except Exception as e:
            logger.warning(f"LLMReranker scoring failed: {e}, defaulting to 0.5")
            return 0.5

    def rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        if not chunks:
            return []

        # Score in parallel using threads
        from concurrent.futures import ThreadPoolExecutor, as_completed

        vector_weight = 0.3
        llm_weight = 0.7

        def score_one(chunk):
            llm_score = self._score_chunk(query, chunk["content"])
            combined = llm_weight * llm_score + vector_weight * chunk["similarity"]
            return chunk, llm_score, round(combined, 4)

        results = []
        # Cap parallel workers to avoid hammering local Ollama
        max_workers = 3 if self.provider == "local" else 8

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(score_one, chunk): chunk for chunk in chunks}
            for future in as_completed(futures):
                try:
                    chunk, llm_score, combined = future.result()
                    chunk["rerank_score"] = llm_score
                    chunk["combined_score"] = combined
                    results.append(chunk)
                except Exception as e:
                    logger.warning(f"Scoring future failed: {e}")
                    chunk = futures[future]
                    chunk["rerank_score"] = 0.5
                    chunk["combined_score"] = chunk["similarity"]
                    results.append(chunk)

        sorted_results = sorted(
            results, key=lambda x: x.get("combined_score", 0), reverse=True
        )
        return sorted_results[:top_k]


# ============================================================
# Factory function — single import point
# ============================================================

def get_reranker(reranker_type: str = None) -> BaseReranker:
    """
    Returns the appropriate reranker based on config.

    Args:
        reranker_type: Override AppConfig.RERANKER_TYPE if provided.

    Returns:
        BaseReranker instance ready to use.
    """
    rtype = (reranker_type or AppConfig.RERANKER_TYPE).lower()

    if rtype == "none":
        logger.info("Reranker: none (top-K by similarity)")
        return NoReranker()

    if rtype == "cross_encoder":
        logger.info(f"Reranker: CrossEncoder ({AppConfig.CROSS_ENCODER_MODEL})")
        return CrossEncoderReranker()

    if rtype == "llm_local":
        logger.info("Reranker: LLM local (Ollama)")
        return LLMReranker(provider="local")

    if rtype == "llm_api":
        provider = AppConfig.RERANKER_API_PROVIDER.lower()
        logger.info(f"Reranker: LLM API ({provider})")
        return LLMReranker(provider=provider)

    logger.warning(f"Unknown RERANKER_TYPE '{rtype}', falling back to NoReranker")
    return NoReranker()