"""
parent_page_retriever.py

Implements the parent-page retrieval pattern:
    1. similarity_search(k=RERANKING_SAMPLE_SIZE)   → top-N chunks (finding)
    2. rerank chunks                                 → score relevance
    3. take top TOP_K chunks
    4. read page_numbers from each chunk's metadata
    5. deduplicate page numbers
    6. lookup full page text from pages_store JSON
    7. return assembled context string + source metadata

This gives the LLM full page context instead of narrow chunk snippets,
while still using chunks for precise semantic retrieval.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from src.config import AppConfig
from src.services.reranker_factory import get_reranker, BaseReranker

logger = logging.getLogger(__name__)


class ParentPageRetriever:
    """
    Drop-in retriever that replaces direct chunk-based context building.

    Usage:
        retriever = ParentPageRetriever()
        context, sources = retriever.retrieve(
            query="What causes gut dysbiosis?",
            vectorstore=vectorstore,
            collection_name="gut_microbiome",
        )
    """

    def __init__(self, reranker: Optional[BaseReranker] = None):
        self.reranker = reranker or get_reranker()
        self.pages_store_base = Path(AppConfig.PAGES_STORE_PATH)
        logger.info(
            f"ParentPageRetriever init | reranker={type(self.reranker).__name__} | "
            f"sample_size={AppConfig.RERANKING_SAMPLE_SIZE} | top_k={AppConfig.TOP_K}"
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        vectorstore: Chroma,
        collection_name: str,
        filename_filter: Optional[str] = None,
        top_k: Optional[int] = None,
        sample_size: Optional[int] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Full parent-page retrieval pipeline.

        Args:
            query:           User query string.
            vectorstore:     ChromaDB vectorstore for the collection.
            collection_name: Used to locate the pages_store JSON files.
            filename_filter: If set, restrict search to a single PDF filename.
            top_k:           Override AppConfig.TOP_K.
            sample_size:     Override AppConfig.RERANKING_SAMPLE_SIZE.

        Returns:
            (context_str, sources_list)

            context_str  → concatenated full-page texts ready for LLM prompt
            sources_list → list of dicts with filename, page_num, similarity,
                           rerank_score, title, collection for frontend display
        """
        _top_k = top_k or AppConfig.TOP_K
        _sample = sample_size or AppConfig.RERANKING_SAMPLE_SIZE

        # Step 1: Vector search → top-N chunks
        chunks = self._vector_search(
            query, vectorstore, k=_sample, filename_filter=filename_filter
        )

        if not chunks:
            logger.warning("ParentPageRetriever: no chunks returned from vector search")
            return "", []

        # Step 2: Rerank chunks
        top_chunks = self.reranker.rerank(query, chunks, top_k=_top_k)

        # Step 3: Extract (filename, page_num) pairs from chunk metadata
        page_refs = self._extract_page_refs(top_chunks)

        # Step 4: Fetch full page texts from pages_store
        pages = self._fetch_pages(page_refs, collection_name)

        if not pages:
            # Graceful fallback: use raw chunk content if pages_store missing
            logger.warning(
                "pages_store lookup returned nothing — "
                "falling back to raw chunk content. Re-ingest PDFs to fix this."
            )
            return self._fallback_context(top_chunks), self._chunks_to_sources(
                top_chunks, collection_name
            )

        # Step 5: Build context string and source metadata
        context = self._build_context(pages)
        sources = self._pages_to_sources(pages, collection_name)

        return context, sources

    # ------------------------------------------------------------------ #
    #  Step 1 — Vector search                                              #
    # ------------------------------------------------------------------ #

    def _vector_search(
        self,
        query: str,
        vectorstore: Chroma,
        k: int,
        filename_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Returns list of chunk dicts with content + metadata + similarity."""
        try:
            search_kwargs = {"k": k}
            if filename_filter:
                search_kwargs["filter"] = {"filename": filename_filter}

            results = vectorstore.similarity_search_with_score(query, **search_kwargs)

            chunks = []
            for doc, score in results:
                chunks.append(
                    {
                        "content": doc.page_content,
                        "filename": doc.metadata.get("filename", "unknown"),
                        "page_numbers": doc.metadata.get("page_numbers", "[1]"),
                        "chunk_id": doc.metadata.get("chunk_id", 0),
                        "title": doc.metadata.get("title", "No Title"),
                        # ChromaDB returns L2 distance; convert to similarity
                        "similarity": round(1 - float(score), 4),
                    }
                )

            logger.debug(f"Vector search returned {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Step 3 — Extract page references from top chunks                   #
    # ------------------------------------------------------------------ #

    def _extract_page_refs(self, chunks: List[Dict]) -> List[Tuple[str, int]]:
        """
        Parse page_numbers metadata string → unique (filename, page_num) pairs.
        Preserves order of first appearance (most relevant chunk first).

        page_numbers is stored as a string like "[1, 2]" or "[3]".
        """
        seen = set()
        refs = []

        for chunk in chunks:
            filename = chunk["filename"]
            page_numbers_str = chunk.get("page_numbers", "[1]")

            try:
                page_list = ast.literal_eval(page_numbers_str)
                if not isinstance(page_list, list):
                    page_list = [page_list]
            except Exception:
                page_list = [1]

            for page_num in page_list:
                try:
                    page_num = int(page_num)
                except Exception:
                    continue

                key = (filename, page_num)
                if key not in seen:
                    seen.add(key)
                    refs.append(key)

        logger.debug(f"Extracted {len(refs)} unique page refs from {len(chunks)} chunks")
        return refs

    # ------------------------------------------------------------------ #
    #  Step 4 — Fetch full page texts from pages_store                    #
    # ------------------------------------------------------------------ #

    def _fetch_pages(
        self,
        page_refs: List[Tuple[str, int]],
        collection_name: str,
    ) -> List[Dict]:
        """
        Load full page text from pages_store JSON files.

        Returns list of dicts:
            {filename, page_num, text, found}
        """
        # Cache loaded page stores to avoid re-reading the same JSON
        store_cache: Dict[str, dict] = {}
        pages = []

        for filename, page_num in page_refs:
            store_key = f"{collection_name}/{filename}"

            if store_key not in store_cache:
                store_cache[store_key] = self._load_pages_store(
                    collection_name, filename
                )

            page_store = store_cache[store_key]
            page_text = page_store.get(str(page_num), "")

            pages.append(
                {
                    "filename": filename,
                    "page_num": page_num,
                    "text": page_text,
                    "found": bool(page_text),
                }
            )

        found_count = sum(1 for p in pages if p["found"])
        logger.debug(f"Fetched {found_count}/{len(pages)} pages from pages_store")
        return pages

    def _load_pages_store(self, collection_name: str, filename: str) -> dict:
        """Load a single pages_store JSON. Returns empty dict on miss."""
        json_name = Path(filename).stem + ".json"
        store_path = self.pages_store_base / collection_name / json_name

        if not store_path.exists():
            logger.warning(f"pages_store not found: {store_path}")
            return {}

        try:
            with open(store_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load pages_store {store_path}: {e}")
            return {}

    # ------------------------------------------------------------------ #
    #  Step 5 — Build context and sources                                  #
    # ------------------------------------------------------------------ #

    def _build_context(self, pages: List[Dict]) -> str:
        """
        Concatenate full page texts into one context string for the LLM.
        Only includes pages where text was actually found.
        """
        parts = []
        for page in pages:
            if not page["text"]:
                continue
            header = (
                f"[Source: {page['filename']} | Page {page['page_num']}]"
            )
            parts.append(f"{header}\n{page['text']}")

        return "\n\n---\n\n".join(parts)

    def _pages_to_sources(
        self, pages: List[Dict], collection_name: str
    ) -> List[Dict]:
        """Convert pages list to sources format expected by frontend."""
        return [
            {
                "content": page["text"][:500] + "..." if len(page["text"]) > 500 else page["text"],
                "filename": page["filename"],
                "collection": collection_name,
                "page_numbers": str([page["page_num"]]),
                "similarity": 1.0,   # page-level, original score already used for ranking
                "title": "No Title",
            }
            for page in pages
            if page["text"]
        ]

    # ------------------------------------------------------------------ #
    #  Fallback (no pages_store)                                           #
    # ------------------------------------------------------------------ #

    def _fallback_context(self, chunks: List[Dict]) -> str:
        """Use raw chunk content when pages_store is unavailable."""
        parts = []
        for chunk in chunks:
            header = (
                f"[Source: {chunk['filename']} | Pages {chunk['page_numbers']}]"
            )
            parts.append(f"{header}\n{chunk['content']}")
        return "\n\n---\n\n".join(parts)

    def _chunks_to_sources(
        self, chunks: List[Dict], collection_name: str
    ) -> List[Dict]:
        return [
            {
                "content": chunk["content"],
                "filename": chunk["filename"],
                "collection": collection_name,
                "page_numbers": chunk["page_numbers"],
                "similarity": chunk["similarity"],
                "title": chunk.get("title", "No Title"),
            }
            for chunk in chunks
        ]