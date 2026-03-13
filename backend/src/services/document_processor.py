"""
Document processor with rich metadata extraction + pages_store saving for parent-page retrieval.
"""

import os
import json
import tempfile
import chromadb
from pathlib import Path
from typing import List
from fastapi import UploadFile
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.utils.pdf_processor import process_pdf_with_pymupdf, chunk_text_content
from src.config import AppConfig


class DocumentProcessor:
    def __init__(self, pdf_storage=None):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=AppConfig.EMBEDDING_MODEL
        )
        self.chroma_client = chromadb.PersistentClient(path=AppConfig.CHROMA_DB_PATH)
        self.pdf_storage = pdf_storage
        self.pages_store_base = Path(AppConfig.PAGES_STORE_PATH)
        self.pages_store_base.mkdir(parents=True, exist_ok=True)

    async def process_files(self, files: List[UploadFile], collection_name: str):
        """Process uploaded PDF files with rich metadata + save pages_store."""
        all_documents = []
        processed_files = []

        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                continue

            content = await file.read()

            # Save original PDF to disk
            if self.pdf_storage:
                try:
                    self.pdf_storage.save_pdf(content, collection_name, file.filename)
                    print(f"Saved PDF to storage: {file.filename}")
                except Exception as e:
                    print(f"Failed to save PDF {file.filename}: {e}")

            # Save temporarily for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name

            try:
                # Process PDF into chunks (for ChromaDB)
                documents, pages_data = self._process_single_pdf(
                    tmp_path, file.filename
                )
                all_documents.extend(documents)
                processed_files.append(file.filename)

                # Save pages_store for parent-page retrieval
                self._save_pages_store(collection_name, file.filename, pages_data)

            finally:
                os.unlink(tmp_path)

        # Create / update vector store
        if all_documents:
            Chroma.from_documents(
                documents=all_documents,
                embedding=self.embedding_model,
                client=self.chroma_client,
                collection_name=collection_name,
                persist_directory=AppConfig.CHROMA_DB_PATH,
            )

        return {
            "files_processed": len(processed_files),
            "chunks_created": len(all_documents),
            "collection": collection_name,
            "processed_files": processed_files,
        }

    # ------------------------------------------------------------------ #
    #  Pages store                                                         #
    # ------------------------------------------------------------------ #

    def _save_pages_store(
        self, collection_name: str, filename: str, pages_data: List[dict]
    ):
        """
        Persist full per-page text to disk so parent-page retrieval can
        fetch complete pages without reconstruction.

        Layout:
            data/pages_store/{collection_name}/{filename}.json
            {
                "1": "full text of page 1 ...",
                "2": "full text of page 2 ...",
                ...
            }
        """
        collection_dir = self.pages_store_base / collection_name
        collection_dir.mkdir(parents=True, exist_ok=True)

        # Use filename stem as JSON key (strip .pdf)
        json_name = Path(filename).stem + ".json"
        store_path = collection_dir / json_name

        pages_dict = {
            str(page["page_num"]): page["text"] for page in pages_data
        }

        with open(store_path, "w", encoding="utf-8") as f:
            json.dump(pages_dict, f, ensure_ascii=False, indent=2)

        print(f"Saved pages_store: {store_path} ({len(pages_dict)} pages)")

    def get_pages_store(self, collection_name: str, filename: str) -> dict:
        """
        Load the pages_store for a specific PDF.

        Returns:
            Dict mapping page_num (str) -> full page text.
            Empty dict if not found.
        """
        json_name = Path(filename).stem + ".json"
        store_path = self.pages_store_base / collection_name / json_name

        if not store_path.exists():
            print(f"pages_store not found: {store_path}")
            return {}

        with open(store_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------ #
    #  PDF processing                                                      #
    # ------------------------------------------------------------------ #

    def _process_single_pdf(
        self, file_path: str, filename: str
    ):
        """
        Process a single PDF into:
          - List[Document] for ChromaDB (chunks with metadata)
          - List[dict] raw pages_data (returned for pages_store saving)
        """
        pages_data = process_pdf_with_pymupdf(file_path)
        if not pages_data:
            return [], []

        full_text = "\n\n".join(
            [page["text"] for page in pages_data if page["text"]]
        )

        # Overall headings for title metadata
        all_headings = []
        for page in pages_data:
            all_headings.extend(page["headings"])
        unique_headings = list(dict.fromkeys(all_headings))[:5]

        text_chunks = chunk_text_content(full_text)

        documents = []
        for chunk_id, chunk_text in enumerate(text_chunks):
            chunk_pages = self._find_chunk_pages(chunk_text, full_text, pages_data)

            doc = Document(
                page_content=chunk_text,
                metadata={
                    "filename": filename,
                    "filepath": file_path,
                    "chunk_id": chunk_id,
                    "total_chunks": len(text_chunks),
                    # Stored as string for ChromaDB metadata compatibility
                    "page_numbers": str(chunk_pages) if chunk_pages else "[]",
                    "title": unique_headings[0] if unique_headings else "No Title",
                    "all_headings": str(unique_headings),
                    "text_length": len(chunk_text),
                    "source_type": "pdf",
                },
            )
            documents.append(doc)

        return documents, pages_data

    def _find_chunk_pages(
        self, chunk_text: str, full_text: str, pages_data: List[dict]
    ) -> List[int]:
        """
        Determine which pages a chunk belongs to (1-indexed).
        Returns list of page numbers, defaulting to [1] if not locatable.
        """
        chunk_pages = []
        chunk_pos = full_text.find(chunk_text)

        if chunk_pos != -1:
            char_count = 0
            for page in pages_data:
                page_end = char_count + len(page["text"])

                if char_count <= chunk_pos < page_end:
                    chunk_pages.append(page["page_num"])

                    chunk_end = chunk_pos + len(chunk_text)
                    if chunk_end > page_end:
                        remaining_pages = [
                            p["page_num"]
                            for p in pages_data
                            if p["page_num"] > page["page_num"]
                        ]
                        for next_page_num in remaining_pages:
                            chunk_pages.append(next_page_num)
                            if len(chunk_pages) >= 3:
                                break
                    break

                char_count = page_end

        return chunk_pages if chunk_pages else [1]