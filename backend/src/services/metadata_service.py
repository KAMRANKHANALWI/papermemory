"""
Metadata service for retrieving PDF and collection information
"""

from typing import List, Dict, Tuple
from langchain_chroma import Chroma


class MetadataService:
    """Service for retrieving metadata about PDFs and collections"""

    @staticmethod
    def get_single_collection_pdfs(vectorstore: Chroma) -> Tuple[List[str], Dict]:
        """
        Get list of all PDFs in a single collection with detailed metadata.

        Args:
            vectorstore: Chroma vectorstore instance

        Returns:
            Tuple of (list of filenames, metadata dict with stats)
        """
        try:
            collection = vectorstore._collection
            doc_count = collection.count()

            # Get all metadata
            result = collection.get(include=["metadatas"], limit=doc_count)

            # Extract unique filenames and gather stats
            pdf_info = {}
            for metadata in result.get("metadatas", []):
                if metadata and metadata.get("filename"):
                    filename = metadata["filename"]
                    if filename not in pdf_info:
                        pdf_info[filename] = {
                            "filename": filename,
                            "chunks": 0,
                            "title": metadata.get("title", "No Title"),
                            "pages": set(),
                        }

                    pdf_info[filename]["chunks"] += 1

                    # Collect page numbers
                    pages_str = metadata.get("page_numbers", "[]")
                    pages_list = (
                        pages_str.strip("[]")
                        .replace("'", "")
                        .replace(" ", "")
                        .split(",")
                    )
                    for page in pages_list:
                        if page and page.isdigit():
                            pdf_info[filename]["pages"].add(int(page))

            # Format output
            filenames = list(pdf_info.keys())

            stats = {
                "total_pdfs": len(filenames),
                "total_chunks": doc_count,
                "pdf_details": [],
            }

            for filename, info in pdf_info.items():
                pages = sorted(list(info["pages"]))
                stats["pdf_details"].append(
                    {
                        "filename": filename,
                        "title": info["title"],
                        "chunks": info["chunks"],
                        "pages": len(pages),
                        "page_range": f"{pages[0]}-{pages[-1]}" if pages else "unknown",
                    }
                )

            # Sort by filename
            stats["pdf_details"].sort(key=lambda x: x["filename"])

            return filenames, stats

        except Exception as e:
            print(f"Error getting PDF list: {e}")
            return [], {"total_pdfs": 0, "total_chunks": 0, "pdf_details": []}

    @staticmethod
    def get_chatall_collection_pdfs(
        all_collections: Dict,
    ) -> Tuple[Dict[str, List[str]], Dict]:
        """
        Get list of all PDFs across all collections.

        Args:
            all_collections: Dict mapping collection names to vectorstore instances

        Returns:
            Tuple of (dict mapping collection_name to filenames, aggregate stats)
        """
        try:
            all_pdfs = {}
            aggregate_stats = {
                "total_collections": len(all_collections),
                "total_pdfs_across_all": 0,
                "total_chunks_across_all": 0,
                "collection_details": [],
            }

            for collection_name, vectorstore in all_collections.items():
                filenames, stats = MetadataService.get_single_collection_pdfs(
                    vectorstore
                )

                all_pdfs[collection_name] = filenames

                aggregate_stats["total_pdfs_across_all"] += stats["total_pdfs"]
                aggregate_stats["total_chunks_across_all"] += stats["total_chunks"]

                aggregate_stats["collection_details"].append(
                    {
                        "collection_name": collection_name,
                        "pdf_count": stats["total_pdfs"],
                        "chunk_count": stats["total_chunks"],
                        "pdfs": stats["pdf_details"],
                    }
                )

            return all_pdfs, aggregate_stats

        except Exception as e:
            print(f"Error getting ChatALL PDF list: {e}")
            return {}, {
                "total_collections": 0,
                "total_pdfs_across_all": 0,
                "total_chunks_across_all": 0,
                "collection_details": [],
            }

    @staticmethod
    def format_pdf_list_for_llm(filenames: List[str], stats: Dict) -> str:
        """
        Format PDF list as context for LLM.

        Args:
            filenames: List of PDF filenames
            stats: Statistics dict from get_single_collection_pdfs

        Returns:
            Formatted string for LLM context
        """
        context_parts = [
            f"AVAILABLE DOCUMENTS IN THIS COLLECTION:",
            f"Total PDFs: {stats['total_pdfs']}",
            f"Total document chunks: {stats['total_chunks']}",
            "",
            "PDF LIST:",
        ]

        for i, pdf_detail in enumerate(stats["pdf_details"], 1):
            context_parts.append(
                f"{i}. {pdf_detail['filename']}"
                f"\n   - Title: {pdf_detail['title']}"
                f"\n   - Pages: {pdf_detail['pages']} (range: {pdf_detail['page_range']})"
                f"\n   - Chunks: {pdf_detail['chunks']}"
            )

        return "\n".join(context_parts)

    @staticmethod
    def format_chatall_pdf_list_for_llm(
        all_pdfs: Dict[str, List[str]], stats: Dict
    ) -> str:
        """
        Format ChatALL PDF list as context for LLM.

        Args:
            all_pdfs: Dict mapping collection names to PDF lists
            stats: Aggregate statistics

        Returns:
            Formatted string for LLM context
        """
        context_parts = [
            f"AVAILABLE DOCUMENTS ACROSS ALL COLLECTIONS:",
            f"Total Collections: {stats['total_collections']}",
            f"Total PDFs (all collections): {stats['total_pdfs_across_all']}",
            f"Total document chunks: {stats['total_chunks_across_all']}",
            "",
        ]

        for coll_detail in stats["collection_details"]:
            context_parts.append(f"\nCOLLECTION: {coll_detail['collection_name']}")
            context_parts.append(
                f"   PDFs: {coll_detail['pdf_count']} | Chunks: {coll_detail['chunk_count']}"
            )
            context_parts.append("")

            for i, pdf_detail in enumerate(coll_detail["pdfs"], 1):
                context_parts.append(
                    f"   {i}. {pdf_detail['filename']}"
                    f"\n      - Title: {pdf_detail['title']}"
                    f"\n      - Pages: {pdf_detail['pages']} (range: {pdf_detail['page_range']})"
                    f"\n      - Chunks: {pdf_detail['chunks']}"
                )

            context_parts.append("")

        return "\n".join(context_parts)
