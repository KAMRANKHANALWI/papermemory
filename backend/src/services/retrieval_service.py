from typing import Optional, List, Dict

from src.config import AppConfig


class RetrievalService:

    @staticmethod
    def retrieve_content(
        message: str,
        is_chatall: bool,
        collection_name: Optional[str],
        chroma_client,
        get_vectorstore,
        reranker,
        logger,
    ) -> tuple[str, List[Dict]]:

        all_chunks = []

        if is_chatall:

            for col in chroma_client.list_collections():

                try:
                    vectorstore = get_vectorstore(col.name)

                    results = vectorstore.similarity_search_with_score(
                        message,
                        k=AppConfig.RERANKING_SAMPLE_SIZE,
                    )

                    for doc, score in results:

                        all_chunks.append(
                            {
                                "content": doc.page_content,
                                "filename": doc.metadata.get(
                                    "filename",
                                    "unknown",
                                ),
                                "page_numbers": doc.metadata.get(
                                    "page_numbers",
                                    "[]",
                                ),
                                "title": doc.metadata.get(
                                    "title",
                                    "No Title",
                                ),
                                "similarity": round(
                                    1 - float(score),
                                    4,
                                ),
                                "collection": col.name,
                            }
                        )

                except Exception as e:
                    logger.warning(
                        f"Search failed for {col.name}: {e}"
                    )

        else:

            if not collection_name:
                raise ValueError(
                    "Collection name required"
                )

            vectorstore = get_vectorstore(
                collection_name
            )

            results = vectorstore.similarity_search_with_score(
                message,
                k=AppConfig.RERANKING_SAMPLE_SIZE,
            )

            for doc, score in results:

                all_chunks.append(
                    {
                        "content": doc.page_content,
                        "filename": doc.metadata.get(
                            "filename",
                            "unknown",
                        ),
                        "page_numbers": doc.metadata.get(
                            "page_numbers",
                            "[]",
                        ),
                        "title": doc.metadata.get(
                            "title",
                            "No Title",
                        ),
                        "similarity": round(
                            1 - float(score),
                            4,
                        ),
                        "collection": collection_name,
                    }
                )

        if not all_chunks:
            return "", []

        top_chunks = reranker.rerank(
            message,
            all_chunks,
            top_k=AppConfig.TOP_K,
        )

        context_parts = []

        for chunk in top_chunks:

            header = (
                f"[Source: {chunk['filename']} | "
                f"Collection: {chunk['collection']} | "
                f"Pages {chunk['page_numbers']}]"
            )

            context_parts.append(
                f"{header}\n{chunk['content']}"
            )

        context = "\n\n---\n\n".join(
            context_parts
        )

        sources = [
            {
                "content": c["content"],
                "filename": c["filename"],
                "collection": c["collection"],
                "page_numbers": c["page_numbers"],
                "similarity": c.get(
                    "rerank_score",
                    c["similarity"],
                ),
                "rerank_score": c.get(
                    "rerank_score",
                    c["similarity"],
                ),
                "title": c.get(
                    "title",
                    "No Title",
                ),
            }
            for c in top_chunks
        ]

        return context, sources
    
    
    @staticmethod
    def retrieve_file_content(
        message: str,
        filename: str,
        collection_name,
        is_chatall: bool,
        chroma_client,
        get_vectorstore,
        file_search_service,
        reranker,
        logger,
    ):
        if is_chatall:

            vectorstores = {
                col.name: get_vectorstore(col.name)
                for col in chroma_client.list_collections()
            }

            _, results, found, _ = (
                file_search_service.search_specific_file_chatall(
                    vectorstores,
                    filename,
                    message,
                    num_results=AppConfig.RERANKING_SAMPLE_SIZE,
                )
            )

            all_chunks = results

        else:

            if not collection_name:
                raise ValueError(
                    "Collection name required"
                )

            vectorstore = get_vectorstore(
                collection_name
            )

            raw_results = (
                vectorstore.similarity_search_with_score(
                    message,
                    k=AppConfig.RERANKING_SAMPLE_SIZE,
                    filter={"filename": filename},
                )
            )

            all_chunks = [
                {
                    "content": doc.page_content,
                    "filename": doc.metadata.get(
                        "filename",
                        "unknown",
                    ),
                    "page_numbers": doc.metadata.get(
                        "page_numbers",
                        "[]",
                    ),
                    "title": doc.metadata.get(
                        "title",
                        "No Title",
                    ),
                    "similarity": round(
                        1 - float(score),
                        4,
                    ),
                    "collection": collection_name,
                }
                for doc, score in raw_results
            ]

        if not all_chunks:
            return "", [], False

        top_chunks = reranker.rerank(
            message,
            all_chunks,
            top_k=AppConfig.TOP_K,
        )

        context_parts = []

        for chunk in top_chunks:

            header = (
                f"[Source: {chunk['filename']} "
                f"| Pages {chunk['page_numbers']}]"
            )

            context_parts.append(
                f"{header}\n{chunk['content']}"
            )

        context = "\n\n---\n\n".join(
            context_parts
        )

        sources = [
            {
                "content": c["content"],
                "filename": c["filename"],
                "collection": c.get(
                    "collection",
                    collection_name,
                ),
                "page_numbers": c["page_numbers"],
                "similarity": c.get(
                    "rerank_score",
                    c["similarity"],
                ),
                "rerank_score": c.get(
                    "rerank_score",
                    c["similarity"],
                ),
                "title": c.get(
                    "title",
                    "No Title",
                ),
            }
            for c in top_chunks
        ]

        return context, sources, True