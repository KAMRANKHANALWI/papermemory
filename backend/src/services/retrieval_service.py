from typing import List, Dict
from src.config import AppConfig


class RetrievalService:

    @staticmethod
    def retrieve_content(
        message,
        is_chatall,
        collection_name,
        chroma_client,
        get_vectorstore,
        logger,
    ):
        all_results = []

        if is_chatall:

            for col in chroma_client.list_collections():

                try:
                    vectorstore = get_vectorstore(col.name)

                    results = vectorstore.similarity_search_with_score(
                        message,
                        k=AppConfig.TOP_K_CHATALL,
                    )

                    for doc, score in results:
                        all_results.append(
                            {
                                "content": doc.page_content,
                                "filename": doc.metadata.get(
                                    "filename",
                                    "unknown",
                                ),
                                "title": doc.metadata.get(
                                    "title",
                                    "No Title",
                                ),
                                "page_numbers": doc.metadata.get(
                                    "page_numbers",
                                    "[]",
                                ),
                                "similarity": round(
                                    1 - score,
                                    4,
                                ),
                                "collection": col.name,
                            }
                        )

                except Exception as e:
                    logger.warning(
                        f"Search failed for {col.name}: {e}"
                    )

            all_results.sort(
                key=lambda x: x["similarity"],
                reverse=True,
            )

            all_results = all_results[: AppConfig.TOP_K]

        else:

            if not collection_name:
                raise ValueError(
                    "Collection name required"
                )

            vectorstore = get_vectorstore(
                collection_name
            )

            results = (
                vectorstore.similarity_search_with_score(
                    message,
                    k=AppConfig.TOP_K,
                )
            )

            for doc, score in results:
                all_results.append(
                    {
                        "content": doc.page_content,
                        "filename": doc.metadata.get(
                            "filename",
                            "unknown",
                        ),
                        "title": doc.metadata.get(
                            "title",
                            "No Title",
                        ),
                        "page_numbers": doc.metadata.get(
                            "page_numbers",
                            "[]",
                        ),
                        "similarity": round(
                            1 - score,
                            4,
                        ),
                        "collection": collection_name,
                    }
                )

        context_parts = []

        for r in all_results:

            src = (
                f"Source: {r['filename']} "
                f"(Collection: {r['collection']})"
            )

            pages = r.get(
                "page_numbers",
                "[]",
            )

            if pages != "[]":

                page_list = (
                    pages.strip("[]")
                    .replace("'", "")
                    .split(",")
                )

                if page_list and page_list[0]:
                    src += (
                        f" - p. "
                        f"{', '.join(page_list)}"
                    )

            context_parts.append(
                f"{r['content']}\n\n{src}"
            )

        context = "\n\n".join(
            context_parts
        )

        return context, all_results