from fastapi import APIRouter, HTTPException
from langchain_chroma import Chroma

from src.models import (
    CollectionPDFsResponse,
    AllCollectionsPDFsResponse,
    CollectionStatsResponse,
    PDFDetail,
)
from src.services.dependencies import chat_service, metadata_service

router = APIRouter(prefix="/api", tags=["metadata"])


@router.get(
    "/collections/{collection_name}/pdfs",
    response_model=CollectionPDFsResponse,
)
async def get_collection_pdfs(collection_name: str):
    """Get list of all PDFs in a collection with metadata"""
    try:
        vectorstore = Chroma(
            client=chat_service.chroma_client,
            collection_name=collection_name,
            embedding_function=chat_service.embedding_model,
        )
        filenames, stats = metadata_service.get_single_collection_pdfs(vectorstore)
        pdfs = [PDFDetail(**pdf_detail) for pdf_detail in stats["pdf_details"]]
        return CollectionPDFsResponse(
            collection_name=collection_name,
            total_pdfs=stats["total_pdfs"],
            total_chunks=stats["total_chunks"],
            pdfs=pdfs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/collections/{collection_name}/stats",
    response_model=CollectionStatsResponse,
)
async def get_collection_stats(collection_name: str):
    """Get detailed statistics for a collection"""
    try:
        vectorstore = Chroma(
            client=chat_service.chroma_client,
            collection_name=collection_name,
            embedding_function=chat_service.embedding_model,
        )
        filenames, stats = metadata_service.get_single_collection_pdfs(vectorstore)
        pdfs = [PDFDetail(**pdf_detail) for pdf_detail in stats["pdf_details"]]
        return CollectionStatsResponse(
            name=collection_name,
            total_pdfs=stats["total_pdfs"],
            total_chunks=stats["total_chunks"],
            pdfs=pdfs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metadata/all-pdfs", response_model=AllCollectionsPDFsResponse)
async def get_all_pdfs():
    """Get list of all PDFs across all collections"""
    try:
        collections = chat_service.chroma_client.list_collections()
        all_vectorstores = {}
        for col in collections:
            all_vectorstores[col.name] = Chroma(
                client=chat_service.chroma_client,
                collection_name=col.name,
                embedding_function=chat_service.embedding_model,
            )

        all_pdfs, stats = metadata_service.get_chatall_collection_pdfs(all_vectorstores)

        collection_responses = []
        for coll_detail in stats["collection_details"]:
            pdfs = [PDFDetail(**pdf) for pdf in coll_detail["pdfs"]]
            collection_responses.append(
                CollectionPDFsResponse(
                    collection_name=coll_detail["collection_name"],
                    total_pdfs=coll_detail["pdf_count"],
                    total_chunks=coll_detail["chunk_count"],
                    pdfs=pdfs,
                )
            )

        return AllCollectionsPDFsResponse(
            total_collections=stats["total_collections"],
            total_pdfs=stats["total_pdfs_across_all"],
            total_chunks=stats["total_chunks_across_all"],
            collections=collection_responses,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))