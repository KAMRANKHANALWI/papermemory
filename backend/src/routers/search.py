from fastapi import APIRouter, HTTPException
from langchain_chroma import Chroma

from src.models import FileSearchRequest, FileSearchResponse
from src.services.dependencies import chat_service, file_search_service

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("/file", response_model=FileSearchResponse)
async def search_specific_file(request: FileSearchRequest):
    """Search within a specific PDF file"""
    try:
        if not request.collection_name:
            raise HTTPException(
                status_code=400, detail="collection_name required for file search"
            )
        vectorstore = Chroma(
            client=chat_service.chroma_client,
            collection_name=request.collection_name,
            embedding_function=chat_service.embedding_model,
        )
        context, results, found = file_search_service.search_specific_file(
            vectorstore, request.filename, request.query, request.num_results
        )
        return FileSearchResponse(
            filename=request.filename,
            collection_name=request.collection_name,
            found=found,
            num_results=len(results),
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/file-all", response_model=FileSearchResponse)
async def search_file_all_collections(request: FileSearchRequest):
    """Search for a specific file across all collections"""
    try:
        collections = chat_service.chroma_client.list_collections()
        all_vectorstores = {}
        for col in collections:
            all_vectorstores[col.name] = Chroma(
                client=chat_service.chroma_client,
                collection_name=col.name,
                embedding_function=chat_service.embedding_model,
            )

        context, results, found, collection_name = (
            file_search_service.search_specific_file_chatall(
                all_vectorstores, request.filename, request.query, request.num_results
            )
        )
        return FileSearchResponse(
            filename=request.filename,
            collection_name=collection_name,
            found=found,
            num_results=len(results),
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))