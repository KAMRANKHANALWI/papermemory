from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List
from urllib.parse import quote

from src.models import (
    RenameCollectionRequest,
    RenamePDFRequest,
    OperationResponse,
)
from src.services.dependencies import (
    collection_manager,
    document_processor,
    pdf_storage,
)

router = APIRouter(prefix="/api/collections", tags=["collections"])


@router.get("")
async def get_collections():
    """Get all available collections"""
    return collection_manager.get_all_collections()


@router.post("/{collection_name}/upload")
async def upload_documents(collection_name: str, files: List[UploadFile] = File(...)):
    """Upload PDFs to a collection"""
    try:
        result = await document_processor.process_files(files, collection_name)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete an entire collection"""
    result = collection_manager.delete_collection(collection_name)
    pdf_storage.delete_collection_pdfs(collection_name)
    document_processor.delete_collection_pages_store(collection_name)
    return result


@router.put("/rename", response_model=OperationResponse)
async def rename_collection(request: RenameCollectionRequest):
    """Rename a collection"""
    result = collection_manager.rename_collection(request.old_name, request.new_name)
    pdf_storage.rename_collection_pdfs(request.old_name, request.new_name)
    document_processor.rename_collection_pages_store(request.old_name, request.new_name)
    return OperationResponse(**result)


@router.post("/{collection_name}/pdfs/add")
async def add_pdfs_to_collection(
    collection_name: str, files: List[UploadFile] = File(...)
):
    """Add new PDFs to an existing collection"""
    try:
        result = await document_processor.process_files(files, collection_name)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{collection_name}/pdfs/{filename}")
async def delete_pdf_from_collection(collection_name: str, filename: str):
    """Delete a specific PDF from a collection"""
    result = collection_manager.delete_pdf_from_collection(collection_name, filename)
    pdf_deleted = pdf_storage.delete_pdf(collection_name, filename)
    document_processor.delete_pdf_pages_store(collection_name, filename)

    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])

    return {**result, "pdf_file_deleted": pdf_deleted}


@router.put("/pdfs/rename", response_model=OperationResponse)
async def rename_pdf_in_collection(request: RenamePDFRequest):
    """Rename a PDF within a collection"""
    result = collection_manager.rename_pdf_in_collection(
        request.collection_name, request.old_filename, request.new_filename
    )
    pdf_storage.rename_pdf(
        request.collection_name, request.old_filename, request.new_filename
    )
    document_processor.rename_pdf_pages_store(
        request.collection_name, request.old_filename, request.new_filename
    )
    return OperationResponse(**result)


@router.get("/{collection_name}/pdfs/{filename:path}/view")
async def view_pdf(collection_name: str, filename: str):
    """Serve PDF file for viewing in browser"""
    pdf_path = pdf_storage.get_pdf_path(collection_name, filename)

    if not pdf_path:
        raise HTTPException(
            status_code=404,
            detail=f"PDF '{filename}' not found in collection '{collection_name}'",
        )

    encoded_filename = quote(filename, safe="")
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{encoded_filename}"},
    )