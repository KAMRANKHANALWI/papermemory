// src/lib/api/collections.ts

import { apiClient } from "./client";
import type {
  Collection,
  CollectionPDFsResponse,
  CollectionStats,
  GenerateNameResponse,
  ValidateNameResponse,
  PDFDetail,
} from "../types/collection";
import type {
  ApiResponse,
  UploadResult,
  OperationResponse,
} from "../types/api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const collectionsApi = {
  // Get all collections
  async getAll(): Promise<Collection[]> {
    return apiClient.get<Collection[]>("/api/collections");
  },

  // Upload files to create/add to collection
  async upload(
    collectionName: string,
    files: File[]
  ): Promise<ApiResponse<UploadResult>> {
    return apiClient.uploadFiles<ApiResponse<UploadResult>>(
      `/api/collections/${collectionName}/upload`,
      files
    );
  },

  // Add files to existing collection
  async addPDFs(
    collectionName: string,
    files: File[]
  ): Promise<ApiResponse<UploadResult>> {
    return apiClient.uploadFiles<ApiResponse<UploadResult>>(
      `/api/collections/${collectionName}/pdfs/add`,
      files
    );
  },

  // Delete collection
  async delete(collectionName: string): Promise<OperationResponse> {
    return apiClient.delete<OperationResponse>(
      `/api/collections/${collectionName}`
    );
  },

  // Rename collection
  async rename(oldName: string, newName: string): Promise<OperationResponse> {
    return apiClient.put<OperationResponse>("/api/collections/rename", {
      old_name: oldName,
      new_name: newName,
    });
  },

  // Get PDFs in collection
  async getPDFs(collectionName: string): Promise<CollectionPDFsResponse> {
    return apiClient.get<CollectionPDFsResponse>(
      `/api/collections/${collectionName}/pdfs`
    );
  },

  // Get collection statistics
  async getStats(collectionName: string): Promise<CollectionStats> {
    return apiClient.get<CollectionStats>(
      `/api/collections/${collectionName}/stats`
    );
  },

  async deletePDF(
    collectionName: string,
    filename: string
  ): Promise<OperationResponse> {
    const encoded = encodeURIComponent(filename);

    return apiClient.delete<OperationResponse>(
      `/api/collections/${collectionName}/pdfs/${encoded}`
    );
  },

  // Rename PDF in collection
  async renamePDF(
    collectionName: string,
    oldFilename: string,
    newFilename: string
  ): Promise<OperationResponse> {
    return apiClient.put<OperationResponse>("/api/collections/pdfs/rename", {
      collection_name: collectionName,
      old_filename: oldFilename,
      new_filename: newFilename,
    });
  },

  // View/Open PDF in new tab
  viewPDF(collectionName: string, filename: string): void {
    const encoded = encodeURIComponent(filename);
    const url = `${API_BASE_URL}/api/collections/${collectionName}/pdfs/${encoded}/view`;
    
    // Open PDF in new tab
    window.open(url, '_blank', 'noopener,noreferrer');
  },

  // Get PDF URL (useful for iframe or custom rendering)
  getPDFUrl(collectionName: string, filename: string): string {
    const encoded = encodeURIComponent(filename);
    return `${API_BASE_URL}/api/collections/${collectionName}/pdfs/${encoded}/view`;
  },

  // Generate collection name
  async generateName(filenames: string[]): Promise<GenerateNameResponse> {
    return apiClient.post<GenerateNameResponse>(
      "/api/collections/generate-name",
      {
        filenames,
        upload_type: "files",
      }
    );
  },

  // Validate collection name
  async validateName(name: string): Promise<ValidateNameResponse> {
    return apiClient.post<ValidateNameResponse>(
      "/api/collections/validate-name",
      {
        name,
      }
    );
  },
};