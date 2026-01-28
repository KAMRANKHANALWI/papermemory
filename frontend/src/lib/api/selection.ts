// src/lib/api/selection.ts

import { apiClient } from "./client";
import type {
  PDFSelectionResponse,
  SelectionSessionResponse,
  SelectionStatsResponse,
  SelectedPDFsSearchResponse,
  SelectPDFRequest,
  BatchSelectPDFsRequest,
  SelectedPDFsSearchRequest,
} from "../types/selection";

export const selectionApi = {
  // Select a single PDF
  async selectPDF(
    sessionId: string,
    filename: string,
    collectionName: string
  ): Promise<PDFSelectionResponse> {
    return apiClient.post<PDFSelectionResponse>(
      `/api/selection/${sessionId}/select`,
      {
        filename,
        collection_name: collectionName,
      } as SelectPDFRequest
    );
  },

  // Deselect a PDF
  async deselectPDF(
    sessionId: string,
    filename: string,
    collectionName: string
  ): Promise<PDFSelectionResponse> {
    return apiClient.post<PDFSelectionResponse>(
      `/api/selection/${sessionId}/deselect`,
      {
        filename,
        collection_name: collectionName,
      }
    );
  },

  // Batch select multiple PDFs
  async batchSelectPDFs(
    sessionId: string,
    selections: Array<{ filename: string; collection_name: string }>
  ): Promise<PDFSelectionResponse> {
    return apiClient.post<PDFSelectionResponse>(
      `/api/selection/${sessionId}/batch-select`,
      {
        selections,
      } as BatchSelectPDFsRequest
    );
  },

  // Clear all selections
  async clearSelection(sessionId: string): Promise<PDFSelectionResponse> {
    return apiClient.delete<PDFSelectionResponse>(
      `/api/selection/${sessionId}/clear`
    );
  },

  // Get current selection
  async getSelection(sessionId: string): Promise<SelectionSessionResponse> {
    return apiClient.get<SelectionSessionResponse>(
      `/api/selection/${sessionId}`
    );
  },

  // Get selection statistics
  async getStats(sessionId: string): Promise<SelectionStatsResponse> {
    return apiClient.get<SelectionStatsResponse>(
      `/api/selection/${sessionId}/stats`
    );
  },

  // Search within selected PDFs
  async searchSelected(
    sessionId: string,
    query: string,
    numResults: number = 25
  ): Promise<SelectedPDFsSearchResponse> {
    return apiClient.post<SelectedPDFsSearchResponse>(
      `/api/selection/${sessionId}/search`,
      {
        query,
        num_results: numResults,
      } as SelectedPDFsSearchRequest
    );
  },

  // Chat with selected PDFs (returns EventSource for streaming)
  createSelectedPDFsStream(
    sessionId: string,
    query: string,
    chatId?: string,
    numResults: number = 25
  ): EventSource {
    const API_BASE_URL =
      process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const url = new URL(`/api/selection/${sessionId}/chat`, API_BASE_URL);

    // Add query parameters to URL (EventSource uses GET, so params go in URL)
    url.searchParams.append("query", query);
    if (chatId) {
      url.searchParams.append("chat_id", chatId);
    }
    url.searchParams.append("num_results", numResults.toString());

    return new EventSource(url.toString());
  },
};
