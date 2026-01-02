// src/lib/types/selection.ts

export interface SelectedPDF {
  filename: string;
  collection_name: string;
  title: string;
  pages: number;
  chunks: number;
}

export interface PDFSelectionResponse {
  success: boolean;
  message: string;
  total_selected: number;
  selected_pdfs: SelectedPDF[];
}

export interface SelectionSessionResponse {
  session_id: string;
  total_selected: number;
  collections_involved: string[];
  selected_pdfs: SelectedPDF[];
  created_at: string;
  updated_at: string;
}

export interface SelectionStatsResponse {
  total_selected: number;
  collections_involved: string[];
  pdfs_by_collection: Record<string, number>;
  total_chunks: number;
  total_pages: number;
}

export interface SelectedPDFsSearchResponse {
  query: string;
  total_results: number;
  total_selected_pdfs: number;
  collections_searched: string[];
  results: SearchResult[];
}

export interface SearchResult {
  content: string;
  filename: string;
  collection: string;
  similarity: number;
  page_numbers?: string;
  title?: string;
}

export interface SelectPDFRequest {
  filename: string;
  collection_name: string;
}

export interface BatchSelectPDFsRequest {
  selections: Array<{
    filename: string;
    collection_name: string;
  }>;
}

export interface SelectedPDFsSearchRequest {
  query: string;
  num_results?: number;
}
