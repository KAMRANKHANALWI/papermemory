// src/lib/types/api.ts

export interface ApiResponse<T = any> {
  status: "success" | "error";
  message?: string;
  result?: T;
  data?: T;
}

export interface UploadResult {
  files_processed: number;
  chunks_created: number;
  collection: string;
  processed_files?: string[];
}

export interface OperationResponse {
  status: "success" | "error";
  message: string;
  data?: any;
}

export interface SearchResult {
  content: string;
  filename: string;
  collection?: string;
  similarity: number;
  page_numbers?: string;
  title?: string;
}

export interface FileSearchResponse {
  filename: string;
  collection_name?: string;
  found: boolean;
  num_results: number;
  results: SearchResult[];
}
