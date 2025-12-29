// src/lib/types/collection.ts

export interface Collection {
  name: string;
  chunk_count: number;
  file_count: number;
}

export interface PDFDetail {
  filename: string;
  title: string;
  chunks: number;
  pages: number;
  page_range: string;
}

export interface CollectionStats {
  name: string;
  total_pdfs: number;
  total_chunks: number;
  pdfs: PDFDetail[];
}

export interface CollectionPDFsResponse {
  collection_name: string;
  total_pdfs: number;
  total_chunks: number;
  pdfs: PDFDetail[];
}

export interface GenerateNameResponse {
  suggested_name: string;
  is_valid: boolean;
  validation_message?: string;
}

export interface ValidateNameResponse {
  is_valid: boolean;
  validated_name?: string;
  message: string;
}
