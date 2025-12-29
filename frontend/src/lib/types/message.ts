// src/lib/types/message.ts

export interface Message {
  id: string;
  type: "user" | "ai";
  content: string;
  sources?: DocumentSource[];
  isLoading?: boolean;
  timestamp?: string;
}

export interface DocumentSource {
  content: string;
  filename: string;
  collection?: string;
  similarity: number;
  page_numbers?: string;
  title?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
}

export interface ConversationMemory {
  chat_id: string;
  message_count: number;
  messages: ChatMessage[];
}

export interface QueryClassification {
  classification: "list_pdfs" | "count_pdfs" | "list_collections" | "file_specific_search" | "content_search";
  filename?: string;
  confidence: number;
}
