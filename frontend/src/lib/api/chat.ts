// // src/lib/api/chat.ts

// import { apiClient } from "./client";
// import type { QueryClassification, DocumentSource } from "../types/message";
// import type { FileSearchResponse } from "../types/api";

// const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// export const chatApi = {
//   // Classify query
//   async classify(query: string, isChatAllMode: boolean = false): Promise<QueryClassification> {
//     return apiClient.post<QueryClassification>("/api/chat/classify", {
//       query,
//       is_chatall_mode: isChatAllMode,
//     });
//   },

//   // Search in specific file
//   async searchFile(
//     filename: string,
//     query: string,
//     collectionName?: string
//   ): Promise<FileSearchResponse> {
//     return apiClient.post<FileSearchResponse>("/api/search/file", {
//       filename,
//       query,
//       collection_name: collectionName,
//       num_results: 25,
//     });
//   },

//   // Search file across all collections
//   async searchFileAll(filename: string, query: string): Promise<FileSearchResponse> {
//     return apiClient.post<FileSearchResponse>("/api/search/file-all", {
//       filename,
//       query,
//       num_results: 25,
//     });
//   },

//   // Chat with single collection (SSE)
//   createSingleCollectionStream(
//     collectionName: string,
//     message: string,
//     chatId?: string
//   ): EventSource {
//     const url = new URL(`/api/chat/single/${collectionName}/${encodeURIComponent(message)}`, API_BASE_URL);
//     if (chatId) {
//       url.searchParams.append("chat_id", chatId);
//     }
//     return new EventSource(url.toString());
//   },

//   // Chat with all collections (SSE)
//   createAllCollectionsStream(message: string, chatId?: string): EventSource {
//     const url = new URL(`/api/chat/all/${encodeURIComponent(message)}`, API_BASE_URL);
//     if (chatId) {
//       url.searchParams.append("chat_id", chatId);
//     }
//     return new EventSource(url.toString());
//   },
// };


// src/lib/api/chat.ts

import { apiClient } from "./client";
import type { QueryClassification } from "../types/message";
import type { FileSearchResponse } from "../types/api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const chatApi = {
  async classify(query: string, isChatAllMode: boolean = false): Promise<QueryClassification> {
    return apiClient.post<QueryClassification>("/api/chat/classify", {
      query,
      is_chatall_mode: isChatAllMode,
    });
  },

  async searchFile(filename: string, query: string, collectionName?: string): Promise<FileSearchResponse> {
    return apiClient.post<FileSearchResponse>("/api/search/file", {
      filename, query, collection_name: collectionName, num_results: 25,
    });
  },

  async searchFileAll(filename: string, query: string): Promise<FileSearchResponse> {
    return apiClient.post<FileSearchResponse>("/api/search/file-all", {
      filename, query, num_results: 25,
    });
  },

  // POST with body — returns raw Response for stream reading
  async createSingleCollectionStream(
    collectionName: string,
    message: string,
    chatId?: string
  ): Promise<Response> {
    return fetch(`${API_BASE_URL}/api/chat/single/${collectionName}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, chat_id: chatId ?? null, eval: false }),
    });
  },

  async createAllCollectionsStream(
    message: string,
    chatId?: string
  ): Promise<Response> {
    return fetch(`${API_BASE_URL}/api/chat/all`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, chat_id: chatId ?? null, eval: false }),
    });
  },
};