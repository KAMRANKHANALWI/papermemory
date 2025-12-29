// src/lib/api/memory.ts

import { apiClient } from "./client";
import type { ConversationMemory } from "../types/message";
import type { OperationResponse } from "../types/api";

export const memoryApi = {
  // Add message to memory
  async addMessage(
    chatId: string,
    role: "user" | "assistant",
    content: string,
    collectionName?: string
  ): Promise<OperationResponse> {
    return apiClient.post<OperationResponse>(`/api/memory/${chatId}/add`, {
      role,
      content,
      collection_name: collectionName,
    });
  },

  // Get conversation history
  async getHistory(chatId: string, maxMessages: number = 10): Promise<ConversationMemory> {
    return apiClient.get<ConversationMemory>(
      `/api/memory/${chatId}?max_messages=${maxMessages}`
    );
  },

  // Clear conversation memory
  async clear(chatId: string): Promise<OperationResponse> {
    return apiClient.delete<OperationResponse>(`/api/memory/${chatId}`);
  },

  // Get conversation summary
  async getSummary(chatId: string): Promise<{ chat_id: string; summary: string; total_messages: number }> {
    return apiClient.get<{ chat_id: string; summary: string; total_messages: number }>(
      `/api/memory/${chatId}/summary`
    );
  },
};
