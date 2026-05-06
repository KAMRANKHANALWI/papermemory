// src/hooks/useChat.ts
"use client";

import { useState, useCallback, useRef } from "react";
import { chatApi } from "@/lib/api/chat";
import { selectionApi } from "@/lib/api/selection";
import { Message } from "@/lib/types/message";
import { useToast } from "./useToast";

type ChatMode = "single" | "chatall" | "selected";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const toast = useToast();

  const sendMessage = useCallback(
    async (
      query: string,
      collectionName: string | null,
      mode: ChatMode,
      sessionId?: string,
      // abortSignal?: AbortSignal
    ) => {
      if (!query.trim()) return;

      if (mode === "single" && !collectionName) {
        toast.error("Please select a collection first");
        return;
      }
      if (mode === "selected" && !sessionId) {
        toast.error("Session ID required for PDF selection mode");
        return;
      }

      const controller = new AbortController();
      abortControllerRef.current = controller;
      const signal = controller.signal;

      const messageId = Date.now().toString();
      const userMessage: Message = {
        id: messageId + "-user",
        type: "user",
        content: query,
      };
      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);

      const aiMessageId = messageId + "-ai";
      setMessages((prev) => [
        ...prev,
        {
          id: aiMessageId,
          type: "ai",
          content: "",
          sources: [],
          isLoading: true,
        },
      ]);

      try {
        let response: Response;

        if (mode === "selected") {
          // selection mode still uses EventSource (GET) — unchanged
          const API_BASE_URL =
            process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
          const url = new URL(`/api/selection/${sessionId}/chat`, API_BASE_URL);
          url.searchParams.append("query", query);
          if (currentChatId) url.searchParams.append("chat_id", currentChatId);

          const es = new EventSource(url.toString());
          handleEventSource(es, aiMessageId, signal);
          return;
        }

        // single and chatall — use fetch POST with body
        if (mode === "single") {
          response = await chatApi.createSingleCollectionStream(
            collectionName!,
            query,
            currentChatId || undefined,
            signal,
          );
        } else {
          response = await chatApi.createAllCollectionsStream(
            query,
            currentChatId || undefined,
            signal,
          );
        }

        if (!response.body) throw new Error("No response body");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        // Handle abort
        signal?.addEventListener("abort", () => {
          setIsLoading(false);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId
                ? {
                    ...msg,
                    content: msg.content || "Response stopped.",
                    isLoading: false,
                  }
                : msg,
            ),
          );
          reader.cancel();
        });

        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          // if (done) break;
          if (done) {
            // stream ended, force cleanup
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === aiMessageId ? { ...msg, isLoading: false } : msg,
              ),
            );
            setIsLoading(false);
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
              const data = JSON.parse(line.slice(6));
              handleSSEEvent(data, aiMessageId);
            } catch {}
          }
        }
      } catch (error: any) {
        if (error.name === "AbortError") {
          // ← Add cleanup here too
          setIsLoading(false);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId && msg.isLoading
                ? { ...msg, isLoading: false }
                : msg,
            ),
          );
          return;
        }
        toast.error(error.message || "Failed to send message");
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === aiMessageId
              ? { ...msg, content: `Error: ${error.message}`, isLoading: false }
              : msg,
          ),
        );
        setIsLoading(false);
      }
    },
    [currentChatId, toast],
  );

  // SSE event handler (shared between fetch stream and EventSource)
  const handleSSEEvent = useCallback(
    (data: any, aiMessageId: string) => {
      switch (data.type) {
        case "chat_id":
          setCurrentChatId(data.chat_id);
          break;
        case "content":
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId
                ? {
                    ...msg,
                    content: (msg.content || "") + data.content,
                    isLoading: false,
                  }
                : msg,
            ),
          );
          break;
        case "sources":
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId ? { ...msg, sources: data.sources } : msg,
            ),
          );
          break;
        case "end":
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId ? { ...msg, isLoading: false } : msg,
            ),
          );
          setIsLoading(false);
          break;
        case "error":
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId
                ? {
                    ...msg,
                    content: `Error: ${data.message}`,
                    isLoading: false,
                  }
                : msg,
            ),
          );
          setIsLoading(false);
          toast.error(data.message);
          break;
      }
    },
    [toast],
  );

  // EventSource handler (for selected PDFs mode — still GET)
  const handleEventSource = useCallback(
    (
      eventSource: EventSource,
      aiMessageId: string,
      abortSignal?: AbortSignal,
    ) => {
      abortSignal?.addEventListener("abort", () => {
        eventSource.close();
        setIsLoading(false);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === aiMessageId
              ? {
                  ...msg,
                  content: msg.content || "Response stopped.",
                  isLoading: false,
                }
              : msg,
          ),
        );
      });

      eventSource.onmessage = (event) => {
        try {
          handleSSEEvent(JSON.parse(event.data), aiMessageId);
          if (JSON.parse(event.data).type === "end") eventSource.close();
        } catch {}
      };

      eventSource.onerror = () => {
        eventSource.close();
        setIsLoading(false);
        toast.error("Connection error");
      };
    },
    [handleSSEEvent, toast],
  );

  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setCurrentChatId(null);
  }, []);

  return {
    messages,
    isLoading,
    currentChatId,
    sendMessage,
    stopGeneration,
    clearMessages,
  };
}

export type { ChatMode };
