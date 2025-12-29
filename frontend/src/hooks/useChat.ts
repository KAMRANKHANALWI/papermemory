// src/hooks/useChat.ts
"use client";

import { useState, useCallback } from "react";
import { chatApi } from "@/lib/api/chat";
import { Message, DocumentSource } from "@/lib/types/message";
import { useToast } from "./useToast";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const toast = useToast();

  const sendMessage = useCallback(
    async (
      query: string,
      collectionName: string | null,
      mode: "single" | "chatall"
    ) => {
      if (!query.trim()) return;
      if (mode === "single" && !collectionName) return;

      const messageId = Date.now().toString();
      const userMessage: Message = {
        id: messageId + "-user",
        type: "user",
        content: query,
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);

      const aiMessageId = messageId + "-ai";
      const aiMessage: Message = {
        id: aiMessageId,
        type: "ai",
        content: "",
        sources: [],
        isLoading: true,
      };
      setMessages((prev) => [...prev, aiMessage]);

      try {
        const eventSource =
          mode === "single" && collectionName
            ? chatApi.createSingleCollectionStream(
                collectionName,
                query,
                currentChatId || undefined
              )
            : chatApi.createAllCollectionsStream(query, currentChatId || undefined);

        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            switch (data.type) {
              case "chat_id":
                setCurrentChatId(data.chat_id);
                break;

              case "search_results":
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === aiMessageId
                      ? { ...msg, content: "Searching documents..." }
                      : msg
                  )
                );
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
                      : msg
                  )
                );
                break;

              case "sources":
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === aiMessageId
                      ? { ...msg, sources: data.sources }
                      : msg
                  )
                );
                break;

              case "end":
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === aiMessageId ? { ...msg, isLoading: false } : msg
                  )
                );
                eventSource.close();
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
                      : msg
                  )
                );
                eventSource.close();
                setIsLoading(false);
                toast.error(data.message);
                break;
            }
          } catch (error) {
            console.error("Error parsing SSE data:", error);
          }
        };

        eventSource.onerror = () => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId
                ? {
                    ...msg,
                    content: "Connection error. Please try again.",
                    isLoading: false,
                  }
                : msg
            )
          );
          eventSource.close();
          setIsLoading(false);
          toast.error("Connection error");
        };
      } catch (error: any) {
        toast.error(error.message || "Failed to send message");
        setIsLoading(false);
      }
    },
    [currentChatId, toast]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setCurrentChatId(null);
  }, []);

  return {
    messages,
    isLoading,
    currentChatId,
    sendMessage,
    clearMessages,
  };
}
