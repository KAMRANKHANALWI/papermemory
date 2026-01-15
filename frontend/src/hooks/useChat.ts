// src/hooks/useChat.ts
"use client";

import { useState, useCallback } from "react";
import { chatApi } from "@/lib/api/chat";
import { selectionApi } from "@/lib/api/selection";
import { Message, DocumentSource } from "@/lib/types/message";
import { useToast } from "./useToast";

// NEW: Extended chat mode type
type ChatMode = "single" | "chatall" | "selected";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [currentEventSource, setCurrentEventSource] = useState<EventSource | null>(null);
  const toast = useToast();

  const sendMessage = useCallback(
    async (
      query: string,
      collectionName: string | null,
      mode: ChatMode,
      sessionId?: string,  // For PDF selection mode
      abortSignal?: AbortSignal // To stop the generation
    ) => {
      if (!query.trim()) return;
      
      // Validation based on mode
      if (mode === "single" && !collectionName) {
        toast.error("Please select a collection first");
        return;
      }
      
      if (mode === "selected" && !sessionId) {
        toast.error("Session ID required for PDF selection mode");
        return;
      }

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
        let eventSource: EventSource;

        // NEW: Handle different chat modes
        switch (mode) {
          case "single":
            if (!collectionName) {
              throw new Error("Collection name required for single mode");
            }
            eventSource = chatApi.createSingleCollectionStream(
              collectionName,
              query,
              currentChatId || undefined
            );
            break;

          case "chatall":
            eventSource = chatApi.createAllCollectionsStream(
              query,
              currentChatId || undefined
            );
            break;

          case "selected":
            if (!sessionId) {
              throw new Error("Session ID required for selected mode");
            }
            // Use selection API for chatting with selected PDFs
            eventSource = selectionApi.createSelectedPDFsStream(
              sessionId,
              query,
              currentChatId || undefined
            );
            break;

          default:
            throw new Error(`Invalid chat mode: ${mode}`);
        }

        // NEW: Store the event source so we can close it later
        setCurrentEventSource(eventSource);

        // NEW: Handle abort signal
        if (abortSignal) {
          abortSignal.addEventListener('abort', () => {
            console.log('🛑 Abort signal received - closing EventSource');
            eventSource.close();
            setCurrentEventSource(null);
            setIsLoading(false);
            
            // Update the AI message to show it was stopped
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === aiMessageId
                  ? {
                      ...msg,
                      content: msg.content || "Response generation stopped.",
                      isLoading: false,
                    }
                  : msg
              )
            );
          });
        }

        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            switch (data.type) {
              case "chat_id":
                setCurrentChatId(data.chat_id);
                break;

              case "search_results":
                // Optional: Handle search results if needed
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
        console.error("Chat error:", error);
        toast.error(error.message || "Failed to send message");
        
        // Remove the loading AI message on error
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === aiMessageId
              ? {
                  ...msg,
                  content: `Error: ${error.message || "Failed to send message"}`,
                  isLoading: false,
                }
              : msg
          )
        );
        setIsLoading(false);
      }
    },
    [currentChatId, toast]
  );

   // NEW: Manual stop function (optional - as a fallback)
  const stopGeneration = useCallback(() => {
    console.log('🛑 Manually stopping generation');
    if (currentEventSource) {
      currentEventSource.close();
      setCurrentEventSource(null);
      setIsLoading(false);
      toast.info("Stopped generating response");
    }
  }, [currentEventSource, toast]);

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



