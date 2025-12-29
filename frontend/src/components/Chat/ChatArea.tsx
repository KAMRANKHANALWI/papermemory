// src/components/Chat/ChatArea.tsx
"use client";

import { Message } from "@/lib/types/message";
import MessageList from "./MessageList";
import ChatInput from "./ChatInput";

interface ChatAreaProps {
  messages: Message[];
  isLoading: boolean;
  selectedCollection: string | null;
  chatMode: "single" | "chatall";
  onSendMessage: (message: string) => void;
}

export default function ChatArea({
  messages,
  isLoading,
  selectedCollection,
  chatMode,
  onSendMessage,
}: ChatAreaProps) {
  const getPlaceholder = () => {
    if (chatMode === "single" && !selectedCollection) {
      return "Select a collection first...";
    }
    return "Ask about your documents...";
  };

  const isDisabled = (chatMode === "single" && !selectedCollection) || isLoading;

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Header */}
      <div className="bg-white p-4 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="font-semibold text-gray-900">
              {chatMode === "single"
                ? selectedCollection || "Select a collection"
                : "Chat All Collections"}
            </h2>
            <p className="text-sm text-gray-500">
              {chatMode === "single"
                ? selectedCollection
                  ? `Chatting with ${selectedCollection}`
                  : "Choose a collection to start"
                : "Search across all documents"}
            </p>
          </div>

          {isLoading && (
            <div className="flex items-center text-amber-600 text-sm">
              <div className="animate-spin w-4 h-4 border-2 border-amber-600 border-t-transparent rounded-full mr-2"></div>
              Thinking...
            </div>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto bg-gray-50">
        <MessageList messages={messages} />
      </div>

      {/* Input */}
      <ChatInput
        onSend={onSendMessage}
        disabled={isDisabled}
        placeholder={getPlaceholder()}
      />
    </div>
  );
}
