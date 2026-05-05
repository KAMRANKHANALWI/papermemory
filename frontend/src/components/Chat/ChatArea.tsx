// src/components/Chat/ChatArea.tsx
"use client";

import { Bars3Icon } from "@heroicons/react/24/outline";
import { Message } from "@/lib/types/message";
import MessageList from "./MessageList";
import ChatInput from "./ChatInput";

interface ChatAreaProps {
  messages: Message[];
  isLoading: boolean;
  selectedCollection: string | null;
  chatMode: "single" | "chatall" | "selected";
  pdfSelectionMode?: boolean;
  selectedPDFsCount?: number;
  onSendMessage: (message: string) => void;
  onStopGeneration?: () => void;
  onOpenSidebar?: () => void;
}

export default function ChatArea({
  messages,
  isLoading,
  selectedCollection,
  chatMode,
  pdfSelectionMode = false,
  selectedPDFsCount = 0,
  onSendMessage,
  onStopGeneration,
  onOpenSidebar,
}: ChatAreaProps) {
  const getPlaceholder = () => {
    if (chatMode === "selected") {
      return selectedPDFsCount === 0
        ? "Select PDFs first..."
        : "Ask about your selected PDFs...";
    }
    if (chatMode === "single" && !selectedCollection)
      return "Select a collection first...";
    return "Ask about your documents...";
  };

  const isDisabled =
    (chatMode === "single" && !selectedCollection) ||
    (chatMode === "selected" && selectedPDFsCount === 0);

  const getHeaderInfo = () => {
    if (chatMode === "selected") {
      return {
        title:
          selectedPDFsCount > 0
            ? `${selectedPDFsCount} PDF${selectedPDFsCount > 1 ? "s" : ""} selected`
            : "Select PDFs",
        subtitle:
          selectedPDFsCount > 0
            ? "Chatting with selected documents"
            : "Choose PDFs from the sidebar to begin",
      };
    }
    if (chatMode === "single") {
      return {
        title: selectedCollection || "Select a collection",
        subtitle: selectedCollection
          ? `Chatting with ${selectedCollection}`
          : "Choose a collection to start",
      };
    }
    return {
      title: "All Collections",
      subtitle: "Searching across all documents",
    };
  };

  const { title, subtitle } = getHeaderInfo();

  return (
    <div
      className="flex-1 flex flex-col h-full"
      style={{ background: "var(--bg-main)" }}
    >
      {/* Header */}
      <div
        className="flex-shrink-0 px-4 py-4 border-b flex items-center gap-3"
        style={{
          background: "var(--bg-card)",
          borderColor: "var(--border-muted)",
        }}
      >
        {/* Mobile hamburger */}
        <button
          onClick={onOpenSidebar}
          className="lg:hidden p-1.5 rounded-lg flex-shrink-0"
          style={{ color: "var(--text-muted)" }}
        >
          <Bars3Icon className="h-5 w-5" />
        </button>

        <div className="flex-1 min-w-0">
          <h2
            className="text-[18px] font-semibold leading-tight truncate"
            style={{
              fontFamily: "var(--font-serif)",
              color: "var(--text-primary)",
            }}
          >
            {title}
          </h2>
          <p
            className="text-[12px] mt-0.5 truncate"
            style={{ color: "var(--text-muted)" }}
          >
            {subtitle}
          </p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <MessageList messages={messages} />
      </div>

      {/* Input */}
      <ChatInput
        onSend={onSendMessage}
        onStop={onStopGeneration}
        disabled={isDisabled}
        isLoading={isLoading}
        placeholder={getPlaceholder()}
      />
    </div>
  );
}
