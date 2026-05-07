// src/components/Chat/ChatArea.tsx
"use client";

import { Bars3Icon } from "@heroicons/react/24/outline";
import { Message } from "@/lib/types/message";
import { ArrowDownTrayIcon } from "@heroicons/react/24/outline";
import MessageList from "./MessageList";
import ChatInput from "./ChatInput";
import { useState, useRef, useEffect } from "react";

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

  // const handleExport = () => {
  //   const content = messages
  //     .filter((m) => m.content)
  //     .map((m) => {
  //       const role = m.type === "user" ? "You" : "GutMiScholar";
  //       return `${role}:\n${m.content}`;
  //     })
  //     .join("\n\n---\n\n");

  //   const header = `GutMiScholar Chat Export\nCollection: ${selectedCollection || "All Collections"}\nDate: ${new Date().toLocaleDateString()}\n\n${"=".repeat(50)}\n\n`;

  //   const blob = new Blob([header + content], { type: "text/plain" });
  //   const url = URL.createObjectURL(blob);
  //   const a = document.createElement("a");
  //   a.href = url;
  //   a.download = `GutMiScholar-${selectedCollection || "chat"}-${Date.now()}.txt`;
  //   a.click();
  //   URL.revokeObjectURL(url);
  // };

  //  EXPORT CHAT OPTION
  const APPLICATION_NAME = "GutMiScholar";
  const [showExportMenu, setShowExportMenu] = useState(false);
  const exportMenuRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (
        exportMenuRef.current &&
        !exportMenuRef.current.contains(e.target as Node)
      ) {
        setShowExportMenu(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Export functions
  const getContent = () =>
    messages
      .filter((m) => m.content)
      .map((m) => {
        const role = m.type === "user" ? "You" : "GutMiScholar";
        return { role, content: m.content };
      });

  const exportAsTxt = () => {
    const header = `${APPLICATION_NAME} Chat Export\nCollection: ${selectedCollection || "All Collections"}\nDate: ${new Date().toLocaleDateString()}\n\n${"=".repeat(50)}\n\n`;
    const body = getContent()
      .map((m) => `${m.role}:\n${m.content}`)
      .join("\n\n---\n\n");
    download(header + body, "txt", "text/plain");
  };

  const exportAsMd = () => {
    const header = `# ${APPLICATION_NAME} Chat Export\n**Collection:** ${selectedCollection || "All Collections"}  \n**Date:** ${new Date().toLocaleDateString()}\n\n---\n\n`;
    const body = getContent()
      .map((m) =>
        m.role === "You"
          ? `**You:**\n${m.content}`
          : `**${APPLICATION_NAME}:**\n${m.content}`,
      )
      .join("\n\n---\n\n");
    download(header + body, "md", "text/markdown");
  };

  const exportAsPdf = () => {
    const content = getContent();
    const html = `
    <html><head><style>
      body { font-family: Georgia, serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #1a1a18; line-height: 1.7; }
      h1 { font-size: 24px; border-bottom: 1px solid #DDD8CE; padding-bottom: 12px; }
      .meta { color: #908c84; font-size: 14px; margin-bottom: 32px; }
      .message { margin-bottom: 28px; }
      .role { font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; color: #0F6E56; margin-bottom: 6px; }
      .user .role { color: #5a5750; }
      .content { font-size: 15px; }
      .divider { border: none; border-top: 0.5px solid #DDD8CE; margin: 24px 0; }
    </style></head><body>
      <h1>${APPLICATION_NAME} Chat Export</h1>
      <div class="meta">Collection: ${selectedCollection || "All Collections"} &nbsp;·&nbsp; ${new Date().toLocaleDateString()}</div>
      ${content
        .map(
          (m) => `
        <div class="message ${m.role === "You" ? "user" : "ai"}">
          <div class="role">${m.role}</div>
          <div class="content">${m.content.replace(/\n/g, "<br/>")}</div>
        </div>
        <hr class="divider"/>
      `,
        )
        .join("")}
    </body></html>
  `;
    const win = window.open("", "_blank");
    if (win) {
      win.document.write(html);
      win.document.close();
      win.print(); // triggers Save as PDF dialog
    }
  };

  const download = (content: string, ext: string, type: string) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${APPLICATION_NAME}-${selectedCollection || "chat"}-${Date.now()}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
    setShowExportMenu(false);
  };

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
            className="text-[22px] font-semibold leading-tight truncate"
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
        {/* Export Chat Button */}
        {messages.length > 0 && (
          <div className="relative flex-shrink-0" ref={exportMenuRef}>
            <button
              onClick={() => setShowExportMenu(!showExportMenu)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] transition-all"
              style={{
                color: "var(--text-secondary)",
                border: "0.5px solid var(--border-soft)",
                background: showExportMenu
                  ? "var(--bg-surface)"
                  : "transparent",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background = "var(--bg-surface)")
              }
              onMouseLeave={(e) => {
                if (!showExportMenu)
                  e.currentTarget.style.background = "transparent";
              }}
            >
              <ArrowDownTrayIcon className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Export</span>
            </button>

            {/* Dropdown menu */}
            {showExportMenu && (
              <div
                className="absolute right-0 top-9 z-50 rounded-xl py-1.5 min-w-[200px] border"
                style={{
                  background: "var(--bg-card)",
                  borderColor: "var(--border-soft)",
                  boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
                }}
              >
                {[
                  { label: "Markdown (.md)", icon: "📝", action: exportAsMd },
                  {
                    label: "Plain Text (.txt)",
                    icon: "📄",
                    action: exportAsTxt,
                  },
                  { label: "PDF Document", icon: "🖨️", action: exportAsPdf },
                ].map((item) => (
                  <button
                    key={item.label}
                    onClick={item.action}
                    className="w-full flex items-center gap-3 px-4 py-2.5 text-[13px] transition-colors text-left"
                    style={{ color: "var(--text-primary)" }}
                    onMouseEnter={(e) =>
                      (e.currentTarget.style.background = "var(--bg-surface)")
                    }
                    onMouseLeave={(e) =>
                      (e.currentTarget.style.background = "transparent")
                    }
                  >
                    <span>{item.icon}</span>
                    <span>{item.label}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
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
