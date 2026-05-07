// src/components/Chat/ChatInput.tsx
"use client";

import { useState, useRef, useEffect } from "react";
import { PaperAirplaneIcon, StopCircleIcon } from "@heroicons/react/24/solid";

interface ChatInputProps {
  onSend: (message: string) => void;
  onStop?: () => void;
  disabled?: boolean;
  isLoading?: boolean;
  placeholder?: string;
}

export default function ChatInput({
  onSend,
  onStop,
  disabled = false,
  isLoading = false,
  placeholder = "Ask about your documents...",
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const resetHeight = () => {
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const adjustHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  };

  useEffect(() => {
    if (!input) resetHeight();
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled && !isLoading) {
      onSend(input);
      setInput("");
    }
  };

  return (
    <div
      className="flex-shrink-0 px-6 py-4 border-t"
      style={{
        background: "var(--bg-card)",
        borderColor: "var(--border-muted)",
      }}
    >
      <form onSubmit={handleSubmit}>
        <div
          className="flex gap-3 items-center rounded-2xl border px-4 py-3 transition-all"
          style={{
            background: "var(--bg-input)",
            borderColor: "var(--border-soft)",
          }}
          onFocus={() => {
            /* optional focus ring */
          }}
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              adjustHeight();
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className="flex-1 text-[15px] outline-none resize-none overflow-y-auto bg-transparent"
            style={{
              color: "var(--text-primary)",
              minHeight: "1.5rem",
              maxHeight: "12.5rem",
              lineHeight: "1.6",
              height: "auto",  
            }}
          />

          {isLoading ? (
            <button
              type="button"
              onClick={onStop}
              className="rounded-xl w-9 h-9 flex items-center justify-center transition-all flex-shrink-0"
              style={{ background: "var(--accent)" }}
            >
              <svg
                viewBox="0 0 24 24"
                className="w-3.5 h-3.5"
                fill="currentColor"
                style={{ color: "#fff" }}
              >
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim() || disabled}
              className="rounded-xl w-9 h-9 flex items-center justify-center transition-all flex-shrink-0 disabled:cursor-not-allowed"
              style={{
                background:
                  !input.trim() || disabled
                    ? "var(--bg-surface)"
                    : "var(--accent)",
              }}
            >
              <svg
                viewBox="0 0 24 24"
                className="w-3.5 h-3.5"
                fill="none"
                stroke="currentColor"
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{
                  color:
                    !input.trim() || disabled ? "var(--text-muted)" : "#fff",
                  transform: "rotate(45deg)",
                }}
              >
                <line x1="12" y1="19" x2="12" y2="5" />
                <polyline points="5 12 12 5 19 12" />
              </svg>
            </button>
          )}
        </div>
      </form>

      <p
        className="text-[11px] mt-2 italic px-1"
        style={{ color: "var(--text-muted)", fontFamily: "var(--font-serif)" }}
      >
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  );
}
