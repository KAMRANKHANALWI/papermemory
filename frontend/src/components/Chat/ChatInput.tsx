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
    if (textareaRef.current) textareaRef.current.style.height = "3rem";
  };

  const adjustHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "3rem";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  };

  useEffect(() => { if (!input) resetHeight(); }, [input]);

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
      style={{ background: "var(--bg-card)", borderColor: "var(--border-muted)" }}
    >
      <form onSubmit={handleSubmit} className="flex gap-3 items-end">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={e => { setInput(e.target.value); adjustHeight(); }}
          onKeyDown={e => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(e); }
          }}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          className="flex-1 rounded-xl px-4 py-3 text-[14px] outline-none resize-none overflow-hidden border transition-all"
          style={{
            background: "var(--bg-input)",
            borderColor: "var(--border-soft)",
            color: "var(--text-primary)",
            minHeight: "3rem",
            maxHeight: "12.5rem",
          }}
          onFocus={e => (e.target.style.borderColor = "var(--accent)")}
          onBlur={e => (e.target.style.borderColor = "var(--border-soft)")}
        />

        {isLoading ? (
          <button
            type="button"
            onClick={onStop}
            className="rounded-xl px-4 h-12 flex items-center justify-center transition-colors"
            style={{ background: "var(--text-secondary)", color: "#fff" }}
          >
            <StopCircleIcon className="h-5 w-5" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim() || disabled}
            className="rounded-xl px-4 h-12 flex items-center justify-center transition-colors disabled:cursor-not-allowed"
            style={{
              background: !input.trim() || disabled ? "var(--border-soft)" : "var(--accent)",
              color: !input.trim() || disabled ? "var(--text-muted)" : "#fff",
            }}
          >
            <PaperAirplaneIcon className="h-5 w-5" />
          </button>
        )}
      </form>
      <p className="text-[11px] mt-2" style={{ color: "var(--text-muted)" }}>
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  );
}