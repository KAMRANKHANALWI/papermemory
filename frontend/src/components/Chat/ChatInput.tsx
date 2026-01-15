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

  const resetTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "3rem";
    }
  };

  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "3rem";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  };

  useEffect(() => {
    if (input === "") {
      resetTextareaHeight();
    }
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled && !isLoading) {
      onSend(input);
      setInput(""); // Height will be reset by useEffect when input becomes ""
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleStopClick = () => {
    if (onStop) {
      onStop();
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    adjustTextareaHeight();
  };

  return (
    <div className="bg-white border-t border-gray-200 p-4">
      <form onSubmit={handleSubmit} className="flex gap-3 items-end">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          className="flex-1 bg-white border border-gray-300 rounded-lg px-4 py-3 text-gray-900 placeholder-gray-400 focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500 disabled:opacity-50 disabled:bg-gray-50 resize-none overflow-hidden"
          style={{
            minHeight: "3rem",
            maxHeight: "12.5rem",
          }}
        />
        
        {/* Conditional Button - Send or Stop */}
        {isLoading ? (
          <button
            type="button"
            onClick={handleStopClick}
            className="bg-gray-700 hover:bg-gray-800 px-4 py-3 rounded-lg font-medium transition-colors flex items-center justify-center text-white min-w-[3.5rem] h-12"
            title="Stop generating"
          >
            <StopCircleIcon className="h-5 w-5" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim() || disabled}
            className="bg-amber-600 hover:bg-amber-700 disabled:bg-gray-300 disabled:cursor-not-allowed px-4 py-3 rounded-lg font-medium transition-colors flex items-center justify-center text-white min-w-[3.5rem] h-12"
            title="Send message"
          >
            <PaperAirplaneIcon className="h-5 w-5" />
          </button>
        )}
      </form>
      <p className="text-xs text-gray-500 mt-2">
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  );
}