// src/components/Chat/MessageList.tsx
"use client";

import { useEffect, useRef } from "react";
import { Message as MessageType } from "@/lib/types/message";
import Message from "./Message";

interface MessageListProps {
  messages: MessageType[];
}

export default function MessageList({ messages }: MessageListProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-8">
        <div className="text-center max-w-sm">
          {/* Decorative book icon */}
          <div
            className="w-16 h-16 rounded-2xl mx-auto mb-5 flex items-center justify-center border"
            style={{
              background: "var(--bg-card)",
              borderColor: "var(--border-soft)",
            }}
          >
            <svg
              className="w-8 h-8"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              strokeWidth={1.2}
              style={{ color: "var(--text-muted)" }}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25"
              />
            </svg>
          </div>
          <h3
            className="text-[20px] font-semibold mb-2"
            style={{
              fontFamily: "var(--font-serif)",
              color: "var(--text-primary)",
            }}
          >
            Start a conversation
          </h3>
          {/* <p className="text-[13px]" style={{ color: "var(--text-muted)" }}>
            Ask questions about your documents
          </p> */}
          <p
            className="text-[14px] italic"
            style={{
              color: "var(--text-muted)",
              fontFamily: "var(--font-serif)",
            }}
          >
            Ask questions about your documents
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-5">
      {messages.map((message) => (
        <Message key={message.id} message={message} />
      ))}
      <div ref={endRef} />
    </div>
  );
}
