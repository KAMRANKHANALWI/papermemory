// src/components/Chat/MessageList.tsx
"use client";

import { useEffect, useRef } from "react";
import { Message as MessageType } from "@/lib/types/message";
import Message from "./Message";

interface MessageListProps {
  messages: MessageType[];
}

export default function MessageList({ messages }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-8">
        <div className="text-center text-gray-500 max-w-md">
          <div className="w-16 h-16 bg-white rounded-lg mx-auto mb-4 flex items-center justify-center border border-gray-200 shadow-sm">
            <svg
              className="w-8 h-8 text-gray-400"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z" />
              <path d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium mb-2 text-gray-900">Start a conversation</h3>
          <p>Ask questions about your documents</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-6">
      {messages.map((message) => (
        <Message key={message.id} message={message} />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
}
