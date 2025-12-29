// src/components/Chat/Message.tsx
"use client";

import { Message as MessageType } from "@/lib/types/message";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeRaw from "rehype-raw";
import SourcesDisplay from "./SourcesDisplay";
import "highlight.js/styles/github-dark.css";

interface MessageProps {
  message: MessageType;
}

export default function Message({ message }: MessageProps) {
  return (
    <div className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-lg p-4 ${
          message.type === "user"
            ? "bg-amber-100 text-gray-900"
            : "bg-white border border-gray-200 shadow-sm text-gray-900"
        }`}
      >
        {message.isLoading ? (
          <div className="flex items-center text-gray-600">
            <div className="animate-pulse">Thinking...</div>
          </div>
        ) : message.type === "user" ? (
          <div className="whitespace-pre-wrap">{message.content}</div>
        ) : (
          <>
            <div className="prose prose-gray prose-sm max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeHighlight, rehypeRaw]}
                components={{
                  h1: ({ node, ...props }) => (
                    <h1 className="text-2xl font-bold mt-4 mb-2 text-gray-900" {...props} />
                  ),
                  h2: ({ node, ...props }) => (
                    <h2 className="text-xl font-bold mt-3 mb-2 text-gray-900" {...props} />
                  ),
                  h3: ({ node, ...props }) => (
                    <h3 className="text-lg font-bold mt-2 mb-1 text-gray-900" {...props} />
                  ),
                  p: ({ node, ...props }) => (
                    <p className="mb-2 leading-relaxed text-gray-800" {...props} />
                  ),
                  ul: ({ node, ...props }) => (
                    <ul className="list-disc list-inside mb-2 space-y-1 text-gray-800" {...props} />
                  ),
                  ol: ({ node, ...props }) => (
                    <ol className="list-decimal list-inside mb-2 space-y-1 text-gray-800" {...props} />
                  ),
                  li: ({ node, ...props }) => <li className="ml-2" {...props} />,
                  code: ({ node, inline, ...props }: any) =>
                    inline ? (
                      <code
                        className="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-mono text-gray-800"
                        {...props}
                      />
                    ) : (
                      <code
                        className="block bg-gray-100 p-3 rounded my-2 overflow-x-auto text-sm"
                        {...props}
                      />
                    ),
                  pre: ({ node, ...props }) => (
                    <pre className="bg-gray-100 rounded my-2 overflow-x-auto" {...props} />
                  ),
                  blockquote: ({ node, ...props }) => (
                    <blockquote
                      className="border-l-4 border-gray-300 pl-4 italic my-2 text-gray-700"
                      {...props}
                    />
                  ),
                  a: ({ node, ...props }) => (
                    <a className="text-amber-600 hover:underline" {...props} />
                  ),
                  strong: ({ node, ...props }) => (
                    <strong className="font-bold text-gray-900" {...props} />
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>

            {message.sources && <SourcesDisplay sources={message.sources} />}
          </>
        )}
      </div>
    </div>
  );
}
