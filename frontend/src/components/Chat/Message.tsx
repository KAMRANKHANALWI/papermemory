// src/components/Chat/Message.tsx
"use client";

import { Message as MessageType } from "@/lib/types/message";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeRaw from "rehype-raw";
import SourcesDisplay from "./SourcesDisplay";
import "highlight.js/styles/github.css";
import { ClipboardIcon, CheckIcon } from "@heroicons/react/24/outline";
import { useState } from "react";

interface MessageProps {
  message: MessageType;
}

export default function Message({ message }: MessageProps) {
  const isUser = message.type === "user";
  const [copied, setCopied] = useState(false);

  // To Clear the mid way stopped chat
  if (!isUser && !message.content && !message.isLoading) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} group`}>
      <div className="relative max-w-[80%]">
        <div
          className="rounded-2xl px-5 py-4 text-[15px] leading-relaxed"
          style={
            isUser
              ? {
                  background: "var(--bg-surface)",
                  color: "var(--text-primary)",
                  borderBottomRightRadius: 4,
                }
              : {
                  background: "var(--bg-card)",
                  color: "var(--text-primary)",
                  border: "0.5px solid var(--border-soft)",
                  borderBottomLeftRadius: 4,
                }
          }
        >
          {message.isLoading ? (
            <div
              className="flex items-center gap-2"
              style={{ color: "var(--text-muted)" }}
            >
              <span
                className="inline-block w-1.5 h-1.5 rounded-full animate-bounce"
                style={{
                  background: "var(--text-muted)",
                  animationDelay: "0ms",
                }}
              />
              <span
                className="inline-block w-1.5 h-1.5 rounded-full animate-bounce"
                style={{
                  background: "var(--text-muted)",
                  animationDelay: "150ms",
                }}
              />
              <span
                className="inline-block w-1.5 h-1.5 rounded-full animate-bounce"
                style={{
                  background: "var(--text-muted)",
                  animationDelay: "300ms",
                }}
              />
            </div>
          ) : isUser ? (
            <div className="whitespace-pre-wrap">{message.content}</div>
          ) : (
            <>
              <div className="prose-parchment prose prose-sm max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight, rehypeRaw]}
                  components={{
                    h1: ({ node, ...props }) => (
                      <h1
                        style={{
                          fontFamily: "var(--font-serif)",
                          color: "var(--text-primary)",
                          fontSize: "1.4rem",
                          fontWeight: 600,
                          marginTop: "1rem",
                          marginBottom: "0.5rem",
                        }}
                        {...props}
                      />
                    ),
                    h2: ({ node, ...props }) => (
                      <h2
                        style={{
                          fontFamily: "var(--font-serif)",
                          color: "var(--text-primary)",
                          fontSize: "1.2rem",
                          fontWeight: 600,
                          marginTop: "0.8rem",
                          marginBottom: "0.4rem",
                        }}
                        {...props}
                      />
                    ),
                    h3: ({ node, ...props }) => (
                      <h3
                        style={{
                          fontFamily: "var(--font-serif)",
                          color: "var(--text-primary)",
                          fontSize: "1.05rem",
                          fontWeight: 600,
                        }}
                        {...props}
                      />
                    ),
                    p: ({ node, ...props }) => (
                      <p
                        style={{
                          marginBottom: "0.6rem",
                          lineHeight: 1.7,
                          color: "var(--text-primary)",
                        }}
                        {...props}
                      />
                    ),
                    ul: ({ node, ...props }) => (
                      <ul
                        style={{
                          listStyleType: "disc",
                          paddingLeft: "1.2rem",
                          marginBottom: "0.6rem",
                          color: "var(--text-primary)",
                        }}
                        {...props}
                      />
                    ),
                    ol: ({ node, ...props }) => (
                      <ol
                        style={{
                          listStyleType: "decimal",
                          paddingLeft: "1.2rem",
                          marginBottom: "0.6rem",
                          color: "var(--text-primary)",
                        }}
                        {...props}
                      />
                    ),
                    li: ({ node, ...props }) => (
                      <li style={{ marginBottom: "0.2rem" }} {...props} />
                    ),
                    code: ({ node, inline, ...props }: any) =>
                      inline ? (
                        <code
                          style={{
                            background: "var(--bg-surface)",
                            borderRadius: 4,
                            padding: "1px 5px",
                            fontSize: "0.85em",
                            fontFamily: "var(--font-mono)",
                            color: "var(--text-primary)",
                          }}
                          {...props}
                        />
                      ) : (
                        <code
                          style={{
                            display: "block",
                            background: "var(--bg-surface)",
                            padding: "0.75rem",
                            borderRadius: 8,
                            overflowX: "auto",
                            fontSize: "0.875em",
                          }}
                          {...props}
                        />
                      ),
                    pre: ({ node, ...props }) => (
                      <pre
                        style={{
                          background: "var(--bg-surface)",
                          borderRadius: 8,
                          marginBottom: "0.6rem",
                          overflowX: "auto",
                        }}
                        {...props}
                      />
                    ),
                    blockquote: ({ node, ...props }) => (
                      <blockquote
                        style={{
                          borderLeft: "3px solid var(--border-soft)",
                          paddingLeft: "1rem",
                          color: "var(--text-secondary)",
                          fontStyle: "italic",
                          margin: "0.5rem 0",
                        }}
                        {...props}
                      />
                    ),
                    a: ({ node, ...props }) => (
                      <a
                        style={{
                          color: "var(--text-accent)",
                          textDecoration: "underline",
                        }}
                        {...props}
                      />
                    ),
                    strong: ({ node, ...props }) => (
                      <strong
                        style={{
                          fontWeight: 600,
                          color: "var(--text-primary)",
                        }}
                        {...props}
                      />
                    ),
                    table: ({ node, ...props }) => (
                      <table
                        style={{
                          width: "100%",
                          borderCollapse: "collapse",
                          marginBottom: "0.6rem",
                          fontSize: "0.875em",
                        }}
                        {...props}
                      />
                    ),
                    th: ({ node, ...props }) => (
                      <th
                        style={{
                          borderBottom: "1px solid var(--border-soft)",
                          padding: "6px 10px",
                          textAlign: "left",
                          fontWeight: 600,
                          color: "var(--text-secondary)",
                          background: "var(--bg-surface)",
                        }}
                        {...props}
                      />
                    ),
                    td: ({ node, ...props }) => (
                      <td
                        style={{
                          borderBottom: "0.5px solid var(--border-muted)",
                          padding: "6px 10px",
                          color: "var(--text-primary)",
                        }}
                        {...props}
                      />
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
        {/* Copy button */}
        {!message.isLoading && message.content && (
          <div
            className={`flex mt-1.5 ${isUser ? "justify-end" : "justify-start"}`}
          >
            <button
              onClick={handleCopy}
              className="opacity-0 group-hover:opacity-100 transition-all duration-150 flex items-center gap-1 px-2 py-1 rounded-lg text-[11px]"
              style={{
                color: "var(--text-muted)",
                background: "var(--bg-card)",
                border: "0.5px solid var(--border-soft)",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.color = "var(--text-primary)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.color = "var(--text-muted)")
              }
            >
              {copied ? (
                <>
                  <CheckIcon
                    className="w-4 h-4"
                    style={{ color: "var(--accent)" }}
                  />{" "}
                  Copied
                </>
              ) : (
                <ClipboardIcon className="w-4 h-4" />
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
