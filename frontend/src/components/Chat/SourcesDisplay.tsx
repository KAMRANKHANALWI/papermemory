// src/components/Chat/SourcesDisplay.tsx
"use client";

import { useState } from "react";
import { DocumentSource } from "@/lib/types/message";
import { ChevronDownIcon, ChevronUpIcon, DocumentTextIcon, ArrowTopRightOnSquareIcon, FolderIcon } from "@heroicons/react/24/outline";
import { collectionsApi } from "@/lib/api/collections";
import { useToast } from "@/hooks/useToast";

interface SourcesDisplayProps {
  sources: DocumentSource[];
}

export default function SourcesDisplay({ sources }: SourcesDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandedDocuments, setExpandedDocuments] = useState<Set<string>>(new Set());
  const toast = useToast();

  if (!sources || sources.length === 0) return null;

  const groupedSources = sources.reduce((acc, source) => {
    const key = source.filename;
    if (!acc[key]) acc[key] = [];
    acc[key].push(source);
    return acc;
  }, {} as Record<string, DocumentSource[]>);

  const toggleDocument = (filename: string) => {
    setExpandedDocuments(prev => {
      const next = new Set(prev);
      next.has(filename) ? next.delete(filename) : next.add(filename);
      return next;
    });
  };

  const handleViewPdf = (collection: string, filename: string) => {
    try { collectionsApi.viewPDF(collection, filename); toast.success(`Opening "${filename}"...`); }
    catch { toast.error("Failed to open PDF"); }
  };

  const handleViewPdfAtPage = (collection: string, filename: string, pages: string) => {
    try {
      const clean = pages.replace(/[\[\]']/g, "");
      const first = clean.split(",")[0]?.trim();
      if (first) {
        const base = collectionsApi.getPDFUrl(collection, filename);
        window.open(`${base}#page=${first}`, "_blank", "noopener,noreferrer");
        toast.success(`Opening at page ${first}...`);
      } else {
        handleViewPdf(collection, filename);
      }
    } catch { toast.error("Failed to open PDF at page"); }
  };

  return (
    <div
      className="mt-4 pt-3 border-t"
      style={{ borderColor: "var(--border-soft)" }}
    >
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 text-[12px] transition-colors"
        style={{ color: "var(--text-muted)" }}
        onMouseEnter={e => (e.currentTarget.style.color = "var(--text-secondary)")}
        onMouseLeave={e => (e.currentTarget.style.color = "var(--text-muted)")}
      >
        <DocumentTextIcon className="h-3.5 w-3.5" />
        <span className="font-medium">
          Sources ({Object.keys(groupedSources).length} document{Object.keys(groupedSources).length !== 1 ? "s" : ""})
        </span>
        {isExpanded ? <ChevronUpIcon className="h-3.5 w-3.5" /> : <ChevronDownIcon className="h-3.5 w-3.5" />}
      </button>

      {isExpanded && (
        <div className="mt-3 space-y-2.5 max-h-96 overflow-y-auto">
          {Object.entries(groupedSources).map(([filename, fileSources]) => {
            const isDocExpanded = expandedDocuments.has(filename);
            const displayed = isDocExpanded ? fileSources : fileSources.slice(0, 3);
            const collection = fileSources[0].collection || "";

            return (
              <div
                key={filename}
                className="rounded-xl p-3"
                style={{ background: "var(--bg-surface)", border: "0.5px solid var(--border-soft)" }}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex flex-col gap-1 flex-1 min-w-0">
                    <button
                      onClick={() => handleViewPdf(collection, filename)}
                      className="text-[12px] font-medium flex items-center gap-1 group text-left"
                      style={{ color: "var(--text-accent)" }}
                    >
                      <span className="group-hover:underline truncate">{filename}</span>
                      <ArrowTopRightOnSquareIcon className="h-3 w-3 flex-shrink-0" />
                    </button>
                    {collection && (
                      <div className="flex items-center gap-1 text-[10px]" style={{ color: "var(--text-muted)" }}>
                        <FolderIcon className="h-3 w-3" />
                        <span>{collection}</span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="space-y-2">
                  {displayed.map((source, idx) => (
                    <div key={idx} className="text-[11px]">
                      <div className="flex items-center justify-between mb-1">
                        <span style={{ color: "var(--text-muted)" }}>
                          Match: {(source.similarity * 100).toFixed(1)}%
                        </span>
                        {source.page_numbers && source.page_numbers !== "[]" && (
                          <button
                            onClick={() => handleViewPdfAtPage(collection, filename, source.page_numbers!)}
                            className="transition-colors"
                            style={{ color: "var(--text-accent)" }}
                          >
                            Pages: {source.page_numbers.replace(/[\[\]']/g, "")}
                          </button>
                        )}
                      </div>
                      <p className="leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                        {source.content}
                      </p>
                    </div>
                  ))}

                  {fileSources.length > 3 && (
                    <button
                      onClick={() => toggleDocument(filename)}
                      className="w-full text-[11px] py-1.5 rounded-lg transition-colors flex items-center justify-center gap-1"
                      style={{ color: "var(--text-accent)" }}
                      onMouseEnter={e => (e.currentTarget.style.background = "var(--bg-base)")}
                      onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                    >
                      {isDocExpanded ? (
                        <><ChevronUpIcon className="h-3 w-3" /> Show less</>
                      ) : (
                        `+${fileSources.length - 3} more excerpts`
                      )}
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}