// src/components/Sidebar/SelectedPDFsDisplay.tsx
"use client";

import { useState } from "react";
import { ChevronDownIcon, ChevronUpIcon, XMarkIcon } from "@heroicons/react/24/outline";

interface SelectedPDF {
  filename: string;
  collection_name: string;
  title?: string;
  page_count?: number;
}

interface SelectedPDFsDisplayProps {
  selectedPDFs: SelectedPDF[];
  stats: any;
  onRemovePDF: (filename: string, collectionName: string) => void;
  onClearAll: () => void;
}

export default function SelectedPDFsDisplay({ selectedPDFs, stats, onRemovePDF, onClearAll }: SelectedPDFsDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const totalSelected = stats?.total_selected || selectedPDFs.length;
  const totalPages = stats?.total_pages || 0;

  const collectionSummary = selectedPDFs.reduce((acc, pdf) => {
    acc[pdf.collection_name] = (acc[pdf.collection_name] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="flex flex-col" style={{ background: "var(--bg-base)" }}>
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between px-4 py-3 transition-colors"
        style={{ borderBottom: isExpanded ? `0.5px solid var(--border-soft)` : "none" }}
        onMouseEnter={e => (e.currentTarget.style.background = "var(--bg-surface)")}
        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
      >
        <div className="flex items-center gap-2.5 flex-1 min-w-0">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ background: "var(--accent-light)" }}
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2} style={{ color: "var(--accent)" }}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div className="flex-1 min-w-0 text-left">
            <div className="flex items-center gap-1.5">
              <span className="text-[13px] font-medium" style={{ color: "var(--text-primary)" }}>
                {totalSelected} PDF{totalSelected !== 1 ? "s" : ""}
              </span>
              <span style={{ color: "var(--text-muted)", fontSize: 10 }}>·</span>
              <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                {totalPages} pages
              </span>
            </div>
            {!isExpanded && (
              <div className="text-[11px] truncate mt-0.5" style={{ color: "var(--text-muted)" }}>
                {Object.entries(collectionSummary).map(([c, n]) => `${c}: ${n}`).join(", ")}
              </div>
            )}
          </div>
          {isExpanded ? (
            <ChevronUpIcon className="w-4 h-4 flex-shrink-0" style={{ color: "var(--text-muted)" }} />
          ) : (
            <ChevronDownIcon className="w-4 h-4 flex-shrink-0" style={{ color: "var(--text-muted)" }} />
          )}
        </div>
      </button>

      {isExpanded && (
        <div className="flex-1 overflow-y-auto min-h-0 max-h-48">
          {selectedPDFs.length > 0 && (
            <div className="px-4 py-2" style={{ borderBottom: `0.5px solid var(--border-muted)` }}>
              <button
                onClick={onClearAll}
                className="text-[11px] font-medium transition-colors"
                style={{ color: "var(--text-danger)" }}
              >
                Clear All
              </button>
            </div>
          )}

          <div className="px-2 py-2 space-y-0.5">
            {selectedPDFs.length === 0 ? (
              <div className="text-center py-6 text-[12px]" style={{ color: "var(--text-muted)" }}>
                No PDFs selected
              </div>
            ) : selectedPDFs.map((pdf, i) => (
              <div
                key={`${pdf.collection_name}-${pdf.filename}-${i}`}
                className="group flex items-start gap-2 p-2 rounded-lg transition-colors"
                onMouseEnter={e => (e.currentTarget.style.background = "var(--bg-surface)")}
                onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
              >
                <svg className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20" style={{ color: "var(--text-danger)" }}>
                  <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                </svg>
                <div className="flex-1 min-w-0">
                  <div className="text-[11px] font-medium truncate" style={{ color: "var(--text-primary)" }}>
                    {pdf.title || pdf.filename}
                  </div>
                  <span
                    className="inline-block mt-0.5 text-[10px] px-1.5 py-0.5 rounded"
                    style={{ background: "var(--accent-light)", color: "var(--accent)" }}
                  >
                    {pdf.collection_name}
                  </span>
                </div>
                <button
                  onClick={() => onRemovePDF(pdf.filename, pdf.collection_name)}
                  className="opacity-0 group-hover:opacity-100 flex-shrink-0 transition-opacity"
                >
                  <XMarkIcon className="w-3.5 h-3.5" style={{ color: "var(--text-muted)" }} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}