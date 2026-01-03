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
  stats: {
    total_selected?: number;
    total_pages?: number;
    total_chunks?: number;
    collections?: Record<string, { count: number; pages: number; chunks?: number }>;
  } | null | undefined;
  onRemovePDF: (filename: string, collectionName: string) => void;
  onClearAll: () => void;
}

export default function SelectedPDFsDisplay({
  selectedPDFs,
  stats,
  onRemovePDF,
  onClearAll,
}: SelectedPDFsDisplayProps) {
  // State to control collapse/expand
  const [isExpanded, setIsExpanded] = useState(false);

  const totalSelected = stats?.total_selected || selectedPDFs.length;
  const totalPages = stats?.total_pages || 0;

  // Group PDFs by collection for summary
  const collectionSummary = selectedPDFs.reduce((acc, pdf) => {
    const col = pdf.collection_name;
    acc[col] = (acc[col] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header - Always Visible (Collapsible) */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition-colors cursor-pointer border-b border-gray-100"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          {/* Icon */}
          <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
            <svg
              className="w-4 h-4 text-blue-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
          </div>

          {/* Summary Text */}
          <div className="flex-1 min-w-0 text-left">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-gray-900">
                {totalSelected} PDF{totalSelected !== 1 ? "s" : ""}
              </span>
              <span className="text-xs text-gray-400">•</span>
              <span className="text-xs text-gray-500">
                {totalPages} page{totalPages !== 1 ? "s" : ""}
              </span>
            </div>
            
            {/* Collections Summary (only when collapsed) */}
            {!isExpanded && (
              <div className="text-xs text-gray-400 truncate mt-0.5">
                {Object.entries(collectionSummary)
                  .map(([col, count]) => `${col}: ${count}`)
                  .join(", ")}
              </div>
            )}
          </div>

          {/* Expand/Collapse Icon */}
          <div className="flex-shrink-0">
            {isExpanded ? (
              <ChevronUpIcon className="w-5 h-5 text-gray-400" />
            ) : (
              <ChevronDownIcon className="w-5 h-5 text-gray-400" />
            )}
          </div>
        </div>
      </button>

      {/* Expanded Content - Scrollable List */}
      {isExpanded && (
        <div className="flex-1 overflow-y-auto min-h-0">
          {/* Clear All Button */}
          {selectedPDFs.length > 0 && (
            <div className="px-4 py-2 border-b border-gray-100">
              <button
                onClick={onClearAll}
                className="text-xs text-red-500 hover:text-red-700 font-medium transition-colors"
              >
                Clear All
              </button>
            </div>
          )}

          {/* PDF List */}
          <div className="px-2 py-2 space-y-1">
            {selectedPDFs.length === 0 ? (
              <div className="text-center py-8 text-gray-400 text-sm">
                No PDFs selected
              </div>
            ) : (
              selectedPDFs.map((pdf, index) => (
                <div
                  key={`${pdf.collection_name}-${pdf.filename}-${index}`}
                  className="group flex items-start gap-2 p-2 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  {/* PDF Icon */}
                  <div className="flex-shrink-0 mt-0.5">
                    <svg
                      className="w-4 h-4 text-red-500"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </div>

                  {/* PDF Info */}
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium text-gray-900 truncate">
                      {pdf.title || pdf.filename}
                    </div>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-blue-50 text-blue-700">
                        {pdf.collection_name}
                      </span>
                      {pdf.page_count && pdf.page_count > 0 && (
                        <span className="text-[10px] text-gray-400">
                          {pdf.page_count}p
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Remove Button */}
                  <button
                    onClick={() =>
                      onRemovePDF(pdf.filename, pdf.collection_name)
                    }
                    className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                    title="Remove PDF"
                  >
                    <XMarkIcon className="w-4 h-4 text-gray-400 hover:text-red-500 transition-colors" />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Stats Footer (when expanded) */}
      {isExpanded && selectedPDFs.length > 0 && (
        <div className="flex-shrink-0 px-4 py-2 border-t border-gray-100 bg-gray-50">
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-500">
              {Object.keys(collectionSummary).length} collection
              {Object.keys(collectionSummary).length !== 1 ? "s" : ""}
            </span>
            <span className="text-gray-500">
              {totalPages} total pages
            </span>
          </div>
        </div>
      )}
    </div>
  );
}