// src/components/Sidebar/SelectedPDFsDisplay.tsx
"use client";

import { SelectedPDF } from "@/lib/types/selection";
import { XMarkIcon, DocumentTextIcon } from "@heroicons/react/24/outline";
import Badge from "../UI/Badge";

interface SelectedPDFsDisplayProps {
  selectedPDFs: SelectedPDF[];
  stats: {
    total_selected: number;
    collections_involved: string[];
    pdfs_by_collection: Record<string, number>;
    total_chunks: number;
    total_pages: number;
  } | null;
  onRemovePDF: (filename: string, collectionName: string) => void;
  onClearAll: () => void;
}

export default function SelectedPDFsDisplay({
  selectedPDFs,
  stats,
  onRemovePDF,
  onClearAll,
}: SelectedPDFsDisplayProps) {
  if (selectedPDFs.length === 0) {
    return (
      <div className="px-3 py-4 text-center">
        <DocumentTextIcon className="h-8 w-8 text-gray-300 mx-auto mb-2" />
        <p className="text-[13px] text-gray-500 font-medium">No PDFs selected</p>
        <p className="text-[12px] text-gray-400 mt-1">
          Select PDFs from collections to search
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header with Stats */}
      <div className="flex-shrink-0 px-3 py-2 border-b border-gray-100">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Badge variant="info" size="md">
              {stats?.total_selected || 0} PDFs
            </Badge>
            <Badge variant="default" size="sm">
              {stats?.total_pages || 0} pages
            </Badge>
          </div>
          <button
            onClick={onClearAll}
            className="text-[12px] text-red-600 hover:text-red-700 font-medium transition-colors"
          >
            Clear All
          </button>
        </div>

        {/* Collections Involved */}
        {stats && stats.collections_involved.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {stats.collections_involved.map((collection) => (
              <Badge key={collection} variant="default" size="sm">
                {collection}: {stats.pdfs_by_collection[collection]}
              </Badge>
            ))}
          </div>
        )}
      </div>

      {/* Selected PDFs List */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1.5">
        {selectedPDFs.map((pdf) => (
          <div
            key={`${pdf.collection_name}:${pdf.filename}`}
            className="group bg-white border border-gray-200 rounded-lg p-2.5 hover:border-gray-300 transition-all"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <p className="text-[13px] font-medium text-gray-900 truncate">
                  {pdf.filename}
                </p>
                <p className="text-[12px] text-gray-600 line-clamp-1 mt-0.5">
                  {pdf.title}
                </p>
                <div className="flex items-center gap-1.5 mt-1.5">
                  <Badge variant="info" size="sm">
                    {pdf.collection_name}
                  </Badge>
                  <span className="text-[11px] text-gray-400">
                    {pdf.pages}p • {pdf.chunks}c
                  </span>
                </div>
              </div>

              {/* Remove Button */}
              <button
                onClick={() => onRemovePDF(pdf.filename, pdf.collection_name)}
                className="flex-shrink-0 p-1 opacity-0 group-hover:opacity-100 
                         text-gray-400 hover:text-red-600 hover:bg-red-50 
                         rounded transition-all"
                title="Remove PDF"
              >
                <XMarkIcon className="h-4 w-4" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
