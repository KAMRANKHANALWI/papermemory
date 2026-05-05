// src/components/Sidebar/PDFCheckboxList.tsx
"use client";

import { useState, useEffect } from "react";
import { collectionsApi } from "@/lib/api/collections";
import { PDFDetail } from "@/lib/types/collection";
import { useToast } from "@/hooks/useToast";
import { DocumentTextIcon, MagnifyingGlassIcon } from "@heroicons/react/24/outline";

interface PDFCheckboxListProps {
  collectionName: string;
  selectedPDFs: Set<string>;
  onTogglePDF: (filename: string, collectionName: string) => void;
}

export default function PDFCheckboxList({ collectionName, selectedPDFs, onTogglePDF }: PDFCheckboxListProps) {
  const [pdfs, setPdfs] = useState<PDFDetail[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const toast = useToast();

  useEffect(() => {
    setIsLoading(true);
    collectionsApi.getPDFs(collectionName)
      .then(data => setPdfs(data.pdfs))
      .catch(() => toast.error("Failed to load PDFs"))
      .finally(() => setIsLoading(false));
  }, [collectionName]);

  const filtered = pdfs.filter(p =>
    p.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
    p.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-6">
        <div
          className="animate-spin h-5 w-5 rounded-full border-2 border-t-transparent"
          style={{ borderColor: "var(--accent) transparent var(--accent) var(--accent)" }}
        />
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      {pdfs.length > 0 && (
        <div className="relative px-1">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 pointer-events-none" style={{ color: "var(--text-muted)" }} />
          <input
            type="text"
            placeholder="Search PDFs..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="w-full pl-7 pr-3 py-1.5 text-[12px] rounded-md border outline-none transition-all"
            style={{
              background: "var(--bg-input)",
              borderColor: "var(--border-soft)",
              color: "var(--text-primary)",
            }}
          />
        </div>
      )}

      <div className="max-h-[280px] overflow-y-auto px-1 space-y-0.5">
        {filtered.length === 0 ? (
          <div className="text-center py-5">
            <DocumentTextIcon className="h-7 w-7 mx-auto mb-1.5" style={{ color: "var(--text-muted)" }} />
            <p className="text-[12px]" style={{ color: "var(--text-muted)" }}>
              {searchQuery ? "No PDFs match search" : "No PDFs in collection"}
            </p>
          </div>
        ) : filtered.map(pdf => {
          const key = `${collectionName}:${pdf.filename}`;
          const isSelected = selectedPDFs.has(key);

          return (
            <label
              key={pdf.filename}
              className="flex items-start gap-2.5 p-2 rounded-lg cursor-pointer transition-all duration-150"
              style={{
                background: isSelected ? "var(--accent-light)" : "transparent",
                border: isSelected ? "0.5px solid var(--accent)" : "0.5px solid transparent",
              }}
              onMouseEnter={e => {
                if (!isSelected) e.currentTarget.style.background = "var(--bg-surface)";
              }}
              onMouseLeave={e => {
                if (!isSelected) e.currentTarget.style.background = "transparent";
              }}
            >
              <input
                type="checkbox"
                checked={isSelected}
                onChange={() => onTogglePDF(pdf.filename, collectionName)}
                className="mt-0.5 h-3.5 w-3.5 rounded cursor-pointer"
                style={{ accentColor: "var(--accent)" }}
              />
              <div className="flex-1 min-w-0">
                <p className="text-[12px] font-medium truncate" style={{ color: "var(--text-primary)" }}>
                  {pdf.filename}
                </p>
                <p className="text-[11px] line-clamp-1 mt-0.5" style={{ color: "var(--text-muted)" }}>
                  {pdf.title}
                </p>
                <span
                  className="inline-block mt-1 text-[10px] px-1.5 py-0.5 rounded"
                  style={{ background: "var(--bg-surface)", color: "var(--text-muted)" }}
                >
                  {pdf.pages} pages
                </span>
              </div>
            </label>
          );
        })}
      </div>
    </div>
  );
}