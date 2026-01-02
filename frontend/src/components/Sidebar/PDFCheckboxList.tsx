// src/components/Sidebar/PDFCheckboxList.tsx
"use client";

import { useState, useEffect } from "react";
import { collectionsApi } from "@/lib/api/collections";
import { PDFDetail } from "@/lib/types/collection";
import { useToast } from "@/hooks/useToast";
import { DocumentTextIcon, MagnifyingGlassIcon } from "@heroicons/react/24/outline";
import Badge from "../UI/Badge";

interface PDFCheckboxListProps {
  collectionName: string;
  selectedPDFs: Set<string>; // Set of "collectionName:filename"
  onTogglePDF: (filename: string, collectionName: string) => void;
}

export default function PDFCheckboxList({
  collectionName,
  selectedPDFs,
  onTogglePDF,
}: PDFCheckboxListProps) {
  const [pdfs, setPdfs] = useState<PDFDetail[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const toast = useToast();

  useEffect(() => {
    fetchPDFs();
  }, [collectionName]);

  const fetchPDFs = async () => {
    setIsLoading(true);
    try {
      const data = await collectionsApi.getPDFs(collectionName);
      setPdfs(data.pdfs);
    } catch (error) {
      console.error("Failed to fetch PDFs:", error);
      toast.error("Failed to load PDFs");
    } finally {
      setIsLoading(false);
    }
  };

  const filteredPDFs = pdfs.filter((pdf) =>
    pdf.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
    pdf.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getPDFKey = (filename: string) => `${collectionName}:${filename}`;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="animate-spin h-6 w-6 border-2 border-amber-600 border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Search */}
      {pdfs.length > 0 && (
        <div className="relative px-2">
          <MagnifyingGlassIcon className="absolute left-4 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search PDFs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 text-[13px] bg-gray-50 border border-transparent rounded-md
                     focus:outline-none focus:bg-white focus:border-gray-200
                     placeholder:text-gray-400 transition-all"
          />
        </div>
      )}

      {/* PDF List */}
      <div className="max-h-[300px] overflow-y-auto px-2 space-y-1">
        {filteredPDFs.length === 0 ? (
          <div className="text-center py-6 px-4">
            <DocumentTextIcon className="h-8 w-8 text-gray-300 mx-auto mb-2" />
            <p className="text-[13px] text-gray-500">
              {searchQuery ? "No PDFs match your search" : "No PDFs in this collection"}
            </p>
          </div>
        ) : (
          filteredPDFs.map((pdf) => {
            const pdfKey = getPDFKey(pdf.filename);
            const isSelected = selectedPDFs.has(pdfKey);

            return (
              <label
                key={pdf.filename}
                className={`
                  flex items-start gap-2.5 p-2.5 rounded-lg cursor-pointer
                  transition-all duration-150
                  ${
                    isSelected
                      ? "bg-amber-50 border border-amber-200"
                      : "hover:bg-gray-50 border border-transparent"
                  }
                `}
              >
                {/* Checkbox */}
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => onTogglePDF(pdf.filename, collectionName)}
                  className="mt-0.5 h-4 w-4 rounded border-gray-300 text-amber-600 
                           focus:ring-amber-500 focus:ring-offset-0 cursor-pointer"
                />

                {/* PDF Info */}
                <div className="flex-1 min-w-0">
                  <p className="text-[13px] font-medium text-gray-900 truncate">
                    {pdf.filename}
                  </p>
                  <p className="text-[12px] text-gray-600 line-clamp-1 mt-0.5">
                    {pdf.title}
                  </p>
                  <div className="flex items-center gap-2 mt-1.5">
                    <Badge variant="info" size="sm">
                      {pdf.pages} pages
                    </Badge>
                    <Badge variant="default" size="sm">
                      {pdf.chunks} chunks
                    </Badge>
                  </div>
                </div>
              </label>
            );
          })
        )}
      </div>
    </div>
  );
}
