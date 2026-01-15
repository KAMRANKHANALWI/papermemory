// src/components/Modals/PDFListModal.tsx
"use client";

import { useEffect, useState, useMemo } from "react";
import Modal from "../UI/Modal";
import Badge from "../UI/Badge";
import { useToast } from "@/hooks/useToast";
import { collectionsApi } from "@/lib/api/collections";
import { PDFDetail } from "@/lib/types/collection";
import { 
  DocumentTextIcon, 
  MagnifyingGlassIcon, 
  TrashIcon,
  EyeIcon 
} from "@heroicons/react/24/outline";

interface PDFListModalProps {
  isOpen: boolean;
  onClose: () => void;
  collectionName: string;
  onPdfDeleted?: () => void;
}

export default function PDFListModal({
  isOpen,
  onClose,
  collectionName,
  onPdfDeleted,
}: PDFListModalProps) {
  const [pdfs, setPdfs] = useState<PDFDetail[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [deletingPdf, setDeletingPdf] = useState<string | null>(null);
  const toast = useToast();

  useEffect(() => {
    if (isOpen && collectionName) {
      fetchPDFs();
    }
    if (!isOpen) {
      setSearchQuery("");
    }
  }, [isOpen, collectionName]);

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

  const filteredPdfs = useMemo(() => {
    if (!searchQuery.trim()) return pdfs;
    const query = searchQuery.toLowerCase();
    return pdfs.filter(
      (pdf) =>
        pdf.filename.toLowerCase().includes(query) ||
        pdf.title.toLowerCase().includes(query)
    );
  }, [pdfs, searchQuery]);

  // ✅ NEW: Handle view PDF
  const handleViewPdf = (pdfFilename: string) => {
    try {
      collectionsApi.viewPDF(collectionName, pdfFilename);
      toast.success(`Opening "${pdfFilename}"...`);
    } catch (error) {
      console.error("Failed to open PDF:", error);
      toast.error("Failed to open PDF");
    }
  };

  const handleDeletePdf = async (pdfFilename: string) => {
    const confirmed = window.confirm(
      `Are you sure you want to delete "${pdfFilename}"?\n\nThis will permanently delete:\n• The PDF file\n• All associated chunks and embeddings\n\nThis action cannot be undone!`
    );

    if (!confirmed) return;

    setDeletingPdf(pdfFilename);
    try {
      const result = await collectionsApi.deletePDF(collectionName, pdfFilename);
      setPdfs((prevPdfs) => prevPdfs.filter((pdf) => pdf.filename !== pdfFilename));
      toast.success(result.message || `Deleted "${pdfFilename}"`);
      if (onPdfDeleted) {
        onPdfDeleted();
      }
    } catch (error: any) {
      console.error("Delete failed:", error);
      toast.error(error.message || "Failed to delete PDF");
    } finally {
      setDeletingPdf(null);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`PDFs in ${collectionName}`} size="lg">
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <div className="flex flex-col items-center space-y-3">
            <svg className="animate-spin h-8 w-8 text-amber-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <p className="text-sm text-gray-600">Loading PDFs...</p>
          </div>
        </div>
      ) : (
        <>
          <div className="space-y-3 mb-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Badge variant="info" size="md">
                  {pdfs.length} {pdfs.length === 1 ? "PDF" : "PDFs"}
                </Badge>
                {searchQuery && filteredPdfs.length !== pdfs.length && (
                  <Badge variant="default" size="md">
                    {filteredPdfs.length} filtered
                  </Badge>
                )}
              </div>
            </div>

            {pdfs.length > 0 && (
              <div className="relative">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400 pointer-events-none" />
                <input
                  type="text"
                  placeholder="Search PDFs by filename or title..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 text-sm bg-gray-50 border border-gray-200 rounded-lg focus:outline-none focus:bg-white focus:border-amber-500 focus:ring-2 focus:ring-amber-500/20 placeholder:text-gray-400 transition-all duration-150"
                />
                {searchQuery && (
                  <button onClick={() => setSearchQuery("")} className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600">
                    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </div>
            )}
          </div>

          {pdfs.length === 0 ? (
            <div className="text-center py-12 px-4">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gray-100 mb-3">
                <DocumentTextIcon className="h-8 w-8 text-gray-400" />
              </div>
              <p className="text-sm text-gray-600 font-medium">No PDFs found</p>
              <p className="text-xs text-gray-400 mt-1">Upload PDFs to this collection to get started</p>
            </div>
          ) : filteredPdfs.length === 0 ? (
            <div className="text-center py-12 px-4">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gray-100 mb-3">
                <MagnifyingGlassIcon className="h-8 w-8 text-gray-400" />
              </div>
              <p className="text-sm text-gray-600 font-medium">No PDFs match your search</p>
              <p className="text-xs text-gray-400 mt-1">Try different keywords or clear the search</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
              {filteredPdfs.map((pdf, idx) => (
                <div key={`${pdf.filename}-${idx}`} className="bg-gray-50 rounded-lg p-4 border border-gray-200 hover:border-gray-300 transition-all duration-150 group">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-start space-x-3 flex-1 min-w-0">
                      <DocumentTextIcon className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <h4 className="font-medium text-gray-900 truncate text-sm">{pdf.filename}</h4>
                        <p className="text-sm text-gray-600 mt-1 line-clamp-2">{pdf.title}</p>
                        <div className="flex items-center space-x-2 mt-2">
                          <Badge variant="info" size="sm">
                            {pdf.pages} {pdf.pages === 1 ? "page" : "pages"}
                          </Badge>
                        </div>
                      </div>
                    </div>

                    {/* ✅ NEW: Action Buttons Container */}
                    <div className="flex items-center space-x-1 flex-shrink-0">
                      {/* View PDF Button */}
                      <button
                        onClick={() => handleViewPdf(pdf.filename)}
                        className="p-2 text-blue-500 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-all duration-150"
                        title="View PDF"
                      >
                        <EyeIcon className="h-5 w-5" strokeWidth={2} />
                      </button>

                      {/* Delete Button */}
                      <button
                        onClick={() => handleDeletePdf(pdf.filename)}
                        disabled={deletingPdf === pdf.filename}
                        className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Delete PDF"
                      >
                        {deletingPdf === pdf.filename ? (
                          <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                        ) : (
                          <TrashIcon className="h-5 w-5" strokeWidth={2} />
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #f1f1f1;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #d1d5db;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #9ca3af;
        }
      `}</style>
    </Modal>
  );
}
