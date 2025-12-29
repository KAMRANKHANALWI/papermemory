// src/components/Modals/PDFListModal.tsx
"use client";

import { useEffect, useState } from "react";
import Modal from "../UI/Modal";
import Badge from "../UI/Badge";
import { collectionsApi } from "@/lib/api/collections";
import { PDFDetail } from "@/lib/types/collection";
import { DocumentTextIcon } from "@heroicons/react/24/outline";

interface PDFListModalProps {
  isOpen: boolean;
  onClose: () => void;
  collectionName: string;
}

export default function PDFListModal({
  isOpen,
  onClose,
  collectionName,
}: PDFListModalProps) {
  const [pdfs, setPdfs] = useState<PDFDetail[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (isOpen && collectionName) {
      setIsLoading(true);
      collectionsApi
        .getPDFs(collectionName)
        .then((data) => setPdfs(data.pdfs))
        .catch(console.error)
        .finally(() => setIsLoading(false));
    }
  }, [isOpen, collectionName]);

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`PDFs in ${collectionName}`} size="lg">
      {isLoading ? (
        <div className="text-center py-8">Loading...</div>
      ) : pdfs.length === 0 ? (
        <div className="text-center py-8 text-gray-500">No PDFs found</div>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {pdfs.map((pdf, idx) => (
            <div key={idx} className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3 flex-1">
                  <DocumentTextIcon className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
                  <div className="flex-1 min-w-0">
                    <h4 className="font-medium text-gray-900 truncate">{pdf.filename}</h4>
                    <p className="text-sm text-gray-600 mt-1">{pdf.title}</p>
                    <div className="flex items-center space-x-2 mt-2">
                      <Badge variant="info" size="sm">
                        {pdf.pages} pages
                      </Badge>
                      <Badge variant="default" size="sm">
                        {pdf.chunks} chunks
                      </Badge>
                      <span className="text-xs text-gray-500">Range: {pdf.page_range}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Modal>
  );
}
