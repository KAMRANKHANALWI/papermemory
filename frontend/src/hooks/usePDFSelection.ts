// src/hooks/usePDFSelection.ts
"use client";

import { useState, useCallback, useEffect } from "react";
import { selectionApi } from "@/lib/api/selection";
import type {
  SelectedPDF,
  SelectionStatsResponse,
} from "@/lib/types/selection";
import { useToast } from "./useToast";

export function usePDFSelection(sessionId: string) {
  const [selectedPDFs, setSelectedPDFs] = useState<SelectedPDF[]>([]);
  const [stats, setStats] = useState<SelectionStatsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();

  // Fetch current selection
  const fetchSelection = useCallback(async () => {
    if (!sessionId) return;
    
    setIsLoading(true);
    try {
      const data = await selectionApi.getSelection(sessionId);
      setSelectedPDFs(data.selected_pdfs);
      
      // Also fetch stats
      const statsData = await selectionApi.getStats(sessionId);
      setStats(statsData);
    } catch (error: any) {
      // Silent fail if no selection exists yet
      if (!error.message?.includes("404")) {
        console.error("Failed to fetch selection:", error);
      }
      setSelectedPDFs([]);
      setStats(null);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  // Select a PDF
  const selectPDF = useCallback(
    async (filename: string, collectionName: string) => {
      try {
        const result = await selectionApi.selectPDF(
          sessionId,
          filename,
          collectionName
        );
        setSelectedPDFs(result.selected_pdfs);
        toast.success(`Selected "${filename}"`);
        await fetchSelection(); // Refresh stats
        return true;
      } catch (error: any) {
        toast.error(error.message || "Failed to select PDF");
        return false;
      }
    },
    [sessionId, toast, fetchSelection]
  );

  // Deselect a PDF
  const deselectPDF = useCallback(
    async (filename: string, collectionName: string) => {
      try {
        const result = await selectionApi.deselectPDF(
          sessionId,
          filename,
          collectionName
        );
        setSelectedPDFs(result.selected_pdfs);
        toast.success(`Removed "${filename}"`);
        await fetchSelection(); // Refresh stats
        return true;
      } catch (error: any) {
        toast.error(error.message || "Failed to deselect PDF");
        return false;
      }
    },
    [sessionId, toast, fetchSelection]
  );

  // Toggle PDF selection
  const togglePDF = useCallback(
    async (filename: string, collectionName: string) => {
      const isSelected = selectedPDFs.some(
        (pdf) =>
          pdf.filename === filename && pdf.collection_name === collectionName
      );

      if (isSelected) {
        return deselectPDF(filename, collectionName);
      } else {
        return selectPDF(filename, collectionName);
      }
    },
    [selectedPDFs, selectPDF, deselectPDF]
  );

  // Check if a PDF is selected
  const isPDFSelected = useCallback(
    (filename: string, collectionName: string): boolean => {
      return selectedPDFs.some(
        (pdf) =>
          pdf.filename === filename && pdf.collection_name === collectionName
      );
    },
    [selectedPDFs]
  );

  // Batch select PDFs
  const batchSelectPDFs = useCallback(
    async (pdfs: Array<{ filename: string; collection_name: string }>) => {
      try {
        const result = await selectionApi.batchSelectPDFs(sessionId, pdfs);
        setSelectedPDFs(result.selected_pdfs);
        toast.success(`Selected ${pdfs.length} PDFs`);
        await fetchSelection(); // Refresh stats
        return true;
      } catch (error: any) {
        toast.error(error.message || "Failed to batch select PDFs");
        return false;
      }
    },
    [sessionId, toast, fetchSelection]
  );

  // Clear all selections
  const clearSelection = useCallback(async () => {
    try {
      await selectionApi.clearSelection(sessionId);
      setSelectedPDFs([]);
      setStats(null);
      toast.success("Cleared all selections");
      return true;
    } catch (error: any) {
      toast.error(error.message || "Failed to clear selection");
      return false;
    }
  }, [sessionId, toast]);

  // Search within selected PDFs
  const searchSelected = useCallback(
    async (query: string, numResults: number = 25) => {
      try {
        const result = await selectionApi.searchSelected(
          sessionId,
          query,
          numResults
        );
        return result;
      } catch (error: any) {
        toast.error(error.message || "Failed to search");
        throw error;
      }
    },
    [sessionId, toast]
  );

  // Load selection on mount
  useEffect(() => {
    fetchSelection();
  }, [fetchSelection]);

  return {
    selectedPDFs,
    stats,
    isLoading,
    selectPDF,
    deselectPDF,
    togglePDF,
    isPDFSelected,
    batchSelectPDFs,
    clearSelection,
    searchSelected,
    fetchSelection,
  };
}
