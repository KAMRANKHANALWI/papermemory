// src/hooks/useCollections.ts
"use client";

import { useState, useEffect, useCallback } from "react";
import { collectionsApi } from "@/lib/api/collections";
import { Collection } from "@/lib/types/collection";
import { useToast } from "./useToast";

export function useCollections() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const toast = useToast();

  const fetchCollections = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await collectionsApi.getAll();
      setCollections(data);
    } catch (err: any) {
      const errorMsg = err.message || "Failed to fetch collections";
      setError(errorMsg);
      toast.error(errorMsg);
    } finally {
      setIsLoading(false);
    }
  }, [toast]);

  const deleteCollection = useCallback(
    async (name: string) => {
      try {
        await collectionsApi.delete(name);
        toast.success(`Collection "${name}" deleted`);
        await fetchCollections();
      } catch (err: any) {
        toast.error(err.message || "Failed to delete collection");
        throw err;
      }
    },
    [fetchCollections, toast]
  );

  const renameCollection = useCallback(
    async (oldName: string, newName: string) => {
      try {
        await collectionsApi.rename(oldName, newName);
        toast.success(`Collection renamed to "${newName}"`);
        await fetchCollections();
      } catch (err: any) {
        toast.error(err.message || "Failed to rename collection");
        throw err;
      }
    },
    [fetchCollections, toast]
  );

  const uploadFiles = useCallback(
    async (collectionName: string, files: File[]) => {
      try {
        const result = await collectionsApi.upload(collectionName, files);
        toast.success(
          `Uploaded ${result.result?.files_processed || 0} file(s) successfully`
        );
        await fetchCollections();
        return result;
      } catch (err: any) {
        toast.error(err.message || "Failed to upload files");
        throw err;
      }
    },
    [fetchCollections, toast]
  );

  useEffect(() => {
    fetchCollections();
  }, [fetchCollections]);

  return {
    collections,
    isLoading,
    error,
    fetchCollections,
    deleteCollection,
    renameCollection,
    uploadFiles,
  };
}
