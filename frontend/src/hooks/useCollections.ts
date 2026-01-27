// src/hooks/useCollections.ts
"use client";
import { useState, useEffect, useCallback, useRef } from "react";
import { collectionsApi } from "@/lib/api/collections";
import { Collection } from "@/lib/types/collection";
import { useToast } from "./useToast";

export function useCollections() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const toast = useToast();

  const toastRef = useRef(toast);

  useEffect(() => {
    toastRef.current = toast;
  }, [toast]);

  const fetchCollections = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await collectionsApi.getAll();
      setCollections(data);
    } catch (err: any) {
      const errorMsg = err.message || "Failed to fetch collections";
      setError(errorMsg);
      toastRef.current.error(errorMsg);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const deleteCollection = useCallback(
    async (name: string) => {
      try {
        await collectionsApi.delete(name);
        toastRef.current.success(`Collection "${name}" deleted`);
        await fetchCollections();
      } catch (err: any) {
        toastRef.current.error(err.message || "Failed to delete collection");
        throw err;
      }
    },
    [fetchCollections]
  );

  const renameCollection = useCallback(
    async (oldName: string, newName: string) => {
      try {
        await collectionsApi.rename(oldName, newName);
        toastRef.current.success(`Collection renamed to "${newName}"`);
        await fetchCollections();
      } catch (err: any) {
        toastRef.current.error(err.message || "Failed to rename collection");
        throw err;
      }
    },
    [fetchCollections]
  );

  const uploadFiles = useCallback(
    async (collectionName: string, files: File[]) => {
      try {
        const result = await collectionsApi.upload(collectionName, files);
        toastRef.current.success(
          `Uploaded ${result.result?.files_processed || 0} file(s) successfully`
        );
        await fetchCollections();
        return result;
      } catch (err: any) {
        toastRef.current.error(err.message || "Failed to upload files");
        throw err;
      }
    },
    [fetchCollections]
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
