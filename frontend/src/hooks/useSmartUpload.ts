// src/hooks/useSmartUpload.ts
"use client";

import { useState, useCallback } from "react";
import { collectionsApi } from "@/lib/api/collections";
import { useToast } from "./useToast";

export function useSmartUpload() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [suggestedName, setSuggestedName] = useState<string>("");
  const [isValid, setIsValid] = useState<boolean>(true);
  const [validationMessage, setValidationMessage] = useState<string>("");
  const toast = useToast();

  const generateName = useCallback(
    async (files: File[]) => {
      setIsGenerating(true);
      try {
        const filenames = files.map((f) => f.name);
        const result = await collectionsApi.generateName(filenames);
        
        setSuggestedName(result.suggested_name);
        setIsValid(result.is_valid);
        setValidationMessage(result.validation_message || "");
        
        return result.suggested_name;
      } catch (error: any) {
        toast.error(error.message || "Failed to generate name");
        throw error;
      } finally {
        setIsGenerating(false);
      }
    },
    [toast]
  );

  const validateName = useCallback(
    async (name: string) => {
      try {
        const result = await collectionsApi.validateName(name);
        setIsValid(result.is_valid);
        setValidationMessage(result.message);
        return result;
      } catch (error: any) {
        toast.error(error.message || "Failed to validate name");
        return { is_valid: false, message: "Validation failed" };
      }
    },
    [toast]
  );

  const uploadWithName = useCallback(
    async (collectionName: string, files: File[]) => {
      try {
        const result = await collectionsApi.upload(collectionName, files);
        toast.success(
          `Uploaded ${result.result?.files_processed || 0} files to "${collectionName}"`
        );
        return result;
      } catch (error: any) {
        toast.error(error.message || "Failed to upload files");
        throw error;
      }
    },
    [toast]
  );

  const reset = useCallback(() => {
    setSuggestedName("");
    setIsValid(true);
    setValidationMessage("");
  }, []);

  return {
    isGenerating,
    suggestedName,
    isValid,
    validationMessage,
    generateName,
    validateName,
    uploadWithName,
    reset,
  };
}
