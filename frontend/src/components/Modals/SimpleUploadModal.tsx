// src/components/Modals/SimpleUploadModal.tsx
"use client";

import { useState } from "react";
import Modal from "../UI/Modal";
import Button from "../UI/Button";
import Input from "../UI/Input";
import { useToast } from "@/hooks/useToast";
import { collectionsApi } from "@/lib/api/collections";

interface SimpleUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  files: File[];
  onUploadComplete: () => void;
}

export default function SimpleUploadModal({
  isOpen,
  onClose,
  files,
  onUploadComplete,
}: SimpleUploadModalProps) {
  const [collectionName, setCollectionName] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState("");
  const toast = useToast();

  const validateName = (name: string): string | null => {
    if (!name.trim()) {
      return "Collection name is required";
    }

    if (name.length < 3) {
      return "Name must be at least 3 characters";
    }

    if (name.length > 63) {
      return "Name must be less than 63 characters";
    }

    // Check valid characters: start with letter/number, then letters/numbers/dots/dashes
    if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]*$/.test(name)) {
      return "Name must start with a letter/number and contain only letters, numbers, dots, and dashes";
    }

    return null;
  };

  const handleNameChange = (name: string) => {
    setCollectionName(name);
    const validationError = validateName(name);
    setError(validationError || "");
  };

  const handleUpload = async () => {
    // Final validation
    const validationError = validateName(collectionName);
    if (validationError) {
      setError(validationError);
      toast.error(validationError);
      return;
    }

    setIsUploading(true);
    try {
      await collectionsApi.upload(collectionName, files);
      toast.success(`Uploaded ${files.length} file(s) to "${collectionName}"`);
      setCollectionName("");
      setError("");
      onUploadComplete();
      onClose();
    } catch (error: any) {
      console.error("Upload failed:", error);
      toast.error(error.message || "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !error && collectionName && !isUploading) {
      e.preventDefault();
      handleUpload();
    }
  };

  const handleClose = () => {
    if (!isUploading) {
      setCollectionName("");
      setError("");
      onClose();
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      title="Upload Documents"
      size="md"
    >
      <div className="space-y-4">
        {/* Info */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <p className="text-sm text-blue-800">
            Enter a name for your collection. This will help you organize and
            find your documents later.
          </p>
        </div>

        {/* Selected Files */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Selected Files ({files.length})
          </label>
          <div className="bg-gray-50 rounded-lg p-3 max-h-32 overflow-y-auto border border-gray-200">
            {files.map((file, idx) => (
              <div
                key={idx}
                className="text-sm text-gray-600 py-1 flex items-center"
              >
                <svg
                  className="w-4 h-4 mr-2 text-red-500"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" />
                </svg>
                {file.name}
              </div>
            ))}
          </div>
        </div>

        {/* Collection Name Input */}
        <div>
          <Input
            label="Collection Name"
            value={collectionName}
            onChange={(e) => handleNameChange(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="e.g., research-papers, aws-docs, my-notes"
            error={error}
            helperText={
              !error
                ? "3-63 characters • Letters, numbers, dots, and dashes • Press Enter to upload"
                : undefined
            }
            disabled={isUploading}
            autoFocus
          />
        </div>

        {/* Examples */}
        <div className="text-xs text-gray-500">
          <p className="font-medium mb-1">Examples:</p>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => handleNameChange("research-papers")}
              className="px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded text-gray-700 transition-colors"
              disabled={isUploading}
            >
              research-papers
            </button>
            <button
              onClick={() => handleNameChange("aws-documentation")}
              className="px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded text-gray-700 transition-colors"
              disabled={isUploading}
            >
              aws-documentation
            </button>
            <button
              onClick={() => handleNameChange("my-notes-2024")}
              className="px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded text-gray-700 transition-colors"
              disabled={isUploading}
            >
              my-notes-2024
            </button>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
          <Button
            variant="secondary"
            onClick={handleClose}
            disabled={isUploading}
          >
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={handleUpload}
            disabled={!collectionName || !!error || isUploading}
            isLoading={isUploading}
          >
            Upload {files.length} {files.length === 1 ? "File" : "Files"}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
