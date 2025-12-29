// src/components/Modals/SmartUploadModal.tsx
"use client";

import { useState, useEffect } from "react";
import Modal from "../UI/Modal";
import Button from "../UI/Button";
import Input from "../UI/Input";
import { useSmartUpload } from "@/hooks/useSmartUpload";
import { SparklesIcon, ArrowPathIcon } from "@heroicons/react/24/outline";

interface SmartUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  files: File[];
  onUploadComplete: () => void;
}

export default function SmartUploadModal({
  isOpen,
  onClose,
  files,
  onUploadComplete,
}: SmartUploadModalProps) {
  const [collectionName, setCollectionName] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const {
    isGenerating,
    suggestedName,
    isValid,
    validationMessage,
    generateName,
    validateName,
    uploadWithName,
    reset,
  } = useSmartUpload();

  useEffect(() => {
    if (isOpen && files.length > 0) {
      generateName(files).then((name) => setCollectionName(name));
    } else {
      reset();
      setCollectionName("");
    }
  }, [isOpen, files, generateName, reset]);

  const handleNameChange = async (name: string) => {
    setCollectionName(name);
    if (name) {
      await validateName(name);
    }
  };

  const handleUpload = async () => {
    if (!collectionName || !isValid) return;

    setIsUploading(true);
    try {
      await uploadWithName(collectionName, files);
      onUploadComplete();
      onClose();
    } catch (error) {
      console.error("Upload failed:", error);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Upload Documents" size="md">
      <div className="space-y-4">
        {/* Selected Files */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Selected Files ({files.length})
          </label>
          <div className="bg-gray-50 rounded-lg p-3 max-h-32 overflow-y-auto">
            {files.map((file, idx) => (
              <div key={idx} className="text-sm text-gray-600 py-1">
                {file.name}
              </div>
            ))}
          </div>
        </div>

        {/* Collection Name */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="block text-sm font-medium text-gray-700">
              Collection Name
            </label>
            <button
              onClick={() =>
                files.length > 0 && generateName(files).then(setCollectionName)
              }
              disabled={isGenerating}
              className="text-xs text-amber-600 hover:text-amber-700 flex items-center"
            >
              {isGenerating ? (
                <>
                  <ArrowPathIcon className="h-3 w-3 mr-1 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <SparklesIcon className="h-3 w-3 mr-1" />
                  Re-generate
                </>
              )}
            </button>
          </div>
          <Input
            value={collectionName}
            onChange={(e) => handleNameChange(e.target.value)}
            placeholder="Enter collection name..."
            error={!isValid ? validationMessage : undefined}
          />
        </div>

        {/* Actions */}
        <div className="flex justify-end space-x-3 pt-4">
          <Button variant="secondary" onClick={onClose} disabled={isUploading}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={handleUpload}
            disabled={!collectionName || !isValid || isUploading}
            isLoading={isUploading}
          >
            Upload {files.length} {files.length === 1 ? "File" : "Files"}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
