// src/components/Modals/CollectionManageModal.tsx
"use client";

import { useState } from "react";
import Modal from "../UI/Modal";
import Button from "../UI/Button";
import Input from "../UI/Input";
import { useCollections } from "@/hooks/useCollections";
import { useToast } from "@/hooks/useToast";
import { TrashIcon } from "@heroicons/react/24/outline";

interface CollectionManageModalProps {
  isOpen: boolean;
  onClose: () => void;
  collectionName: string;
  onViewPDFs: () => void;
  onAddPDFs: () => void;
  onDeleted?: () => void;  // Callback after collection deletion
  onPdfDeleted?: () => void;  // NEW: Callback after PDF deletion (to refresh collections)
}

export default function CollectionManageModal({
  isOpen,
  onClose,
  collectionName,
  onViewPDFs,
  onAddPDFs,
  onDeleted,
  onPdfDeleted,
}: CollectionManageModalProps) {
  const [isRenaming, setIsRenaming] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [newName, setNewName] = useState("");
  const [error, setError] = useState("");
  const { renameCollection, deleteCollection } = useCollections();
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
    
    if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]*$/.test(name)) {
      return "Name must start with a letter/number and contain only letters, numbers, dots, and dashes";
    }
    
    if (name === collectionName) {
      return "New name must be different from current name";
    }
    
    return null;
  };

  const handleNameChange = (name: string) => {
    setNewName(name);
    const validationError = validateName(name);
    setError(validationError || "");
  };

  const handleRename = async () => {
    const validationError = validateName(newName);
    if (validationError) {
      setError(validationError);
      toast.error(validationError);
      return;
    }

    try {
      await renameCollection(collectionName, newName);
      setIsRenaming(false);
      setNewName("");
      setError("");
      toast.success(`Renamed to "${newName}"`);
      onClose();
    } catch (error: any) {
      console.error("Rename failed:", error);
      toast.error(error.message || "Failed to rename collection");
    }
  };

  const handleDelete = async () => {
    // Confirmation dialog
    const confirmed = window.confirm(
      `Are you sure you want to delete "${collectionName}"?\n\nThis will permanently delete:\n• All PDFs in this collection\n• All associated chunks and embeddings\n\nThis action cannot be undone!`
    );

    if (!confirmed) return;

    setIsDeleting(true);
    try {
      await deleteCollection(collectionName);
      toast.success(`Collection "${collectionName}" deleted`);
      onClose();
      if (onDeleted) {
        onDeleted();  // Notify parent
      }
    } catch (error: any) {
      console.error("Delete failed:", error);
      toast.error(error.message || "Failed to delete collection");
    } finally {
      setIsDeleting(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !error && newName && !isDeleting) {
      e.preventDefault();
      handleRename();
    } else if (e.key === "Escape") {
      e.preventDefault();
      setIsRenaming(false);
      setNewName("");
      setError("");
    }
  };

  const startRenaming = () => {
    setIsRenaming(true);
    setNewName(collectionName);
    setError("");
  };

  const cancelRenaming = () => {
    setIsRenaming(false);
    setNewName("");
    setError("");
  };

  return (
    <Modal 
      isOpen={isOpen} 
      onClose={isDeleting ? () => {} : onClose}
      title={`${collectionName} Collection`} 
      size="md"
    >
      <div className="space-y-3">
        {/* Rename Section */}
        {isRenaming ? (
          <div className="space-y-3 pb-3 border-b border-gray-200">
            <Input
              label="New Collection Name"
              value={newName}
              onChange={(e) => handleNameChange(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Enter new name..."
              error={error}
              helperText={
                !error
                  ? "3-63 characters • Letters, numbers, dots, and dashes • Press Enter to save, Esc to cancel"
                  : undefined
              }
              autoFocus
            />
            <div className="flex space-x-2">
              <Button 
                variant="primary" 
                onClick={handleRename} 
                disabled={!newName || !!error}
                size="sm"
              >
                Save
              </Button>
              <Button variant="secondary" onClick={cancelRenaming} size="sm">
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <>
            {/* Rename Button */}
            <Button 
              variant="secondary"
              onClick={startRenaming} 
              className="w-full"
              disabled={isDeleting}
            >
              Rename
            </Button>

            {/* View PDFs Button */}
            <Button 
              variant="secondary" 
              onClick={onViewPDFs} 
              className="w-full"
              disabled={isDeleting}
            >
              View PDFs
            </Button>

            {/* Add More PDFs Button */}
            <Button 
              variant="secondary" 
              onClick={onAddPDFs} 
              className="w-full"
              disabled={isDeleting}
            >
              Add More PDFs
            </Button>

            {/* Divider */}
            <div className="border-t border-gray-200 my-2" />

            {/* Delete Button - DANGER ZONE */}
            <div className="bg-red-50 rounded-lg p-3 border border-red-200">
              <p className="text-xs text-red-600 font-medium mb-2">Danger Zone</p>
              <Button
                variant="ghost"
                onClick={handleDelete}
                disabled={isDeleting}
                isLoading={isDeleting}
                className="w-full justify-start text-red-600 hover:bg-red-100 hover:text-red-700"
              >
                <TrashIcon className="h-4 w-4 mr-2" />
                {isDeleting ? "Deleting..." : "Delete Collection"}
              </Button>
              <p className="text-xs text-gray-500 mt-2">
                This will permanently delete all PDFs and data. This action cannot be undone.
              </p>
            </div>
          </>
        )}
      </div>
    </Modal>
  );
}

