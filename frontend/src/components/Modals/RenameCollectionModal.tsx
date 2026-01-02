// src/components/Modals/RenameCollectionModal.tsx
"use client";

import { useState } from "react";
import Modal from "../UI/Modal";
import Button from "../UI/Button";
import Input from "../UI/Input";
import { useCollections } from "@/hooks/useCollections";
import { useToast } from "@/hooks/useToast";

interface RenameCollectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  collectionName: string;
}

export default function RenameCollectionModal({
  isOpen,
  onClose,
  collectionName,
}: RenameCollectionModalProps) {
  const [newName, setNewName] = useState(collectionName);
  const [error, setError] = useState("");
  const [isRenaming, setIsRenaming] = useState(false);
  const { renameCollection } = useCollections();
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

    setIsRenaming(true);
    try {
      await renameCollection(collectionName, newName);
      toast.success(`Renamed to "${newName}"`);
      onClose();
    } catch (error: any) {
      console.error("Rename failed:", error);
      toast.error(error.message || "Failed to rename collection");
    } finally {
      setIsRenaming(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !error && newName && !isRenaming) {
      e.preventDefault();
      handleRename();
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Rename Collection" size="md">
      <div className="space-y-4">
        <Input
          label="Collection Name"
          value={newName}
          onChange={(e) => handleNameChange(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Enter new name..."
          error={error}
          helperText={
            !error
              ? "3-63 characters • Letters, numbers, dots, and dashes • Press Enter to save"
              : undefined
          }
          autoFocus
          disabled={isRenaming}
        />

        <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
          <Button variant="secondary" onClick={onClose} disabled={isRenaming}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={handleRename}
            disabled={!newName || !!error || isRenaming}
            isLoading={isRenaming}
          >
            Rename
          </Button>
        </div>
      </div>
    </Modal>
  );
}
