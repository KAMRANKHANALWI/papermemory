// src/components/Modals/CollectionManageModal.tsx
"use client";

import { useState } from "react";
import Modal from "../UI/Modal";
import Button from "../UI/Button";
import Input from "../UI/Input";
import { useCollections } from "@/hooks/useCollections";
import { useToast } from "@/hooks/useToast";

interface CollectionManageModalProps {
  isOpen: boolean;
  onClose: () => void;
  collectionName: string;
  onViewPDFs: () => void;
  onAddPDFs: () => void;
}

export default function CollectionManageModal({
  isOpen,
  onClose,
  collectionName,
  onViewPDFs,
  onAddPDFs,
}: CollectionManageModalProps) {
  const [isRenaming, setIsRenaming] = useState(false);
  const [newName, setNewName] = useState("");
  const { renameCollection } = useCollections();
  const toast = useToast();

  const handleRename = async () => {
    if (!newName) return;

    try {
      await renameCollection(collectionName, newName);
      setIsRenaming(false);
      setNewName("");
      onClose();
    } catch (error) {
      console.error("Rename failed:", error);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={`Manage: ${collectionName}`}
      size="md"
    >
      <div className="space-y-4">
        {/* Rename Section */}
        {isRenaming ? (
          <div className="space-y-3">
            <Input
              label="New Collection Name"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Enter new name..."
            />
            <div className="flex space-x-2">
              <Button
                variant="primary"
                onClick={handleRename}
                disabled={!newName}
              >
                Save
              </Button>
              <Button variant="secondary" onClick={() => setIsRenaming(false)}>
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <Button
            variant="ghost"
            onClick={() => setIsRenaming(true)}
            className="w-full"
          >
            Rename Collection
          </Button>
        )}

        {/* Actions */}
        <div className="space-y-2">
          <Button variant="secondary" onClick={onViewPDFs} className="w-full">
            View PDFs in Collection
          </Button>
          <Button variant="secondary" onClick={onAddPDFs} className="w-full">
            Add More PDFs
          </Button>
        </div>
      </div>
    </Modal>
  );
}
