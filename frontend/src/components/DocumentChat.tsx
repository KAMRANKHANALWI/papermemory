// src/components/DocumentChat.tsx
"use client";

import { useState } from "react";
import { ToastProvider } from "@/contexts/ToastContext";
import ToastContainer from "./UI/ToastContainer";
import Sidebar from "./Sidebar/Sidebar";
import ChatArea from "./Chat/ChatArea";
import SmartUploadModal from "./Modals/SmartUploadModal";
import CollectionManageModal from "./Modals/CollectionManageModal";
import PDFListModal from "./Modals/PDFListModal";
import { useCollections } from "@/hooks/useCollections";
import { useChat } from "@/hooks/useChat";
import { useToast } from "@/hooks/useToast";

function DocumentChatContent() {
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null);
  const [chatMode, setChatMode] = useState<"single" | "chatall">("single");
  
  // Modal states
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [manageModalOpen, setManageModalOpen] = useState(false);
  const [pdfListModalOpen, setPdfListModalOpen] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [managingCollection, setManagingCollection] = useState<string | null>(null);

  const {
    collections,
    isLoading: collectionsLoading,
    deleteCollection,
    fetchCollections,
  } = useCollections();

  const { messages, isLoading: chatLoading, sendMessage, clearMessages } = useChat();
  const toast = useToast();

  // Handle collection selection
  const handleSelectCollection = (name: string) => {
    setSelectedCollection(name);
    if (chatMode === "chatall") {
      setChatMode("single");
    }
  };

  // Handle chat mode change
  const handleChatModeChange = (mode: "single" | "chatall") => {
    setChatMode(mode);
    if (mode === "chatall") {
      setSelectedCollection(null);
    }
  };

  // Handle file upload
  const handleFilesSelected = (files: File[]) => {
    setPendingFiles(files);
    setUploadModalOpen(true);
  };

  const handleUploadComplete = () => {
    setPendingFiles([]);
    fetchCollections();
  };

  // Handle collection deletion
  const handleDeleteCollection = async (name: string) => {
    if (confirm(`Delete collection "${name}"? This cannot be undone.`)) {
      try {
        await deleteCollection(name);
        if (selectedCollection === name) {
          setSelectedCollection(null);
        }
      } catch (error) {
        console.error("Delete failed:", error);
      }
    }
  };

  // Handle collection management
  const handleManageCollection = (name: string) => {
    setManagingCollection(name);
    setManageModalOpen(true);
  };

  const handleViewPDFs = () => {
    if (managingCollection) {
      setManageModalOpen(false);
      setPdfListModalOpen(true);
    }
  };

  const handleAddPDFsToCollection = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ".pdf";
    input.onchange = (e: any) => {
      const files = Array.from(e.target.files || []) as File[];
      if (files.length > 0 && managingCollection) {
        // Upload directly to the collection
        import("@/lib/api/collections").then(({ collectionsApi }) => {
          collectionsApi
            .addPDFs(managingCollection, files)
            .then(() => {
              toast.success(`Added ${files.length} files to ${managingCollection}`);
              fetchCollections();
              setManageModalOpen(false);
            })
            .catch((error) => {
              toast.error(error.message || "Failed to add files");
            });
        });
      }
    };
    input.click();
  };

  // Handle message send
  const handleSendMessage = (message: string) => {
    sendMessage(message, selectedCollection, chatMode);
  };

  return (
    <div className="h-screen flex bg-gray-50 overflow-hidden">
      {/* Sidebar with Toggle */}
      <Sidebar
        collections={collections}
        selectedCollection={selectedCollection}
        chatMode={chatMode}
        onSelectCollection={handleSelectCollection}
        onDeleteCollection={handleDeleteCollection}
        onManageCollection={handleManageCollection}
        onUploadClick={handleFilesSelected}
        onChatModeChange={handleChatModeChange}
        onClearChat={clearMessages}
      />

      {/* Chat Area - Takes remaining space */}
      <div className="flex-1 flex flex-col min-w-0">
        <ChatArea
          messages={messages}
          isLoading={chatLoading}
          selectedCollection={selectedCollection}
          chatMode={chatMode}
          onSendMessage={handleSendMessage}
        />
      </div>

      {/* Modals */}
      <SmartUploadModal
        isOpen={uploadModalOpen}
        onClose={() => setUploadModalOpen(false)}
        files={pendingFiles}
        onUploadComplete={handleUploadComplete}
      />

      {managingCollection && (
        <>
          <CollectionManageModal
            isOpen={manageModalOpen}
            onClose={() => setManageModalOpen(false)}
            collectionName={managingCollection}
            onViewPDFs={handleViewPDFs}
            onAddPDFs={handleAddPDFsToCollection}
          />

          <PDFListModal
            isOpen={pdfListModalOpen}
            onClose={() => setPdfListModalOpen(false)}
            collectionName={managingCollection}
          />
        </>
      )}

      {/* Toast Container */}
      <ToastContainer />
    </div>
  );
}

// Wrap with ToastProvider
export default function DocumentChat() {
  return (
    <ToastProvider>
      <DocumentChatContent />
    </ToastProvider>
  );
}
