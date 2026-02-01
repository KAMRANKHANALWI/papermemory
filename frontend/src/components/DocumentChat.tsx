// src/components/DocumentChat.tsx
"use client";

import { useState, useEffect } from "react";
import { ToastProvider } from "@/contexts/ToastContext";
import ToastContainer from "./UI/ToastContainer";
import Sidebar from "./Sidebar/Sidebar";
import ChatArea from "./Chat/ChatArea";
import SmartUploadModal from "./Modals/SimpleUploadModal";
import RenameCollectionModal from "./Modals/RenameCollectionModal";
import PDFListModal from "./Modals/PDFListModal";
import { useCollections } from "@/hooks/useCollections";
import { useChat, ChatMode } from "@/hooks/useChat";
import { useToast } from "@/hooks/useToast";
import { usePDFSelection } from "@/hooks/usePDFSelection";

function DocumentChatContent() {
  const [selectedCollection, setSelectedCollection] = useState<string | null>(
    null,
  );
  const [chatMode, setChatMode] = useState<ChatMode>("single");
  const [pdfSelectionMode, setPdfSelectionMode] = useState(false);

  const sessionId = "user_session_123";

  const {
    selectedPDFs,
    stats,
    fetchSelection,
    togglePDF,
    clearSelection,
    deselectPDF,
  } = usePDFSelection(sessionId, false);

  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [renameModalOpen, setRenameModalOpen] = useState(false);
  const [pdfListModalOpen, setPdfListModalOpen] = useState(false);
  const [addPDFsModalOpen, setAddPDFsModalOpen] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [selectedCollectionForAction, setSelectedCollectionForAction] =
    useState<string | null>(null);

  const [abortController, setAbortController] =
    useState<AbortController | null>(null);

  const {
    collections,
    isLoading: collectionsLoading,
    deleteCollection,
    fetchCollections,
  } = useCollections();

  const {
    messages,
    isLoading: chatLoading,
    sendMessage,
    stopGeneration,
    clearMessages,
  } = useChat();
  const toast = useToast();

  useEffect(() => {}, [
    pdfSelectionMode,
    chatMode,
    selectedPDFs.length,
    selectedCollection,
  ]);

  // Handle PDF selection mode toggle
  const handleTogglePDFMode = () => {
    const newMode = !pdfSelectionMode;

    if (newMode) {
      setPdfSelectionMode(true);
      setSelectedCollection(null);
      // Fetch selection ONLY when entering PDF mode
      fetchSelection();
      toast.info("Select PDFs Mode");
    } else {
      console.log("   Exiting PDF selection mode");
      setPdfSelectionMode(false);
      setChatMode("single");
      toast.info("Switched to single collection mode");
    }
  };

  // Determine the display mode for ChatArea
  const getDisplayMode = (): "single" | "chatall" | "selected" => {
    if (pdfSelectionMode && selectedPDFs.length > 0) {
      return "selected";
    }
    return chatMode as "single" | "chatall";
  };

  // Handle collection selection
  const handleSelectCollection = (name: string) => {
    // Selecting a collection exits PDF selection mode
    setPdfSelectionMode(false);
    setSelectedCollection(name);
    setChatMode("single");
  };

  const handleChatModeChange = (mode: ChatMode) => {
    if (mode === "selected") {
      return;
    }

    // Exit PDF selection mode when switching to single or chatall
    if (pdfSelectionMode) {
      setPdfSelectionMode(false);
    }

    setChatMode(mode);

    if (mode === "chatall") {
      setSelectedCollection(null);
      toast.info("All Collections Mode");
    } else if (mode === "single") {
      toast.info("Single Collection Mode");
    }
  };

  // Handle file upload for new collection
  const handleFilesSelected = (files: File[]) => {
    setPendingFiles(files);
    setUploadModalOpen(true);
  };

  const handleUploadComplete = () => {
    setPendingFiles([]);
    fetchCollections();
  };

  // Callback after successful rename
  const handleRenameComplete = () => {
    fetchCollections(); // Refresh the collections list
    setRenameModalOpen(false);
    setSelectedCollectionForAction(null);
  };

  // Handle collection management
  const handleManageCollection = (name: string) => {
    setSelectedCollectionForAction(name);
  };

  // Handle rename collection
  const handleRenameCollection = (name: string) => {
    setSelectedCollectionForAction(name);
    setRenameModalOpen(true);
  };

  // Handle view PDFs
  const handleListPDFs = (name: string) => {
    setSelectedCollectionForAction(name);
    setPdfListModalOpen(true);
  };

  // Handle add PDFs to existing collection
  const handleAddPDFs = (name: string) => {
    setSelectedCollectionForAction(name);

    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ".pdf";
    input.onchange = (e: any) => {
      const files = Array.from(e.target.files || []) as File[];
      if (files.length > 0) {
        import("@/lib/api/collections").then(({ collectionsApi }) => {
          collectionsApi
            .addPDFs(name, files)
            .then(() => {
              toast.success(`Added ${files.length} file(s) to "${name}"`);
              fetchCollections();
            })
            .catch((error) => {
              toast.error(error.message || "Failed to add files");
            });
        });
      }
    };
    input.click();
  };

  // Handle delete collection
  const handleDeleteCollection = async (name: string) => {
    const confirmed = window.confirm(
      `Are you sure you want to delete "${name}"?\n\nThis will permanently delete:\n• All PDFs in this collection\n• All associated chunks and embeddings\n\nThis action cannot be undone!`,
    );

    if (!confirmed) return;

    try {
      await deleteCollection(name);
      toast.success(`Collection "${name}" deleted`);

      if (selectedCollection === name) {
        setSelectedCollection(null);
      }

      fetchCollections();
    } catch (error: any) {
      console.error("Delete failed:", error);
      toast.error(error.message || "Failed to delete collection");
    }
  };

  const handleSendMessage = (message: string) => {
    // Create abort controller for this request
    const controller = new AbortController();
    setAbortController(controller);

    // Determine the actual chat mode based on PDF selection state
    if (pdfSelectionMode && selectedPDFs.length > 0) {
      sendMessage(message, null, "selected", sessionId, controller.signal);
    } else if (pdfSelectionMode && selectedPDFs.length === 0) {
      toast.warning("Please select PDFs first");
      setAbortController(null);
      return;
    } else if (chatMode === "single" && selectedCollection) {
      sendMessage(
        message,
        selectedCollection,
        "single",
        undefined,
        controller.signal,
      );
    } else if (chatMode === "chatall") {
      sendMessage(message, null, "chatall", undefined, controller.signal);
    } else {
      toast.warning("Please select a collection or PDFs first");
      setAbortController(null);
      return;
    }
  };

  // Add the stop handler
  const handleStopGeneration = () => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
    } else {
      // Fallback to manual stop
      stopGeneration();
    }
  };

  // Determine if input should be disabled
  const isInputDisabled = () => {
    if (pdfSelectionMode) {
      const disabled = selectedPDFs.length === 0;
      return disabled;
    }

    if (chatMode === "single") {
      const disabled = !selectedCollection;
      return disabled;
    }

    return false;
  };

  return (
    <div className="h-screen flex bg-gray-50 overflow-hidden">
      {/* Sidebar with all callbacks */}
      <Sidebar
        collections={collections}
        selectedCollection={selectedCollection}
        chatMode={chatMode}
        onSelectCollection={handleSelectCollection}
        onManageCollection={handleManageCollection}
        onRenameCollection={handleRenameCollection}
        onListPDFs={handleListPDFs}
        onAddPDFs={handleAddPDFs}
        onDeleteCollection={handleDeleteCollection}
        onUploadClick={handleFilesSelected}
        onChatModeChange={handleChatModeChange}
        onClearChat={clearMessages}
        pdfSelectionMode={pdfSelectionMode}
        onTogglePDFMode={handleTogglePDFMode}
        selectedPDFs={selectedPDFs}
        pdfStats={stats}
        onTogglePDF={togglePDF}
        onClearPDFSelection={clearSelection}
        onDeselectPDF={deselectPDF}
      />

      {/* Chat Area - Takes remaining space */}
      <div className="flex-1 flex flex-col min-w-0">
        <ChatArea
          messages={messages}
          isLoading={chatLoading}
          selectedCollection={selectedCollection}
          chatMode={getDisplayMode()}
          onSendMessage={handleSendMessage}
          onStopGeneration={handleStopGeneration}
          pdfSelectionMode={pdfSelectionMode}
          selectedPDFsCount={selectedPDFs.length}
        />
      </div>

      {/* Upload Modal - For new collections */}
      <SmartUploadModal
        isOpen={uploadModalOpen}
        onClose={() => setUploadModalOpen(false)}
        files={pendingFiles}
        onUploadComplete={handleUploadComplete}
      />

      {/* Rename Modal */}
      {selectedCollectionForAction && renameModalOpen && (
        <RenameCollectionModal
          isOpen={renameModalOpen}
          onClose={() => {
            setRenameModalOpen(false);
            setSelectedCollectionForAction(null);
          }}
          collectionName={selectedCollectionForAction}
          onSuccess={handleRenameComplete}
        />
      )}

      {/* PDF List Modal */}
      {selectedCollectionForAction && pdfListModalOpen && (
        <PDFListModal
          isOpen={pdfListModalOpen}
          onClose={() => {
            setPdfListModalOpen(false);
            setSelectedCollectionForAction(null);
          }}
          collectionName={selectedCollectionForAction}
        />
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
