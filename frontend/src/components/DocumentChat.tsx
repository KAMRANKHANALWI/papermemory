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
    null
  );
  const [chatMode, setChatMode] = useState<ChatMode>("single");

  // PDF Selection Mode State
  const [pdfSelectionMode, setPdfSelectionMode] = useState(false);
  const sessionId = "user_session_123";
  const { selectedPDFs } = usePDFSelection(sessionId);

  // Modal states
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [renameModalOpen, setRenameModalOpen] = useState(false);
  const [pdfListModalOpen, setPdfListModalOpen] = useState(false);
  const [addPDFsModalOpen, setAddPDFsModalOpen] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [selectedCollectionForAction, setSelectedCollectionForAction] =
    useState<string | null>(null);

  // To Abort Response
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
    stopGeneration, // Get the manual stop function
    clearMessages,
  } = useChat();
  const toast = useToast();

  // 🐛 DEBUG: Monitor all state changes
  useEffect(() => {
    console.log("═══════════════════════════════════════════════");
    console.log("📊 STATE UPDATE:");
    console.log("   pdfSelectionMode:", pdfSelectionMode);
    console.log("   chatMode:", chatMode);
    console.log("   selectedPDFs.length:", selectedPDFs.length);
    console.log("   selectedCollection:", selectedCollection);
    console.log("═══════════════════════════════════════════════");
  }, [pdfSelectionMode, chatMode, selectedPDFs.length, selectedCollection]);

  // Handle PDF selection mode toggle
  const handleTogglePDFMode = () => {
    console.log("🔘 TOGGLE PDF MODE BUTTON CLICKED");
    console.log("   Current pdfSelectionMode:", pdfSelectionMode);

    const newMode = !pdfSelectionMode;

    if (newMode) {
      console.log("   ✅ Entering PDF selection mode");
      // Entering PDF selection mode
      setPdfSelectionMode(true);
      setSelectedCollection(null); // Clear any selected collection
      // Don't change chatMode - we'll override it in getDisplayMode
      toast.info("Select PDFs from collections");
    } else {
      console.log("   ❌ Exiting PDF selection mode");
      // Exiting PDF selection mode - switch back to single mode
      setPdfSelectionMode(false);
      setChatMode("single");
      toast.info("Switched to single collection mode");
    }
  };

  // Handle collection selection
  const handleSelectCollection = (name: string) => {
    console.log("📁 COLLECTION SELECTED:", name);
    console.log("   Exiting PDF selection mode (if active)");

    // Selecting a collection exits PDF selection mode
    setPdfSelectionMode(false);
    setSelectedCollection(name);
    setChatMode("single");
  };

  // Handle chat mode change (Single Collection or All Collections buttons)
  const handleChatModeChange = (mode: ChatMode) => {
    console.log("🔄 CHAT MODE CHANGE:", mode);
    console.log("   Previous mode:", chatMode);
    console.log("   PDF selection mode active:", pdfSelectionMode);

    if (mode === "selected") {
      console.log(
        '   ⚠️ Ignoring "selected" mode - use handleTogglePDFMode instead'
      );
      return; // Don't allow direct setting of "selected" mode
    }

    // Exit PDF selection mode when switching to single or chatall
    if (pdfSelectionMode) {
      console.log("   ❌ Exiting PDF selection mode");
      setPdfSelectionMode(false);
    }

    setChatMode(mode);

    if (mode === "chatall") {
      setSelectedCollection(null);
      toast.info("Searching all collections");
    } else if (mode === "single") {
      toast.info("Single collection mode");
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
      `Are you sure you want to delete "${name}"?\n\nThis will permanently delete:\n• All PDFs in this collection\n• All associated chunks and embeddings\n\nThis action cannot be undone!`
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

  // Enhanced message send handler with PDF selection mode support
  const handleSendMessage = (message: string) => {
    console.log("💬 SEND MESSAGE ATTEMPT");
    console.log("   Message:", message);
    console.log("   pdfSelectionMode:", pdfSelectionMode);
    console.log("   selectedPDFs.length:", selectedPDFs.length);
    console.log("   chatMode:", chatMode);
    console.log("   selectedCollection:", selectedCollection);

    // Create abort controller for this request
    const controller = new AbortController();
    setAbortController(controller);

    // Determine the actual chat mode based on PDF selection state
    if (pdfSelectionMode && selectedPDFs.length > 0) {
      console.log("   ✅ Using SELECTED PDFs mode");
      sendMessage(message, null, "selected", sessionId, controller.signal);
    } else if (pdfSelectionMode && selectedPDFs.length === 0) {
      console.log("   ⚠️ PDF mode active but no PDFs selected");
      toast.warning("Please select PDFs first");
      setAbortController(null);
      return;
    } else if (chatMode === "single" && selectedCollection) {
      console.log("   ✅ Using SINGLE collection mode:", selectedCollection);
      sendMessage(
        message,
        selectedCollection,
        "single",
        undefined,
        controller.signal
      );
    } else if (chatMode === "chatall") {
      console.log("   ✅ Using CHAT ALL mode");
      sendMessage(message, null, "chatall", undefined, controller.signal);
    } else {
      console.log("   ❌ Invalid state - no valid mode");
      toast.warning("Please select a collection or PDFs first");
      setAbortController(null);
      return;
    }
  };

  // Add the stop handler
  const handleStopGeneration = () => {
    console.log("🛑 STOP GENERATION CLICKED");
    if (abortController) {
      abortController.abort();
      setAbortController(null);
    } else {
      // Fallback to manual stop
      stopGeneration();
    }
  };

  // Determine the display mode for ChatArea
  const getDisplayMode = (): "single" | "chatall" | "selected" => {
    console.log("🎯 getDisplayMode called:");
    console.log("   pdfSelectionMode:", pdfSelectionMode);
    console.log("   selectedPDFs.length:", selectedPDFs.length);
    console.log("   chatMode:", chatMode);

    // Priority 1: If PDF selection mode is active AND PDFs are selected
    if (pdfSelectionMode && selectedPDFs.length > 0) {
      console.log('   ✅ Returning "selected"');
      return "selected";
    }

    // Priority 2: Return the current chat mode
    console.log("   ➡️ Returning chatMode:", chatMode);
    return chatMode as "single" | "chatall";
  };

  // Determine if input should be disabled
  const isInputDisabled = () => {
    if (pdfSelectionMode) {
      // In PDF selection mode, only disabled if no PDFs selected
      const disabled = selectedPDFs.length === 0;
      console.log("🔒 Input disabled check (PDF mode):", disabled);
      return disabled;
    }

    if (chatMode === "single") {
      // In single mode, disabled if no collection selected
      const disabled = !selectedCollection;
      console.log("🔒 Input disabled check (single mode):", disabled);
      return disabled;
    }

    // In chatall mode, never disabled
    console.log("🔒 Input disabled check (chatall mode): false");
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

      {/* Modals */}

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
