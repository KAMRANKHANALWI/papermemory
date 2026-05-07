// src/components/DocumentChat.tsx
// ── Only the outer wrapper div needs updating — everything else unchanged ──
// Change: className="h-screen flex bg-gray-50 overflow-hidden"
//      to: className="h-screen flex overflow-hidden" style={{ background: "var(--bg-main)" }}
//
// Full file with that one change applied:

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
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
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
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [selectedCollectionForAction, setSelectedCollectionForAction] =
    useState<string | null>(null);
  // const [abortController, setAbortController] =
  //   useState<AbortController | null>(null);

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

  const handleTogglePDFMode = () => {
    const newMode = !pdfSelectionMode;
    if (newMode) {
      setPdfSelectionMode(true);
      setSelectedCollection(null);
      fetchSelection();
      toast.info("Select PDFs Mode");
    } else {
      setPdfSelectionMode(false);
      setChatMode("single");
      toast.info("Switched to single collection mode");
    }
  };

  const getDisplayMode = (): "single" | "chatall" | "selected" => {
    if (pdfSelectionMode && selectedPDFs.length > 0) return "selected";
    return chatMode as "single" | "chatall";
  };

  const handleSelectCollection = (name: string) => {
    setPdfSelectionMode(false);
    setSelectedCollection(name);
    setChatMode("single");
  };

  const handleChatModeChange = (mode: ChatMode) => {
    if (mode === "selected") return;
    if (pdfSelectionMode) setPdfSelectionMode(false);
    setChatMode(mode);
    if (mode === "chatall") {
      setSelectedCollection(null);
      toast.info("All Collections Mode");
    } else if (mode === "single") {
      toast.info("Single Collection Mode");
    }
  };

  const handleFilesSelected = (files: File[]) => {
    setPendingFiles(files);
    setUploadModalOpen(true);
  };
  const handleUploadComplete = () => {
    setPendingFiles([]);
    fetchCollections();
  };
  const handleRenameComplete = () => {
    fetchCollections();
    setRenameModalOpen(false);
    setSelectedCollectionForAction(null);
  };
  const handleManageCollection = (name: string) => {
    setSelectedCollectionForAction(name);
  };
  const handleRenameCollection = (name: string) => {
    setSelectedCollectionForAction(name);
    setRenameModalOpen(true);
  };
  const handleListPDFs = (name: string) => {
    setSelectedCollectionForAction(name);
    setPdfListModalOpen(true);
  };

  // const handleAddPDFs = (name: string) => {
  //   setSelectedCollectionForAction(name);
  //   const input = document.createElement("input");
  //   input.type = "file"; input.multiple = true; input.accept = ".pdf";
  //   input.onchange = (e: any) => {
  //     const files = Array.from(e.target.files || []) as File[];
  //     if (files.length > 0) {
  //       import("@/lib/api/collections").then(({ collectionsApi }) => {
  //         collectionsApi.addPDFs(name, files)
  //           .then(() => { toast.success(`Added ${files.length} file(s) to "${name}"`); fetchCollections(); })
  //           .catch(err => toast.error(err.message || "Failed to add files"));
  //       });
  //     }
  //   };
  //   input.click();
  // };

  const handleAddPDFs = (name: string) => {
    setSelectedCollectionForAction(name);

    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ".pdf";
    input.onchange = async (e: any) => {
      const files = Array.from(e.target.files || []) as File[];
      if (files.length === 0) return;

      // 1. Show a persistent loading toast immediately
      const loadingToastId = toast.info(
        `Processing ${files.length} file${files.length > 1 ? "s" : ""}... this may take a minute`,
        0, // duration 0 = stays until manually removed
      );

      try {
        const { collectionsApi } = await import("@/lib/api/collections");
        await collectionsApi.addPDFs(name, files);
        toast.removeToast(loadingToastId); // dismiss loading
        toast.success(
          `Added ${files.length} file${files.length > 1 ? "s" : ""} to "${name}"`,
        );
        fetchCollections();
      } catch (error: any) {
        toast.removeToast(loadingToastId);
        toast.error(error.message || "Failed to add files");
      }
    };
    input.click();
  };

  const handleDeleteCollection = async (name: string) => {
    if (
      !window.confirm(
        `Delete "${name}"?\n\nThis will permanently delete all PDFs and data.`,
      )
    )
      return;
    try {
      await deleteCollection(name);
      toast.success(`Collection "${name}" deleted`);
      if (selectedCollection === name) setSelectedCollection(null);
      fetchCollections();
    } catch (err: any) {
      toast.error(err.message || "Failed to delete collection");
    }
  };

  // const handleSendMessage = (message: string) => {
  //   const controller = new AbortController();
  //   setAbortController(controller);
  //   if (pdfSelectionMode && selectedPDFs.length > 0) {
  //     sendMessage(message, null, "selected", sessionId, controller.signal);
  //   } else if (pdfSelectionMode && selectedPDFs.length === 0) {
  //     toast.warning("Please select PDFs first");
  //     setAbortController(null);
  //   } else if (chatMode === "single" && selectedCollection) {
  //     sendMessage(
  //       message,
  //       selectedCollection,
  //       "single",
  //       undefined,
  //       controller.signal,
  //     );
  //   } else if (chatMode === "chatall") {
  //     sendMessage(message, null, "chatall", undefined, controller.signal);
  //   } else {
  //     toast.warning("Please select a collection or PDFs first");
  //     setAbortController(null);
  //   }
  // };

  const handleSendMessage = (message: string) => {
    if (pdfSelectionMode && selectedPDFs.length > 0) {
      sendMessage(message, null, "selected", sessionId);
    } else if (pdfSelectionMode && selectedPDFs.length === 0) {
      toast.warning("Please select PDFs first");
    } else if (chatMode === "single" && selectedCollection) {
      sendMessage(message, selectedCollection, "single");
    } else if (chatMode === "chatall") {
      sendMessage(message, null, "chatall");
    } else {
      toast.warning("Please select a collection or PDFs first");
    }
  };


  const handleStopGeneration = () => {
    stopGeneration();
  };

  const isInputDisabled = () => {
    if (pdfSelectionMode) return selectedPDFs.length === 0;
    if (chatMode === "single") return !selectedCollection;
    return false;
  };

  return (
    <div
      className="h-screen flex overflow-hidden"
      style={{ background: "var(--bg-main)" }}
    >
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
        isMobileOpen={mobileSidebarOpen}
        onMobileOpenChange={setMobileSidebarOpen}
      />

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
          onOpenSidebar={() => setMobileSidebarOpen(true)}
        />
      </div>

      <SmartUploadModal
        isOpen={uploadModalOpen}
        onClose={() => setUploadModalOpen(false)}
        files={pendingFiles}
        onUploadComplete={handleUploadComplete}
      />

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

      <ToastContainer />
    </div>
  );
}

export default function DocumentChat() {
  return (
    <ToastProvider>
      <DocumentChatContent />
    </ToastProvider>
  );
}
