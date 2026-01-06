// src/components/Sidebar/Sidebar.tsx
"use client";

import { useState } from "react";
import { Collection } from "@/lib/types/collection";
import CollectionList from "./CollectionList";
import UploadButton from "./UploadButton";
import {
  ChatBubbleLeftRightIcon,
  Bars3Icon,
  XMarkIcon,
  PlusIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from "@heroicons/react/24/outline";
import { useToast } from "@/hooks/useToast";
import Badge from "../UI/Badge";

// NEW: PDF Selection imports
import { usePDFSelection } from "@/hooks/usePDFSelection";
import SelectedPDFsDisplay from "./SelectedPDFsDisplay";
import { ChatMode } from "@/hooks/useChat";

interface SidebarProps {
  collections: Collection[];
  selectedCollection: string | null;
  chatMode: ChatMode;
  onSelectCollection: (name: string) => void;
  onManageCollection: (name: string) => void;
  onRenameCollection: (name: string) => void;
  onListPDFs: (name: string) => void;
  onAddPDFs: (name: string) => void;
  onDeleteCollection: (name: string) => void;
  onUploadClick: (files: File[]) => void;
  onChatModeChange: (mode: ChatMode) => void;
  onClearChat: () => void;
  // NEW: PDF Selection Mode props
  pdfSelectionMode?: boolean;
  onTogglePDFMode?: () => void;
}

export default function Sidebar({
  collections,
  selectedCollection,
  chatMode,
  onSelectCollection,
  onManageCollection,
  onRenameCollection,
  onListPDFs,
  onAddPDFs,
  onDeleteCollection,
  onUploadClick,
  onChatModeChange,
  onClearChat,
  pdfSelectionMode = false,
  onTogglePDFMode = () => {},
}: SidebarProps) {
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const toast = useToast();

  // NEW: Initialize PDF Selection Hook
  const sessionId = "user_session_123"; // You can make this dynamic based on user ID
  const {
    selectedPDFs,
    stats,
    togglePDF,
    clearSelection,
    deselectPDF,
  } = usePDFSelection(sessionId);

  // NEW: Create Set for quick lookup
  const selectedPDFsSet = new Set(
    selectedPDFs.map((pdf) => `${pdf.collection_name}:${pdf.filename}`)
  );

  return (
    <>
      {/* Mobile Toggle Button */}
      <button
        onClick={() => setIsMobileOpen(!isMobileOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-white rounded-lg shadow-md border border-gray-200 hover:bg-gray-50 transition-colors"
        aria-label="Toggle sidebar"
      >
        {isMobileOpen ? (
          <XMarkIcon className="h-5 w-5 text-gray-700" />
        ) : (
          <Bars3Icon className="h-5 w-5 text-gray-700" />
        )}
      </button>

      {/* Mobile Overlay */}
      {isMobileOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-40 backdrop-blur-sm"
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          bg-white flex flex-col h-full border-r border-gray-100
          transition-all duration-300 ease-in-out
          fixed lg:relative z-40
          ${isCollapsed ? "lg:w-16" : "w-[300px]"}
          ${
            isMobileOpen
              ? "translate-x-0"
              : "-translate-x-full lg:translate-x-0"
          }
        `}
      >
        {/* Collapsed State - Icon Only */}
        {isCollapsed ? (
          <div className="hidden lg:flex flex-col items-center py-4 space-y-2 h-full">
            {/* Expand Button at Top */}
            <button
              onClick={() => setIsCollapsed(false)}
              className="w-12 h-12 flex items-center justify-center hover:bg-gray-100 rounded-xl transition-colors mb-2"
              title="Expand sidebar"
              aria-label="Expand sidebar"
            >
              <ChevronRightIcon className="h-5 w-5 text-gray-600" />
            </button>

            {/* Upload Icon */}
            <button
              onClick={() => {
                const input = document.createElement("input");
                input.type = "file";
                input.multiple = true;
                input.accept = ".pdf";
                input.onchange = (e: any) => {
                  const files = Array.from(e.target.files || []) as File[];
                  if (files.length > 0) {
                    onUploadClick(files);
                  }
                };
                input.click();
              }}
              className="w-12 h-12 flex items-center justify-center bg-stone-500 hover:bg-stone-600 text-white rounded-xl transition-colors shadow-sm cursor-pointer"
              title="Upload PDFs"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </button>

            {/* Single Collection Icon */}
            <button
              onClick={() => onChatModeChange("single")}
              className={`w-12 h-12 flex items-center justify-center rounded-xl transition-colors cursor-pointer ${
                chatMode === "single"
                  ? "bg-stone-100 text-stone-600"
                  : "bg-gray-100 text-gray-500 hover:bg-gray-200"
              }`}
              title="Single Collection Mode"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"
                />
              </svg>
            </button>

            {/* All Collections Icon */}
            <button
              onClick={() => onChatModeChange("chatall")}
              className={`w-12 h-12 flex items-center justify-center rounded-xl transition-colors cursor-pointer ${
                chatMode === "chatall"
                  ? "bg-stone-100 text-stone-600"
                  : "bg-gray-100 text-gray-500 hover:bg-gray-200"
              }`}
              title="Chat All Collections"
            >
              <ChatBubbleLeftRightIcon className="w-6 h-6" strokeWidth={2} />
            </button>

            {/* NEW: PDF Selection Mode Icon */}
            <button
              onClick={onTogglePDFMode}
              className={`w-12 h-12 flex items-center justify-center rounded-xl transition-colors cursor-pointer ${
                pdfSelectionMode
                  ? "bg-amber-100 text-amber-600"
                  : "bg-gray-100 text-gray-500 hover:bg-gray-200"
              }`}
              title="Select PDFs"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </button>

            {/* Clear Chat Icon */}
            <button
              onClick={() => {
                onClearChat();
                toast.success("Chat cleared successfully");
              }}
              className="w-12 h-12 flex items-center justify-center bg-gray-100 text-red-500 hover:bg-red-200 rounded-xl transition-colors cursor-pointer"
              title="Clear Chat"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
            </button>

            {/* Collections Count at Bottom */}
            <div className="mt-auto pt-4 w-full flex flex-col items-center border-t border-gray-100">
              <div className="text-center">
                <div className="text-xl font-bold text-gray-900">
                  {collections.length}
                </div>
                <div className="text-[10px] text-gray-400 font-medium">
                  Collections
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Expanded State - Full Sidebar */
          <div className="flex flex-col h-full bg-[#F8F8F3]">
            {/* Header */}
            <div className="flex-shrink-0 px-3 pt-4 pb-3">
              <div className="flex items-center justify-between mb-4">
                <h1 className="text-[19px] font-semibold text-gray-900">
                  PaperMemory
                </h1>

                {/* Desktop Toggle Button - Inside header like Claude */}
                <button
                  onClick={() => setIsCollapsed(true)}
                  className="hidden lg:flex p-1.5 hover:bg-gray-100 rounded-md transition-colors"
                  aria-label="Close sidebar"
                  title="Close sidebar"
                >
                  <ChevronLeftIcon className="h-5 w-5 text-gray-600" />
                </button>

                {/* Mobile Close Button */}
                <button
                  onClick={() => setIsMobileOpen(false)}
                  className="lg:hidden p-1.5 hover:bg-gray-100 rounded-md transition-colors"
                  aria-label="Close sidebar"
                >
                  <XMarkIcon className="h-5 w-5 text-gray-500" />
                </button>
              </div>

              {/* New Chat Button */}
              <button
                onClick={onClearChat}
                className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg
                       hover:bg-stone-100 text-gray-900
                         transition-colors duration-150 group cursor-pointer active:bg-stone-300"
              >
                <div
                  className="flex-shrink-0 w-8 h-8 rounded-full bg-stone-500 
                              flex items-center justify-center group-hover:bg-stone-600 transition-colors"
                >
                  <PlusIcon className="h-5 w-5 text-white" strokeWidth={2.5} />
                </div>
                <span className="text-[15px] font-medium">New Chat</span>
              </button>
            </div>

            {/* Navigation Section */}
            <nav className="flex-shrink-0 px-2 pb-3 space-y-1 border-b border-gray-100">
              {/* Upload Button */}
              <UploadButton onFilesSelected={onUploadClick} />

              {/* Chat Mode Buttons */}
              <button
                onClick={() => onChatModeChange("single")}
                className={`
                  w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[15px]
                  transition-all duration-150 cursor-pointer active:bg-stone-300
                  ${
                    chatMode === "single" && !pdfSelectionMode
                      ? "bg-stone-200 text-black"
                      : "text-gray-900 hover:bg-stone-100"
                  }
                `}
              >
                <svg
                  className="w-6 h-6 flex-shrink-0"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={1.5}
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"
                  />
                </svg>
                <span>Single collection</span>
              </button>

              <button
                onClick={() => onChatModeChange("chatall")}
                className={`
                  w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[15px]
                  transition-all duration-150 cursor-pointer active:bg-stone-300
                  ${
                    chatMode === "chatall" && !pdfSelectionMode
                      ? "bg-stone-200 text-black"
                      : "text-gray-900 hover:bg-stone-100"
                  }
                `}
              >
                <ChatBubbleLeftRightIcon
                  className="w-6 h-6 flex-shrink-0"
                  strokeWidth={1.5}
                />
                <span>All Collections</span>
              </button>

              {/* NEW: PDF Selection Mode Button */}
              <button
                onClick={onTogglePDFMode}
                className={`
                  w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[15px]
                  transition-all duration-150 cursor-pointer active:bg-stone-300
                  ${
                    pdfSelectionMode
                      ? "bg-amber-100 text-amber-900"
                      : "text-gray-900 hover:bg-stone-100"
                  }
                `}
              >
                <svg
                  className="w-6 h-6 flex-shrink-0"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={1.5}
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <span>Select PDFs</span>
                {selectedPDFs.length > 0 && (
                  <Badge variant="info" size="sm">
                    {selectedPDFs.length}
                  </Badge>
                )}
              </button>

              <button
                onClick={() => {
                  onClearChat();
                  toast.success("Chat cleared successfully");
                }}
                className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[15px]
             text-gray-900 hover:bg-stone-100 active:bg-stone-300
             transition-all duration-150 cursor-pointer"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className="w-6 h-6 flex-shrink-0"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0"
                  />
                </svg>
                <span>Clear Chat</span>
              </button>
            </nav>

            {/* Main Content Area - Changes based on PDF Selection Mode */}
            {pdfSelectionMode ? (
              <>
                {/* NEW: Selected PDFs Display - Collapsible Header */}
                <div className="flex-shrink-0 border-b border-gray-100">
                  <SelectedPDFsDisplay
                    selectedPDFs={selectedPDFs}
                    stats={stats}
                    onRemovePDF={deselectPDF}
                    onClearAll={clearSelection}
                  />
                </div>

                {/* NEW: Collections with PDF Checkboxes - Takes Remaining Space */}
                <div className="flex-1 min-h-0 overflow-hidden">
                  <CollectionList
                    collections={collections}
                    selectedCollection={selectedCollection}
                    onSelectCollection={onSelectCollection}
                    onManageCollection={onManageCollection}
                    onRenameCollection={onRenameCollection}
                    onListPDFs={onListPDFs}
                    onAddPDFs={onAddPDFs}
                    onDeleteCollection={onDeleteCollection}
                    pdfSelectionMode={true}
                    selectedPDFs={selectedPDFsSet}
                    onTogglePDF={togglePDF}
                  />
                </div>
              </>
            ) : (
              /* Normal Mode - Regular Collections List */
              <div className="flex-1 min-h-0 overflow-hidden">
                <CollectionList
                  collections={collections}
                  selectedCollection={selectedCollection}
                  onSelectCollection={onSelectCollection}
                  onManageCollection={onManageCollection}
                  onRenameCollection={onRenameCollection}
                  onListPDFs={onListPDFs}
                  onAddPDFs={onAddPDFs}
                  onDeleteCollection={onDeleteCollection}
                  pdfSelectionMode={false}
                />
              </div>
            )}

            {/* Footer */}
            <div className="flex-shrink-0 px-3 py-2.5 border-t border-gray-100">
              <p className="text-[12px] text-gray-400 text-center font-medium">
                RAG System • Powered by AI
              </p>
            </div>
          </div>
        )}
      </aside>
    </>
  );
}
