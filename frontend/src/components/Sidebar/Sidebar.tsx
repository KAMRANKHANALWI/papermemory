// src/components/Sidebar/Sidebar.tsx
"use client";

import { useState } from "react";
import { Collection } from "@/lib/types/collection";
import CollectionList from "./CollectionList";
import UploadButton from "./UploadButton";
import ThemeToggle from "../UI/ThemeToggle";
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
  pdfSelectionMode?: boolean;
  onTogglePDFMode?: () => void;
  selectedPDFs?: any[];
  pdfStats?: any;
  onTogglePDF?: (filename: string, collectionName: string) => void;
  onClearPDFSelection?: () => void;
  onDeselectPDF?: (filename: string, collectionName: string) => void;
  isMobileOpen?: boolean;
  onMobileOpenChange?: (v: boolean) => void;
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
  selectedPDFs = [],
  pdfStats = null,
  onTogglePDF = () => {},
  onClearPDFSelection = () => {},
  onDeselectPDF = () => {},
  isMobileOpen = false,
  onMobileOpenChange = () => {},
}: SidebarProps) {
  const setIsMobileOpen = onMobileOpenChange;
  const [isCollapsed, setIsCollapsed] = useState(false);
  const toast = useToast();

  const selectedPDFsSet = new Set(
    selectedPDFs.map((pdf) => `${pdf.collection_name}:${pdf.filename}`),
  );

  /* ── Nav item style helper ── */
  const navItem = (active: boolean, danger = false) =>
    [
      "w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[15px] transition-all duration-150 cursor-pointer",
      danger
        ? "text-[var(--text-danger)] hover:bg-[var(--danger-light)]"
        : active
          ? "bg-[var(--bg-surface)] text-[var(--text-primary)] font-medium"
          : "text-[var(--text-secondary)] hover:bg-[var(--bg-surface)] hover:text-[var(--text-primary)]",
    ].join(" ");

  return (
    <>
      {/* Mobile Overlay */}
      {isMobileOpen && (
        <div
          className="lg:hidden fixed inset-0 z-40 backdrop-blur-sm"
          style={{ background: "rgba(0,0,0,0.4)" }}
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          flex flex-col h-full border-r transition-all duration-300 ease-in-out
          fixed lg:relative z-40
          ${isCollapsed ? "lg:w-[60px]" : "w-[280px]"}
          ${isMobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
        `}
        style={{
          background: "var(--bg-base)",
          borderColor: "var(--border-soft)",
        }}
      >
        {/* ── Collapsed State ── */}
        {isCollapsed ? (
          <div className="hidden lg:flex flex-col items-center py-4 gap-1 h-full">
            {/* Expand */}
            <button
              onClick={() => setIsCollapsed(false)}
              className="w-10 h-10 flex items-center justify-center rounded-lg transition-colors mb-2"
              style={{ color: "var(--text-muted)" }}
              title="Expand sidebar"
            >
              <ChevronRightIcon className="h-5 w-5" />
            </button>

            {/* Upload */}
            <button
              onClick={() => {
                const input = document.createElement("input");
                input.type = "file";
                input.multiple = true;
                input.accept = ".pdf";
                input.onchange = (e: any) => {
                  const files = Array.from(e.target.files || []) as File[];
                  if (files.length > 0) onUploadClick(files);
                };
                input.click();
              }}
              className="w-10 h-10 flex items-center justify-center rounded-lg transition-colors"
              style={{ color: "var(--text-secondary)" }}
              title="Upload PDFs"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </button>

            {/* Single */}
            <button
              onClick={() => onChatModeChange("single")}
              className="w-10 h-10 flex items-center justify-center rounded-lg transition-colors"
              style={{
                background:
                  chatMode === "single" ? "var(--bg-surface)" : "transparent",
                color:
                  chatMode === "single"
                    ? "var(--text-primary)"
                    : "var(--text-muted)",
              }}
              title="Single Collection"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"
                />
              </svg>
            </button>

            {/* All */}
            <button
              onClick={() => onChatModeChange("chatall")}
              className="w-10 h-10 flex items-center justify-center rounded-lg transition-colors"
              style={{
                background:
                  chatMode === "chatall" ? "var(--bg-surface)" : "transparent",
                color:
                  chatMode === "chatall"
                    ? "var(--text-primary)"
                    : "var(--text-muted)",
              }}
              title="All Collections"
            >
              <ChatBubbleLeftRightIcon className="w-5 h-5" strokeWidth={1.5} />
            </button>

            {/* Select PDFs */}
            <button
              onClick={onTogglePDFMode}
              className="w-10 h-10 flex items-center justify-center rounded-lg transition-colors"
              style={{
                background: pdfSelectionMode
                  ? "var(--accent-light)"
                  : "transparent",
                color: pdfSelectionMode ? "var(--accent)" : "var(--text-muted)",
              }}
              title="Select PDFs"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </button>

            {/* Clear */}
            <button
              onClick={() => {
                onClearChat();
                toast.success("Chat cleared");
              }}
              className="w-10 h-10 flex items-center justify-center rounded-lg transition-colors"
              style={{ color: "var(--text-danger)" }}
              title="Clear Chat"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
            </button>

            {/* Count */}
            <div className="mt-auto pb-4 flex flex-col items-center gap-0.5">
              <span
                className="text-base font-semibold"
                style={{
                  fontFamily: "var(--font-serif)",
                  color: "var(--text-primary)",
                }}
              >
                {collections.length}
              </span>
              <span
                className="text-[9px] uppercase tracking-widest"
                style={{ color: "var(--text-muted)" }}
              >
                cols
              </span>
            </div>
          </div>
        ) : (
          /* ── Expanded State ── */
          <div className="flex flex-col h-full">
            {/* ── Header / Wordmark ── */}
            {/* ── Header / Wordmark only ── */}
            <div className="flex-shrink-0 px-4 pt-4 pb-3">
              <div className="flex items-start justify-between">
                <div className="flex flex-col gap-0">
                  <span
                    className="text-[10px] tracking-[0.18em] uppercase font-semibold"
                    style={{
                      color: "var(--text-accent)",
                      fontFamily: "var(--font-serif)",
                    }}
                  >
                    AI Research Assistant
                  </span>
                  <h1
                    className="text-[32px] font-bold tracking-[-0.02em] leading-none"
                    style={{
                      fontFamily: "var(--font-serif)",
                      color: "var(--text-primary)",
                    }}
                  >
                    GutMiScholar
                  </h1>
                </div>

                <div className="flex items-center gap-1 pt-1">
                  <ThemeToggle />
                  <button
                    onClick={() => setIsCollapsed(true)}
                    className="hidden lg:flex p-1.5 rounded-md transition-colors"
                    style={{ color: "var(--text-muted)" }}
                  >
                    <ChevronLeftIcon className="h-4 w-4" />
                  </button>
                  <button
                    onClick={() => setIsMobileOpen(false)}
                    className="lg:hidden p-1.5 rounded-md transition-colors"
                    style={{ color: "var(--text-muted)" }}
                  >
                    <XMarkIcon className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* ── Navigation ── */}
            <nav
              className="flex-shrink-0 px-2 py-3 space-y-0.5 border-b"
              style={{ borderColor: "var(--border-soft)" }}
            >
              {/* New Chat — first item in nav */}
              <button
                onClick={onClearChat}
                className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg transition-colors duration-150"
                onMouseEnter={(e) =>
                  (e.currentTarget.style.background = "var(--bg-surface)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.background = "transparent")
                }
              >
                <div
                  className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0"
                  style={{ background: "var(--text-muted)" }}
                >
                  <PlusIcon
                    className="h-3.5 w-3.5 text-white"
                    strokeWidth={2.5}
                  />
                </div>
                <span
                  className="text-[14.5px]"
                  style={{ color: "var(--text-primary)" }}
                >
                  New Chat
                </span>
              </button>

              <UploadButton onFilesSelected={onUploadClick} />

              {/* Single collection */}
              <button
                onClick={() => onChatModeChange("single")}
                className={navItem(chatMode === "single" && !pdfSelectionMode)}
              >
                <svg
                  className="w-[18px] h-[18px] flex-shrink-0"
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

              {/* All collections */}
              <button
                onClick={() => onChatModeChange("chatall")}
                className={navItem(chatMode === "chatall" && !pdfSelectionMode)}
              >
                <ChatBubbleLeftRightIcon
                  className="w-[18px] h-[18px] flex-shrink-0"
                  strokeWidth={1.5}
                />
                <span>All Collections</span>
              </button>

              {/* Select PDFs */}
              <button
                onClick={onTogglePDFMode}
                className={navItem(pdfSelectionMode)}
                style={
                  pdfSelectionMode
                    ? {
                        background: "var(--accent-light)",
                        color: "var(--accent)",
                      }
                    : {}
                }
              >
                <svg
                  className="w-[18px] h-[18px] flex-shrink-0"
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
                  <span
                    className="ml-auto text-[11px] px-1.5 py-0.5 rounded-full font-medium"
                    style={{
                      background: "var(--accent-light)",
                      color: "var(--accent)",
                    }}
                  >
                    {selectedPDFs.length}
                  </span>
                )}
              </button>

              {/* Clear Chat */}
              <button
                onClick={() => {
                  onClearChat();
                  toast.success("Chat cleared");
                }}
                className={navItem(false, true)}
              >
                <svg
                  className="w-[18px] h-[18px] flex-shrink-0"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={1.5}
                  viewBox="0 0 24 24"
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

            {/* ── Collections / PDF Selection ── */}
            {pdfSelectionMode ? (
              <>
                <div
                  className="flex-shrink-0 border-b"
                  style={{ borderColor: "var(--border-soft)" }}
                >
                  <SelectedPDFsDisplay
                    selectedPDFs={selectedPDFs}
                    stats={pdfStats}
                    onRemovePDF={onDeselectPDF}
                    onClearAll={onClearPDFSelection}
                  />
                </div>
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
                    onTogglePDF={onTogglePDF}
                  />
                </div>
              </>
            ) : (
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

            {/* ── Footer ── */}
            <div
              className="flex-shrink-0 px-4 py-3 border-t"
              style={{ borderColor: "var(--border-soft)" }}
            >
              <p
                className="text-[12px] text-center"
                style={{
                  fontFamily: "var(--font-serif)",
                  fontStyle: "italic",
                  color: "var(--text-muted)",
                }}
              >
                RAG System · Powered by AI
              </p>
            </div>
          </div>
        )}
      </aside>
    </>
  );
}
