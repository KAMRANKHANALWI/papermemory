// src/components/Sidebar/Sidebar.tsx
"use client";

import { useState } from "react";
import { Collection } from "@/lib/types/collection";
import CollectionList from "./CollectionList";
import UploadButton from "./UploadButton";
import Button from "../UI/Button";
import {
  ChatBubbleLeftRightIcon,
  Bars3Icon,
  XMarkIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from "@heroicons/react/24/outline";

interface SidebarProps {
  collections: Collection[];
  selectedCollection: string | null;
  chatMode: "single" | "chatall";
  onSelectCollection: (name: string) => void;
  onDeleteCollection: (name: string) => void;
  onManageCollection: (name: string) => void;
  onUploadClick: (files: File[]) => void;
  onChatModeChange: (mode: "single" | "chatall") => void;
  onClearChat: () => void;
}

export default function Sidebar({
  collections,
  selectedCollection,
  chatMode,
  onSelectCollection,
  onDeleteCollection,
  onManageCollection,
  onUploadClick,
  onChatModeChange,
  onClearChat,
}: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  return (
    <>
      {/* Mobile Toggle Button (visible on small screens) */}
      <button
        onClick={() => setIsMobileOpen(!isMobileOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-white rounded-lg shadow-lg border border-gray-200"
      >
        {isMobileOpen ? (
          <XMarkIcon className="h-6 w-6 text-gray-700" />
        ) : (
          <Bars3Icon className="h-6 w-6 text-gray-700" />
        )}
      </button>

      {/* Mobile Overlay */}
      {isMobileOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-40"
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div
        className={`
          ${isCollapsed ? "w-16" : "w-80"} 
          bg-white border-r border-gray-200 flex flex-col h-full
          transition-all duration-300 ease-in-out
          fixed lg:relative z-40
          ${
            isMobileOpen
              ? "translate-x-0"
              : "-translate-x-full lg:translate-x-0"
          }
        `}
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          {!isCollapsed && (
            <div className="flex-1">
              <h1 className="text-xl font-bold text-gray-900">PaperMemory</h1>
              <p className="text-xs text-gray-500 mt-1">
                Turn documents into intelligent answers.
                {/* Supercharge your LLMs with document memory */}
              </p>
            </div>
          )}

          {/* Desktop Toggle Button */}
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="hidden lg:block p-2 hover:bg-gray-100 rounded-lg transition-colors"
            title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {isCollapsed ? (
              <ChevronRightIcon className="h-5 w-5 text-gray-600" />
            ) : (
              <ChevronLeftIcon className="h-5 w-5 text-gray-600" />
            )}
          </button>
        </div>

        {/* Collapsed State - Show Icons Only */}
        {isCollapsed ? (
          <div className="flex-1 flex flex-col items-center py-4 space-y-4">
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
              className="p-3 bg-amber-600 hover:bg-amber-700 text-white rounded-lg transition-colors"
              title="Upload PDFs"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </button>

            {/* Chat Mode Toggle Icons */}
            <button
              onClick={() => onChatModeChange("single")}
              className={`p-3 rounded-lg transition-colors ${
                chatMode === "single"
                  ? "bg-amber-100 text-amber-600"
                  : "text-gray-400 hover:bg-gray-100"
              }`}
              title="Single Collection Mode"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"
                />
              </svg>
            </button>

            <button
              onClick={() => onChatModeChange("chatall")}
              className={`p-3 rounded-lg transition-colors ${
                chatMode === "chatall"
                  ? "bg-purple-100 text-purple-600"
                  : "text-gray-400 hover:bg-gray-100"
              }`}
              title="Chat All Collections"
            >
              <ChatBubbleLeftRightIcon className="w-6 h-6" />
            </button>

            {/* Clear Chat Icon */}
            <button
              onClick={onClearChat}
              className="p-3 text-gray-400 hover:bg-gray-100 rounded-lg transition-colors"
              title="Clear Chat"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
            </button>

            {/* Collections Count */}
            <div className="mt-auto pt-4 border-t border-gray-200 w-full flex flex-col items-center">
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900">
                  {collections.length}
                </div>
                <div className="text-xs text-gray-500">Collections</div>
              </div>
            </div>
          </div>
        ) : (
          /* Expanded State - Full Sidebar */
          <>
            {/* Chat Mode Toggle */}
            <div className="p-3 border-b border-gray-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                  Chat Mode
                </span>
                <button
                  onClick={onClearChat}
                  className="text-xs text-amber-600 hover:text-amber-700 font-medium"
                >
                  Clear Chat
                </button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => onChatModeChange("single")}
                  className={`
                    px-3 py-2 text-sm font-medium rounded-lg transition-colors
                    ${
                      chatMode === "single"
                        ? "bg-amber-600 text-white"
                        : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                    }
                  `}
                >
                  Single
                </button>
                <button
                  onClick={() => onChatModeChange("chatall")}
                  className={`
                    px-3 py-2 text-sm font-medium rounded-lg transition-colors
                    ${
                      chatMode === "chatall"
                        ? "bg-purple-600 text-white"
                        : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                    }
                  `}
                >
                  <ChatBubbleLeftRightIcon className="h-4 w-4 inline mr-1" />
                  All
                </button>
              </div>
            </div>

            {/* Upload Button */}
            <div className="p-3 border-b border-gray-200">
              <UploadButton onFilesSelected={onUploadClick} />
            </div>

            {/* Collections List */}
            <CollectionList
              collections={collections}
              selectedCollection={selectedCollection}
              onSelectCollection={onSelectCollection}
              onDeleteCollection={onDeleteCollection}
              onManageCollection={onManageCollection}
            />

            {/* Footer */}
            <div className="p-3 border-t border-gray-200">
              <p className="text-xs text-gray-500 text-center">
                Powered by AI • Enhanced RAG System
              </p>
            </div>
          </>
        )}
      </div>
    </>
  );
}
