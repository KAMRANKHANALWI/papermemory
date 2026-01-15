// src/components/Sidebar/CollectionList.tsx
"use client";

import { Collection } from "@/lib/types/collection";
import CollectionItem from "./CollectionItem";
import CollectionDropdown from "./CollectionDropdown";
import PDFCheckboxList from "./PDFCheckboxList";
import {
  MagnifyingGlassIcon,
  ChevronDownIcon,
} from "@heroicons/react/24/outline";
import { useState, useEffect, useMemo, useRef } from "react";

interface CollectionListProps {
  collections: Collection[];
  selectedCollection: string | null;
  onSelectCollection: (name: string) => void;
  onManageCollection: (name: string) => void;
  onRenameCollection: (name: string) => void;
  onListPDFs: (name: string) => void;
  onAddPDFs: (name: string) => void;
  onDeleteCollection: (name: string) => void;

  // NEW: PDF Selection props
  pdfSelectionMode?: boolean;
  selectedPDFs?: Set<string>;
  onTogglePDF?: (filename: string, collectionName: string) => void;
}

// Load access times from localStorage
const loadAccessTimes = (): Record<string, number> => {
  if (typeof window === "undefined") return {};
  try {
    const stored = localStorage.getItem("collection-access-times");
    return stored ? JSON.parse(stored) : {};
  } catch {
    return {};
  }
};

// Save access times to localStorage
const saveAccessTimes = (times: Record<string, number>) => {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem("collection-access-times", JSON.stringify(times));
  } catch {
    // Ignore errors
  }
};

export default function CollectionList({
  collections,
  selectedCollection,
  onSelectCollection,
  onManageCollection,
  onRenameCollection,
  onListPDFs,
  onAddPDFs,
  onDeleteCollection,
  pdfSelectionMode = false,
  selectedPDFs = new Set(),
  onTogglePDF = () => {},
}: CollectionListProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [lastAccessTimes, setLastAccessTimes] =
    useState<Record<string, number>>(loadAccessTimes);
  const [dropdownOpen, setDropdownOpen] = useState<string | null>(null);
  const [dropdownButtonRef, setDropdownButtonRef] =
    useState<React.RefObject<HTMLButtonElement> | null>(null);

  // NEW: State for expanded collection in PDF selection mode
  const [expandedCollection, setExpandedCollection] = useState<string | null>(
    null
  );

  // Track when new collections are added (they get current timestamp)
  useEffect(() => {
    const newCollections = collections.filter(
      (col) => !lastAccessTimes[col.name]
    );
    if (newCollections.length > 0) {
      const now = Date.now();
      setLastAccessTimes((prev) => {
        const updated = { ...prev };
        newCollections.forEach((col) => {
          updated[col.name] = now;
        });
        saveAccessTimes(updated);
        return updated;
      });
    }
  }, [collections, lastAccessTimes]);

  // Update last access time when a collection is selected
  useEffect(() => {
    if (selectedCollection) {
      setLastAccessTimes((prev) => {
        const updated = {
          ...prev,
          [selectedCollection]: Date.now(),
        };
        saveAccessTimes(updated);
        return updated;
      });
    }
  }, [selectedCollection]);

  // Sort collections by recency (most recently accessed/created first)
  const sortedCollections = useMemo(() => {
    return [...collections].sort((a, b) => {
      const aTime = lastAccessTimes[a.name] || 0;
      const bTime = lastAccessTimes[b.name] || 0;

      // Sort by most recent (higher timestamp = more recent = comes first)
      return bTime - aTime;
    });
  }, [collections, lastAccessTimes]);

  // Filter after sorting
  const filteredCollections = sortedCollections.filter((col) =>
    col.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleManage = (
    collectionName: string,
    buttonRef: React.RefObject<HTMLButtonElement>
  ) => {
    setDropdownOpen(collectionName);
    setDropdownButtonRef(buttonRef);
    onManageCollection(collectionName);
  };

  const handleCloseDropdown = () => {
    setDropdownOpen(null);
    setDropdownButtonRef(null);
  };

  // NEW: Handle collection click based on mode
  const handleCollectionClick = (collectionName: string) => {
    if (pdfSelectionMode) {
      // In PDF selection mode, toggle expansion to show PDFs
      setExpandedCollection(
        expandedCollection === collectionName ? null : collectionName
      );
    } else {
      // Normal mode - select collection for chat
      onSelectCollection(collectionName);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Section Header - Minimal like Claude */}
      <div className="flex-shrink-0 px-3 pt-3 pb-2">
        <h2 className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide">
          {pdfSelectionMode ? "Select from Collections" : "Collections"}
        </h2>
      </div>

      {/* Search - Only show if collections exist */}
      {collections.length > 0 && (
        <div className="flex-shrink-0 px-2 pb-2">
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-2.5 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" />
            <input
              type="text"
              placeholder="Search..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-8 pr-2.5 py-1.5 text-[13px] bg-gray-50 border border-transparent rounded-md
                       focus:outline-none focus:bg-white focus:border-gray-200 focus:ring-1 focus:ring-gray-200
                       placeholder:text-gray-400 transition-all duration-150"
            />
          </div>
        </div>
      )}

      {/* Collections List - SCROLLABLE */}
      <div className="flex-1 overflow-y-auto px-1.5 pb-2 space-y-0.5 min-h-0 bg-[#F8F8F3]">
        {filteredCollections.length === 0 ? (
          <div className="text-center py-12 px-4">
            <div className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-gray-100 mb-2.5">
              <svg
                className="w-5 h-5 text-gray-400"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.5}
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                />
              </svg>
            </div>
            {searchQuery ? (
              <>
                <p className="text-[13px] text-gray-600 font-medium">
                  No collections found
                </p>
                <p className="text-[12px] text-gray-400 mt-0.5">
                  Try a different search
                </p>
              </>
            ) : (
              <>
                <p className="text-[13px] text-gray-600 font-medium">
                  No collections yet
                </p>
                <p className="text-[12px] text-gray-400 mt-0.5">
                  Upload PDFs to get started
                </p>
              </>
            )}
          </div>
        ) : pdfSelectionMode ? (
          // PDF Selection Mode - Show expandable collections with checkboxes
          filteredCollections.map((collection) => (
            <div key={collection.name} className="mb-1">
              {/* Collection Header - Expandable */}
              <button
                onClick={() => handleCollectionClick(collection.name)}
                className={`
                  w-full flex items-center justify-between px-3 py-2.5 rounded-lg
                  transition-all duration-150 text-left
                  ${
                    expandedCollection === collection.name
                      ? "bg-stone-200 text-black"
                      : "hover:bg-stone-100 text-gray-900"
                  }
                `}
              >
                <div className="flex items-center gap-2">
                  <svg
                    className="w-5 h-5 text-gray-400"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={1.5}
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                    />
                  </svg>
                  <span className="text-[14px] font-medium">
                    {collection.name}
                  </span>
                </div>
                <ChevronDownIcon
                  className={`h-4 w-4 text-gray-400 transition-transform ${
                    expandedCollection === collection.name ? "rotate-180" : ""
                  }`}
                />
              </button>

              {/* PDF Checkboxes - Show when expanded */}
              {expandedCollection === collection.name && (
                <div className="mt-1 ml-2">
                  <PDFCheckboxList
                    collectionName={collection.name}
                    selectedPDFs={selectedPDFs}
                    onTogglePDF={onTogglePDF}
                  />
                </div>
              )}
            </div>
          ))
        ) : (
          // Normal Mode - Regular collection items
          filteredCollections.map((collection) => (
            <CollectionItem
              key={collection.name}
              collection={collection}
              isSelected={selectedCollection === collection.name}
              onSelect={() => handleCollectionClick(collection.name)}
              onManage={(buttonRef) => handleManage(collection.name, buttonRef)}
            />
          ))
        )}
      </div>

      {/* Dropdown Menu - Only in normal mode */}
      {!pdfSelectionMode && dropdownOpen && dropdownButtonRef && (
        <CollectionDropdown
          isOpen={true}
          onClose={handleCloseDropdown}
          collectionName={dropdownOpen}
          onRename={() => onRenameCollection(dropdownOpen)}
          onListPDFs={() => onListPDFs(dropdownOpen)}
          onAddPDFs={() => onAddPDFs(dropdownOpen)}
          onDelete={() => onDeleteCollection(dropdownOpen)}
          buttonRef={dropdownButtonRef}
        />
      )}
    </div>
  );
}
