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
import { useState, useEffect, useMemo } from "react";

interface CollectionListProps {
  collections: Collection[];
  selectedCollection: string | null;
  onSelectCollection: (name: string) => void;
  onManageCollection: (name: string) => void;
  onRenameCollection: (name: string) => void;
  onListPDFs: (name: string) => void;
  onAddPDFs: (name: string) => void;
  onDeleteCollection: (name: string) => void;
  pdfSelectionMode?: boolean;
  selectedPDFs?: Set<string>;
  onTogglePDF?: (filename: string, collectionName: string) => void;
}

const loadAccessTimes = (): Record<string, number> => {
  if (typeof window === "undefined") return {};
  try {
    const stored = localStorage.getItem("collection-access-times");
    return stored ? JSON.parse(stored) : {};
  } catch {
    return {};
  }
};

const saveAccessTimes = (times: Record<string, number>) => {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem("collection-access-times", JSON.stringify(times));
  } catch {}
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
  const [expandedCollection, setExpandedCollection] = useState<string | null>(
    null,
  );

  useEffect(() => {
    const newCollections = collections.filter(
      (col) => !lastAccessTimes[col.name],
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

  useEffect(() => {
    if (selectedCollection) {
      setLastAccessTimes((prev) => {
        const updated = { ...prev, [selectedCollection]: Date.now() };
        saveAccessTimes(updated);
        return updated;
      });
    }
  }, [selectedCollection]);

  const sortedCollections = useMemo(
    () =>
      [...collections].sort(
        (a, b) =>
          (lastAccessTimes[b.name] || 0) - (lastAccessTimes[a.name] || 0),
      ),
    [collections, lastAccessTimes],
  );

  const filteredCollections = sortedCollections.filter((col) =>
    col.name.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  const handleManage = (
    collectionName: string,
    buttonRef: React.RefObject<HTMLButtonElement>,
  ) => {
    setDropdownOpen(collectionName);
    setDropdownButtonRef(buttonRef);
    onManageCollection(collectionName);
  };

  const handleCollectionClick = (collectionName: string) => {
    if (pdfSelectionMode) {
      setExpandedCollection(
        expandedCollection === collectionName ? null : collectionName,
      );
    } else {
      onSelectCollection(collectionName);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Section label */}
      <div className="flex-shrink-0 px-4 pt-3 pb-2">
        <h2
          className="text-[10px] uppercase tracking-[0.16em] font-medium"
          style={{ color: "var(--text-muted)" }}
        >
          {pdfSelectionMode ? "Select from Collections" : "Collections"}
        </h2>
      </div>

      {/* Search */}
      {collections.length > 0 && (
        <div className="flex-shrink-0 px-2 pb-2">
          <div className="relative">
            <MagnifyingGlassIcon
              className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 pointer-events-none"
              style={{ color: "var(--text-muted)" }}
            />
            <input
              type="text"
              placeholder="Search collections..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-7 pr-2.5 py-1.5 rounded-lg border outline-none transition-all text-[13px]"
              style={{
                background: "transparent",
                borderColor: "var(--border-soft)",
                color: "var(--text-primary)",
                fontFamily: "var(--font-serif)",
                letterSpacing: "0.01em",
              }}
              onFocus={(e) => {
                e.target.style.borderColor = "var(--accent)";
                e.target.style.background = "var(--bg-input)";
              }}
              onBlur={(e) => {
                e.target.style.borderColor = "var(--border-soft)";
                e.target.style.background = "transparent";
              }}
            />
          </div>
        </div>
      )}

      {/* List */}
      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5 min-h-0">
        {filteredCollections.length === 0 ? (
          <div className="text-center py-12 px-4">
            <div
              className="inline-flex items-center justify-center w-10 h-10 rounded-full mb-3"
              style={{ background: "var(--bg-surface)" }}
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.5}
                viewBox="0 0 24 24"
                style={{ color: "var(--text-muted)" }}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                />
              </svg>
            </div>
            <p
              className="text-[13px] font-medium"
              style={{ color: "var(--text-secondary)" }}
            >
              {searchQuery ? "No collections found" : "No collections yet"}
            </p>
            <p
              className="text-[11px] mt-1"
              style={{ color: "var(--text-muted)" }}
            >
              {searchQuery
                ? "Try a different search"
                : "Upload PDFs to get started"}
            </p>
          </div>
        ) : pdfSelectionMode ? (
          filteredCollections.map((collection) => (
            <div key={collection.name} className="mb-0.5">
              <button
                onClick={() => handleCollectionClick(collection.name)}
                className="w-full flex items-center justify-between px-3 py-2 rounded-lg transition-all duration-150 text-left"
                style={{
                  background:
                    expandedCollection === collection.name
                      ? "var(--bg-surface)"
                      : "transparent",
                  color: "var(--text-primary)",
                }}
                onMouseEnter={(e) => {
                  if (expandedCollection !== collection.name)
                    e.currentTarget.style.background = "var(--bg-surface)";
                }}
                onMouseLeave={(e) => {
                  if (expandedCollection !== collection.name)
                    e.currentTarget.style.background = "transparent";
                }}
              >
                <div className="flex items-center gap-2">
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={1.5}
                    viewBox="0 0 24 24"
                    style={{ color: "var(--text-muted)" }}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                    />
                  </svg>
                  <span className="text-[13px]">{collection.name}</span>
                </div>
                <ChevronDownIcon
                  className={`h-3.5 w-3.5 transition-transform`}
                  style={{
                    color: "var(--text-muted)",
                    transform:
                      expandedCollection === collection.name
                        ? "rotate(180deg)"
                        : "rotate(0deg)",
                  }}
                />
              </button>

              {expandedCollection === collection.name && (
                <div className="mt-0.5 ml-2">
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

      {/* Dropdown */}
      {!pdfSelectionMode && dropdownOpen && dropdownButtonRef && (
        <CollectionDropdown
          isOpen={true}
          onClose={() => {
            setDropdownOpen(null);
            setDropdownButtonRef(null);
          }}
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
