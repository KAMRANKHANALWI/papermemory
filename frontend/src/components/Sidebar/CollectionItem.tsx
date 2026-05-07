// src/components/Sidebar/CollectionItem.tsx
"use client";

import { useRef } from "react";
import { Collection } from "@/lib/types/collection";
import { FolderIcon, EllipsisHorizontalIcon } from "@heroicons/react/24/outline";

interface CollectionItemProps {
  collection: Collection;
  isSelected: boolean;
  onSelect: () => void;
  onManage: (buttonRef: React.RefObject<HTMLButtonElement>) => void;
}

export default function CollectionItem({
  collection,
  isSelected,
  onSelect,
  onManage,
}: CollectionItemProps) {
  const buttonRef = useRef<HTMLButtonElement>(null);

  return (
    <div
      className="group relative px-2 py-2 rounded-lg cursor-pointer transition-all duration-150"
      style={{
        background: isSelected ? "var(--bg-surface)" : "transparent",
      }}
      onMouseEnter={e => {
        if (!isSelected) e.currentTarget.style.background = "var(--bg-surface)";
      }}
      onMouseLeave={e => {
        if (!isSelected) e.currentTarget.style.background = "transparent";
      }}
      onClick={onSelect}
    >
      <div className="flex items-center gap-2.5">
        <FolderIcon
          className="h-4 w-4 flex-shrink-0 transition-colors"
          style={{ color: isSelected ? "var(--text-accent)" : "var(--text-primary)" }}
        />

        <span
          className="flex-1 text-[14px] truncate"
          style={{
            color: isSelected ? "var(--text-primary)" : "var(--text-secondary)",
            fontWeight: isSelected ? 500 : 400,
          }}
        >
          {collection.name}
        </span>

        {/* Manage button */}
        <button
          ref={buttonRef}
          onClick={e => {
            e.stopPropagation();
            onManage(buttonRef as any);
          }}
          className="flex-shrink-0 p-1 rounded transition-all duration-150 opacity-0 group-hover:opacity-100"
          style={{ color: "var(--text-muted)" }}
          onMouseEnter={e => (e.currentTarget.style.color = "var(--text-secondary)")}
          onMouseLeave={e => (e.currentTarget.style.color = "var(--text-muted)")}
          title="Manage collection"
        >
          <EllipsisHorizontalIcon className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}