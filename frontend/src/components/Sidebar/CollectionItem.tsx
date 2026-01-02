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
      className={`
        group relative px-3 py-2.5 rounded-lg cursor-pointer transition-all duration-150 active:bg-stone-300
        ${
          isSelected
            // ? "bg-[#BC6D51] hover:text-black"
            ? "bg-stone-200 hover:text-black"
            : "hover:text-black hover:bg-stone-200"
        }
      `}
      onClick={onSelect}
    >
      <div className="flex items-center gap-3">
        {/* Folder Icon - Properly aligned */}
        <FolderIcon 
          className={`h-5 w-5 flex-shrink-0 transition-colors ${
            isSelected ? "text-black" : "text-gray-400 group-hover:text-black"
          }`}
        />
        
        {/* Collection Name - Centered with icon */}
        <div className="flex-1 min-w-0 flex items-center">
          <span className={`text-[14px] leading-5 truncate font-normal ${
            isSelected ? "text-black font-medium" : "text-gray-700"
          }`}>
            {collection.name}
          </span>
        </div>

        {/* Manage Button - Shows on hover */}
        <button
          ref={buttonRef}
          onClick={(e) => {
            e.stopPropagation();
            onManage(buttonRef as any);
          }}
          className={`
            flex-shrink-0 p-1 rounded transition-all duration-150 cursor-pointer
            ${
              isSelected 
                ? "hover:bg-stone-300 text-black" 
                : "opacity-0 group-hover:opacity-100 text-gray-400 hover:text-gray-600"
            }
          `}
          title="Manage collection"
        >
          <EllipsisHorizontalIcon className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}