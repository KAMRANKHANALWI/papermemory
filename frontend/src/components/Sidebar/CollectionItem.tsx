// src/components/Sidebar/CollectionItem.tsx
"use client";

import { Collection } from "@/lib/types/collection";
import { FolderIcon, TrashIcon, PencilIcon } from "@heroicons/react/24/outline";
import Badge from "../UI/Badge";

interface CollectionItemProps {
  collection: Collection;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onManage: () => void;
}

export default function CollectionItem({
  collection,
  isSelected,
  onSelect,
  onDelete,
  onManage,
}: CollectionItemProps) {
  return (
    <div
      className={`
        group p-3 rounded-lg cursor-pointer transition-all
        ${
          isSelected
            ? "bg-amber-50 border-2 border-amber-500"
            : "bg-white border-2 border-gray-200 hover:border-amber-300"
        }
      `}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-2 flex-1 min-w-0">
          <FolderIcon className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <h3 className="font-medium text-gray-900 truncate text-sm">
              {collection.name}
            </h3>
            <div className="flex items-center space-x-2 mt-1">
              <Badge variant="info" size="sm">
                {collection.file_count} files
              </Badge>
              <Badge variant="default" size="sm">
                {collection.chunk_count} chunks
              </Badge>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onManage();
            }}
            className="p-1 text-gray-400 hover:text-amber-600 rounded transition-colors"
            title="Manage collection"
          >
            <PencilIcon className="h-4 w-4" />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            className="p-1 text-gray-400 hover:text-red-600 rounded transition-colors"
            title="Delete collection"
          >
            <TrashIcon className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
