// src/components/Sidebar/CollectionList.tsx
"use client";

import { Collection } from "@/lib/types/collection";
import CollectionItem from "./CollectionItem";
import { MagnifyingGlassIcon } from "@heroicons/react/24/outline";
import { useState } from "react";

interface CollectionListProps {
  collections: Collection[];
  selectedCollection: string | null;
  onSelectCollection: (name: string) => void;
  onDeleteCollection: (name: string) => void;
  onManageCollection: (name: string) => void;
}

export default function CollectionList({
  collections,
  selectedCollection,
  onSelectCollection,
  onDeleteCollection,
  onManageCollection,
}: CollectionListProps) {
  const [searchQuery, setSearchQuery] = useState("");

  const filteredCollections = collections.filter((col) =>
    col.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Search */}
      <div className="p-3 border-b border-gray-200">
        <div className="relative">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search collections..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
          />
        </div>
      </div>

      {/* Header */}
      <div className="px-3 py-2 border-b border-gray-200">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
          Collections ({filteredCollections.length})
        </h3>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {filteredCollections.length === 0 ? (
          <div className="text-center py-8 text-gray-500 text-sm">
            {searchQuery ? (
              <>
                <p>No collections found</p>
                <p className="text-xs mt-1">Try a different search term</p>
              </>
            ) : (
              <>
                <p>No collections yet</p>
                <p className="text-xs mt-1">Upload PDFs to get started</p>
              </>
            )}
          </div>
        ) : (
          filteredCollections.map((collection) => (
            <CollectionItem
              key={collection.name}
              collection={collection}
              isSelected={selectedCollection === collection.name}
              onSelect={() => onSelectCollection(collection.name)}
              onDelete={() => onDeleteCollection(collection.name)}
              onManage={() => onManageCollection(collection.name)}
            />
          ))
        )}
      </div>
    </div>
  );
}
