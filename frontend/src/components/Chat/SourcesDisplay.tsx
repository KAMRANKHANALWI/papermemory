// src/components/Chat/SourcesDisplay.tsx
"use client";

import { useState } from "react";
import { DocumentSource } from "@/lib/types/message";
import { ChevronDownIcon, ChevronUpIcon, DocumentTextIcon } from "@heroicons/react/24/outline";
import Badge from "../UI/Badge";

interface SourcesDisplayProps {
  sources: DocumentSource[];
}

export default function SourcesDisplay({ sources }: SourcesDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!sources || sources.length === 0) return null;

  // Group sources by filename
  const groupedSources = sources.reduce((acc, source) => {
    const key = source.filename;
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(source);
    return acc;
  }, {} as Record<string, DocumentSource[]>);

  return (
    <div className="mt-3 pt-3 border-t border-gray-200">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center text-sm text-gray-600 hover:text-gray-900 transition-colors"
      >
        <DocumentTextIcon className="h-4 w-4 mr-1" />
        <span className="font-medium">
          Sources ({Object.keys(groupedSources).length} documents)
        </span>
        {isExpanded ? (
          <ChevronUpIcon className="h-4 w-4 ml-1" />
        ) : (
          <ChevronDownIcon className="h-4 w-4 ml-1" />
        )}
      </button>

      {isExpanded && (
        <div className="mt-2 space-y-3 max-h-96 overflow-y-auto">
          {Object.entries(groupedSources).map(([filename, fileSources]) => (
            <div key={filename} className="bg-gray-50 rounded-lg p-3">
              <div className="flex items-start justify-between mb-2">
                <h4 className="font-medium text-sm text-gray-900">{filename}</h4>
                {fileSources[0].collection && (
                  <Badge variant="info" size="sm">
                    {fileSources[0].collection}
                  </Badge>
                )}
              </div>

              {fileSources[0].title && fileSources[0].title !== "No Title" && (
                <p className="text-xs text-gray-600 mb-2">{fileSources[0].title}</p>
              )}

              <div className="space-y-2">
                {fileSources.slice(0, 3).map((source, idx) => (
                  <div key={idx} className="text-xs">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-gray-500">
                        Match: {(source.similarity * 100).toFixed(1)}%
                      </span>
                      {source.page_numbers && source.page_numbers !== "[]" && (
                        <span className="text-gray-500">
                          Pages: {source.page_numbers.replace(/[\[\]']/g, "")}
                        </span>
                      )}
                    </div>
                    <p className="text-gray-700 leading-relaxed line-clamp-2">
                      {source.content}
                    </p>
                  </div>
                ))}

                {fileSources.length > 3 && (
                  <p className="text-xs text-gray-500 text-center pt-1">
                    +{fileSources.length - 3} more excerpts
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
