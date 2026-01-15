// // src/components/Chat/SourcesDisplay.tsx
// "use client";

// import { useState } from "react";
// import { DocumentSource } from "@/lib/types/message";
// import {
//   ChevronDownIcon,
//   ChevronUpIcon,
//   DocumentTextIcon,
//   ArrowTopRightOnSquareIcon,
//   FolderIcon,
// } from "@heroicons/react/24/outline";
// import Badge from "../UI/Badge";
// import { collectionsApi } from "@/lib/api/collections";
// import { useToast } from "@/hooks/useToast";


// interface SourcesDisplayProps {
//   sources: DocumentSource[];
// }

// export default function SourcesDisplay({ sources }: SourcesDisplayProps) {
//   const [isExpanded, setIsExpanded] = useState(false);
//   const [expandedDocuments, setExpandedDocuments] = useState<Set<string>>(
//     new Set()
//   );
//   const toast = useToast();

//   if (!sources || sources.length === 0) return null;

//   // Group sources by filename
//   const groupedSources = sources.reduce((acc, source) => {
//     const key = source.filename;
//     if (!acc[key]) {
//       acc[key] = [];
//     }
//     acc[key].push(source);
//     return acc;
//   }, {} as Record<string, DocumentSource[]>);

//   const toggleDocumentExpansion = (filename: string) => {
//     setExpandedDocuments((prev) => {
//       const newSet = new Set(prev);
//       if (newSet.has(filename)) {
//         newSet.delete(filename);
//       } else {
//         newSet.add(filename);
//       }
//       return newSet;
//     });
//   };

//   // Handle PDF view
//   const handleViewPdf = (collectionName: string, filename: string) => {
//     try {
//       collectionsApi.viewPDF(collectionName, filename);
//       toast.success(`Opening "${filename}"...`);
//     } catch (error) {
//       console.error("Failed to open PDF:", error);
//       toast.error("Failed to open PDF");
//     }
//   };

//   // Handle viewing PDF at specific page
//   const handleViewPdfAtPage = (
//     collectionName: string,
//     filename: string,
//     pageNumbers: string
//   ) => {
//     try {
//       // Parse the first page number from the string
//       const cleanPageNumbers = pageNumbers.replace(/[\[\]']/g, "");
//       const firstPage = cleanPageNumbers.split(",")[0]?.trim();

//       if (firstPage) {
//         // Get the PDF URL and append page parameter
//         const baseUrl = collectionsApi.getPDFUrl(collectionName, filename);
//         const urlWithPage = `${baseUrl}#page=${firstPage}`;

//         // Open PDF at specific page in new tab
//         window.open(urlWithPage, "_blank", "noopener,noreferrer");
//         toast.success(`Opening "${filename}" at page ${firstPage}...`);
//       } else {
//         // Fallback to opening without page number
//         handleViewPdf(collectionName, filename);
//       }
//     } catch (error) {
//       console.error("Failed to open PDF at page:", error);
//       toast.error("Failed to open PDF at page");
//     }
//   };

//   return (
//     <div className="mt-3 pt-3 border-t border-gray-200">
//       <button
//         onClick={() => setIsExpanded(!isExpanded)}
//         className="flex items-center text-sm text-gray-600 hover:text-gray-900 transition-colors"
//       >
//         <DocumentTextIcon className="h-4 w-4 mr-1" />
//         <span className="font-medium">
//           Sources ({Object.keys(groupedSources).length} documents)
//         </span>
//         {isExpanded ? (
//           <ChevronUpIcon className="h-4 w-4 ml-1" />
//         ) : (
//           <ChevronDownIcon className="h-4 w-4 ml-1" />
//         )}
//       </button>

//       {isExpanded && (
//         <div className="mt-2 space-y-3 max-h-96 overflow-y-auto">
//           {Object.entries(groupedSources).map(([filename, fileSources]) => {
//             const isDocExpanded = expandedDocuments.has(filename);
//             const displayedSources = isDocExpanded
//               ? fileSources
//               : fileSources.slice(0, 3);
//             const collectionName = fileSources[0].collection || "";

//             return (
//               <div key={filename} className="bg-gray-50 rounded-lg p-3">
//                 <div className="flex items-start justify-between mb-2">
//                   {/* Clickable PDF filename */}
//                   <button
//                     onClick={() => handleViewPdf(collectionName, filename)}
//                     className="font-medium text-sm text-blue-600 transition-colors flex items-center gap-1 group"
//                     title="Click to open PDF"
//                   >
//                     <span className="group-hover:underline">{filename}</span>
//                     <ArrowTopRightOnSquareIcon className="h-3.5 w-3.5  group-hover:opacity-100 transition-opacity" />
//                   </button>

//                   {collectionName && (
//                     <Badge variant="info" size="sm">
//                       {collectionName}
//                     </Badge>
//                   )}
//                 </div>

//                 {fileSources[0].title &&
//                   fileSources[0].title !== "No Title" && (
//                     <p className="text-xs text-gray-600 mb-2">
//                       {fileSources[0].title}
//                     </p>
//                   )}

//                 <div className="space-y-2">
//                   {displayedSources.map((source, idx) => (
//                     <div key={idx} className="text-xs">
//                       <div className="flex items-center justify-between mb-1">
//                         <span className="text-gray-500">
//                           Match: {(source.similarity * 100).toFixed(1)}%
//                         </span>
//                         {(() => {
//                           const pages = source.page_numbers;

//                           if (!pages || pages === "[]") return null;

//                           return (
//                             <button
//                               onClick={() =>
//                                 handleViewPdfAtPage(
//                                   collectionName,
//                                   filename,
//                                   pages 
//                                 )
//                               }
//                               className="text-blue-600 hover:text-blue-800 hover:underline transition-colors"
//                               title="Click to open PDF at this page"
//                             >
//                               Pages: {pages.replace(/[\[\]']/g, "")}
//                             </button>
//                           );
//                         })()}
//                       </div>
//                       <p className="text-gray-700 leading-relaxed">
//                         {source.content}
//                       </p>
//                     </div>
//                   ))}

//                   {fileSources.length > 3 && (
//                     <button
//                       onClick={() => toggleDocumentExpansion(filename)}
//                       className="w-full text-xs text-blue-600 hover:text-blue-800 hover:bg-blue-50 py-2 rounded transition-colors font-medium"
//                     >
//                       {isDocExpanded ? (
//                         <span className="flex items-center justify-center gap-1">
//                           <ChevronUpIcon className="h-3 w-3" />
//                           Show less
//                         </span>
//                       ) : (
//                         `+${fileSources.length - 3} more excerpts`
//                       )}
//                     </button>
//                   )}
//                 </div>
//               </div>
//             );
//           })}
//         </div>
//       )}
//     </div>
//   );
// }

// src/components/Chat/SourcesDisplay.tsx
"use client";

import { useState } from "react";
import { DocumentSource } from "@/lib/types/message";
import {
  ChevronDownIcon,
  ChevronUpIcon,
  DocumentTextIcon,
  ArrowTopRightOnSquareIcon,
  FolderIcon,
} from "@heroicons/react/24/outline";
import Badge from "../UI/Badge";
import { collectionsApi } from "@/lib/api/collections";
import { useToast } from "@/hooks/useToast";

interface SourcesDisplayProps {
  sources: DocumentSource[];
}

export default function SourcesDisplay({ sources }: SourcesDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandedDocuments, setExpandedDocuments] = useState<Set<string>>(
    new Set()
  );
  const toast = useToast();

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

  const toggleDocumentExpansion = (filename: string) => {
    setExpandedDocuments((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(filename)) {
        newSet.delete(filename);
      } else {
        newSet.add(filename);
      }
      return newSet;
    });
  };

  // Handle PDF view
  const handleViewPdf = (collectionName: string, filename: string) => {
    try {
      collectionsApi.viewPDF(collectionName, filename);
      toast.success(`Opening "${filename}"...`);
    } catch (error) {
      console.error("Failed to open PDF:", error);
      toast.error("Failed to open PDF");
    }
  };

  // Handle viewing PDF at specific page
  const handleViewPdfAtPage = (
    collectionName: string,
    filename: string,
    pageNumbers: string
  ) => {
    try {
      // Parse the first page number from the string
      const cleanPageNumbers = pageNumbers.replace(/[\[\]']/g, "");
      const firstPage = cleanPageNumbers.split(",")[0]?.trim();

      if (firstPage) {
        // Get the PDF URL and append page parameter
        const baseUrl = collectionsApi.getPDFUrl(collectionName, filename);
        const urlWithPage = `${baseUrl}#page=${firstPage}`;

        // Open PDF at specific page in new tab
        window.open(urlWithPage, "_blank", "noopener,noreferrer");
        toast.success(`Opening "${filename}" at page ${firstPage}...`);
      } else {
        // Fallback to opening without page number
        handleViewPdf(collectionName, filename);
      }
    } catch (error) {
      console.error("Failed to open PDF at page:", error);
      toast.error("Failed to open PDF at page");
    }
  };

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
          {Object.entries(groupedSources).map(([filename, fileSources]) => {
            const isDocExpanded = expandedDocuments.has(filename);
            const displayedSources = isDocExpanded
              ? fileSources
              : fileSources.slice(0, 3);
            const collectionName = fileSources[0].collection || "";

            return (
              <div key={filename} className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex flex-col gap-1.5 flex-1 min-w-0">
                    {/* Clickable PDF filename */}
                    <button
                      onClick={() => handleViewPdf(collectionName, filename)}
                      className="font-medium text-sm text-blue-600 transition-colors flex items-center gap-1 group text-left"
                      title="Click to open PDF"
                    >
                      <span className="group-hover:underline truncate">
                        {filename}
                      </span>
                      <ArrowTopRightOnSquareIcon className="h-3.5 w-3.5 flex-shrink-0 group-hover:opacity-100 transition-opacity" />
                    </button>

                    {/* Collection badge with folder icon */}
                    {collectionName && (
                      <div className="flex items-center gap-1.5 text-xs text-black bg-stone-50 rounded-md w-fit">
                        <FolderIcon className="h-3.5 w-3.5 text-black" />
                        <span className="font-medium">Collection Name: {collectionName}</span>
                      </div>
                    )}
                  </div>
                </div>

                {fileSources[0].title &&
                  fileSources[0].title !== "No Title" && (
                    <p className="text-xs text-gray-600 mb-2">
                      {fileSources[0].title}
                    </p>
                  )}

                <div className="space-y-2">
                  {displayedSources.map((source, idx) => (
                    <div key={idx} className="text-xs">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-gray-500">
                          Match: {(source.similarity * 100).toFixed(1)}%
                        </span>
                        {(() => {
                          const pages = source.page_numbers;

                          if (!pages || pages === "[]") return null;

                          return (
                            <button
                              onClick={() =>
                                handleViewPdfAtPage(
                                  collectionName,
                                  filename,
                                  pages
                                )
                              }
                              className="text-blue-600 hover:text-blue-800 hover:underline transition-colors"
                              title="Click to open PDF at this page"
                            >
                              Pages: {pages.replace(/[\[\]']/g, "")}
                            </button>
                          );
                        })()}
                      </div>
                      <p className="text-gray-700 leading-relaxed">
                        {source.content}
                      </p>
                    </div>
                  ))}

                  {fileSources.length > 3 && (
                    <button
                      onClick={() => toggleDocumentExpansion(filename)}
                      className="w-full text-xs text-blue-600 hover:text-blue-800 hover:bg-blue-50 py-2 rounded transition-colors font-medium"
                    >
                      {isDocExpanded ? (
                        <span className="flex items-center justify-center gap-1">
                          <ChevronUpIcon className="h-3 w-3" />
                          Show less
                        </span>
                      ) : (
                        `+${fileSources.length - 3} more excerpts`
                      )}
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}