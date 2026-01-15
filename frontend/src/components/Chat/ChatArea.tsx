// // src/components/Chat/ChatArea.tsx
// "use client";

// import { Message } from "@/lib/types/message";
// import MessageList from "./MessageList";
// import ChatInput from "./ChatInput";
// import Badge from "../UI/Badge";

// interface ChatAreaProps {
//   messages: Message[];
//   isLoading: boolean;
//   selectedCollection: string | null;
//   chatMode: "single" | "chatall" | "selected";
//   onSendMessage: (message: string) => void;
//   // NEW: PDF Selection props
//   pdfSelectionMode?: boolean;
//   selectedPDFsCount?: number;
// }

// export default function ChatArea({
//   messages,
//   isLoading,
//   selectedCollection,
//   chatMode,
//   onSendMessage,
//   pdfSelectionMode = false,
//   selectedPDFsCount = 0,
// }: ChatAreaProps) {
//   const getPlaceholder = () => {
//     if (chatMode === "selected") {
//       if (selectedPDFsCount === 0) {
//         return "Select PDFs first...";
//       }
//       return "Ask about your selected PDFs...";
//     }
//     if (chatMode === "single" && !selectedCollection) {
//       return "Select a collection first...";
//     }
//     return "Ask about your documents...";
//   };

//   const isDisabled = 
//     (chatMode === "single" && !selectedCollection) || 
//     (chatMode === "selected" && selectedPDFsCount === 0) ||
//     isLoading;

//   // NEW: Get header info based on mode
//   const getHeaderInfo = () => {
//     if (chatMode === "selected") {
//       return {
//         title: "Selected PDFs",
//         subtitle: selectedPDFsCount > 0 
//           ? `Chatting with ${selectedPDFsCount} selected PDF${selectedPDFsCount > 1 ? 's' : ''}`
//           : "No PDFs selected - Select PDFs from the sidebar",
//       };
//     }
    
//     if (chatMode === "single") {
//       return {
//         title: selectedCollection || "Select a collection",
//         subtitle: selectedCollection
//           ? `Chatting with ${selectedCollection}`
//           : "Choose a collection to start",
//       };
//     }
    
//     return {
//       title: "Chat All Collections",
//       subtitle: "Search across all documents",
//     };
//   };

//   const headerInfo = getHeaderInfo();

//   return (
//     <div className="flex-1 flex flex-col h-full">
//       {/* Header */}
//       <div className="bg-white p-4 border-b border-gray-200">
//         <div className="flex justify-between items-center">
//           <div className="flex items-center gap-3">
//             <div>
//               <h2 className="font-semibold text-gray-900 flex items-center gap-2">
//                 {headerInfo.title}
//                 {/* NEW: Badge for selected PDFs mode */}
//                 {chatMode === "selected" && selectedPDFsCount > 0 && (
//                   <Badge variant="info" size="md">
//                     {selectedPDFsCount} PDF{selectedPDFsCount > 1 ? 's' : ''}
//                   </Badge>
//                 )}
//               </h2>
//               <p className="text-sm text-gray-500">
//                 {headerInfo.subtitle}
//               </p>
//             </div>
//           </div>

//           {isLoading && (
//             <div className="flex items-center text-amber-600 text-sm">
//               <div className="animate-spin w-4 h-4 border-2 border-amber-600 border-t-transparent rounded-full mr-2"></div>
//               Thinking...
//             </div>
//           )}
//         </div>
//       </div>

//       {/* Messages */}
//       <div className="flex-1 overflow-y-auto bg-gray-50">
//         <MessageList messages={messages} />
//       </div>

//       {/* Input */}
//       <ChatInput
//         onSend={onSendMessage}
//         disabled={isDisabled}
//         placeholder={getPlaceholder()}
//       />
//     </div>
//   );
// }

// src/components/Chat/ChatArea.tsx
"use client";

import { Message } from "@/lib/types/message";
import MessageList from "./MessageList";
import ChatInput from "./ChatInput";
import Badge from "../UI/Badge";

interface ChatAreaProps {
  messages: Message[];
  isLoading: boolean;
  selectedCollection: string | null;
  chatMode: "single" | "chatall" | "selected";
  onSendMessage: (message: string) => void;
  onStopGeneration?: () => void;
  pdfSelectionMode?: boolean;
  selectedPDFsCount?: number;
}

export default function ChatArea({
  messages,
  isLoading,
  selectedCollection,
  chatMode,
  onSendMessage,
  onStopGeneration,
  pdfSelectionMode = false,
  selectedPDFsCount = 0,
}: ChatAreaProps) {
  const getPlaceholder = () => {
    if (chatMode === "selected") {
      if (selectedPDFsCount === 0) {
        return "Select PDFs first...";
      }
      return "Ask about your selected PDFs...";
    }
    if (chatMode === "single" && !selectedCollection) {
      return "Select a collection first...";
    }
    return "Ask about your documents...";
  };

  const isDisabled = 
    (chatMode === "single" && !selectedCollection) || 
    (chatMode === "selected" && selectedPDFsCount === 0);

  const getHeaderInfo = () => {
    if (chatMode === "selected") {
      return {
        title: "Selected PDFs",
        subtitle: selectedPDFsCount > 0 
          ? `Chatting with ${selectedPDFsCount} selected PDF${selectedPDFsCount > 1 ? 's' : ''}`
          : "No PDFs selected - Select PDFs from the sidebar",
      };
    }
    
    if (chatMode === "single") {
      return {
        title: selectedCollection || "Select a collection",
        subtitle: selectedCollection
          ? `Chatting with ${selectedCollection}`
          : "Choose a collection to start",
      };
    }
    
    return {
      title: "Chat All Collections",
      subtitle: "Search across all documents",
    };
  };

  const headerInfo = getHeaderInfo();

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Header - Removed loading indicator from here */}
      <div className="bg-white p-4 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div>
              <h2 className="font-semibold text-gray-900 flex items-center gap-2">
                {headerInfo.title}
                {chatMode === "selected" && selectedPDFsCount > 0 && (
                  <Badge variant="info" size="md">
                    {selectedPDFsCount} PDF{selectedPDFsCount > 1 ? 's' : ''}
                  </Badge>
                )}
              </h2>
              <p className="text-sm text-gray-500">
                {headerInfo.subtitle}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto bg-gray-50">
        <MessageList messages={messages} />
      </div>

      {/* Input with loading state */}
      <ChatInput
        onSend={onSendMessage}
        onStop={onStopGeneration}
        disabled={isDisabled}
        isLoading={isLoading}
        placeholder={getPlaceholder()}
      />
    </div>
  );
}