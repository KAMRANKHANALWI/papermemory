// src/components/Sidebar/UploadButton.tsx
"use client";

import { useRef } from "react";
import { CloudArrowUpIcon } from "@heroicons/react/24/outline";

interface UploadButtonProps {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
}

export default function UploadButton({
  onFilesSelected,
  disabled = false,
}: UploadButtonProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleClick = () => {
    if (!disabled) {
      fileInputRef.current?.click();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      onFilesSelected(files);
    }
    // Reset input so same files can be selected again
    e.target.value = "";
  };

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf"
        onChange={handleFileChange}
        className="hidden"
        disabled={disabled}
      />
      <button
        onClick={handleClick}
        disabled={disabled}
        className={`
          w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[15px]
          transition-all duration-150 cursor-pointer active:bg-stone-300
          ${
            disabled
              ? "text-gray-400 cursor-not-allowed"
              : "text-black hover:bg-stone-100"
          }
        `}
      >
        <CloudArrowUpIcon className="w-6 h-6 flex-shrink-0" strokeWidth={1.5} />
        <span>Upload PDFs</span>
      </button>
    </>
  );
}
