// src/components/Sidebar/UploadButton.tsx
"use client";

import { useRef } from "react";
import { CloudArrowUpIcon } from "@heroicons/react/24/outline";

interface UploadButtonProps {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
}

export default function UploadButton({ onFilesSelected, disabled = false }: UploadButtonProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) onFilesSelected(files);
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
        onClick={() => !disabled && fileInputRef.current?.click()}
        disabled={disabled}
        className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[14px] transition-all duration-150"
        style={{
          color: disabled ? "var(--text-muted)" : "var(--text-secondary)",
          cursor: disabled ? "not-allowed" : "pointer",
        }}
        onMouseEnter={e => {
          if (!disabled) e.currentTarget.style.background = "var(--bg-surface)";
        }}
        onMouseLeave={e => {
          e.currentTarget.style.background = "transparent";
        }}
      >
        <CloudArrowUpIcon className="w-[18px] h-[18px] flex-shrink-0" strokeWidth={1.5} />
        <span>Upload PDFs</span>
      </button>
    </>
  );
}