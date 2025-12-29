// src/components/Sidebar/UploadButton.tsx
"use client";

import { useRef } from "react";
import { CloudArrowUpIcon } from "@heroicons/react/24/outline";
import Button from "../UI/Button";

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
    fileInputRef.current?.click();
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
      />
      <Button
        onClick={handleClick}
        disabled={disabled}
        variant="primary"
        className="w-full"
      >
        <CloudArrowUpIcon className="h-5 w-5 mr-2" />
        Upload PDFs
      </Button>
    </>
  );
}
