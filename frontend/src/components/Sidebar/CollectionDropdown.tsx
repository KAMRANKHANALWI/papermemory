// src/components/Sidebar/CollectionDropdown.tsx
"use client";

import { useEffect, useRef } from "react";
import { PencilIcon, DocumentTextIcon, PlusCircleIcon, TrashIcon } from "@heroicons/react/24/outline";

interface CollectionDropdownProps {
  isOpen: boolean;
  onClose: () => void;
  collectionName: string;
  onRename: () => void;
  onListPDFs: () => void;
  onAddPDFs: () => void;
  onDelete: () => void;
  buttonRef: React.RefObject<HTMLButtonElement>;
}

export default function CollectionDropdown({
  isOpen, onClose, collectionName,
  onRename, onListPDFs, onAddPDFs, onDelete, buttonRef,
}: CollectionDropdownProps) {
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen) return;
    const handleClick = (e: MouseEvent) => {
      if (
        dropdownRef.current && !dropdownRef.current.contains(e.target as Node) &&
        buttonRef.current && !buttonRef.current.contains(e.target as Node)
      ) onClose();
    };
    const handleEsc = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleEsc);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleEsc);
    };
  }, [isOpen, onClose, buttonRef]);

  useEffect(() => {
    if (!isOpen || !buttonRef.current || !dropdownRef.current) return;
    const btn = buttonRef.current.getBoundingClientRect();
    const dd = dropdownRef.current.getBoundingClientRect();
    const vw = window.innerWidth, vh = window.innerHeight;
    let left = btn.right + 8;
    let top = btn.top;
    if (left + dd.width > vw - 8) left = btn.left - dd.width - 8;
    if (top + dd.height > vh - 8) top = Math.max(8, vh - dd.height - 8);
    dropdownRef.current.style.left = `${left}px`;
    dropdownRef.current.style.top = `${top}px`;
  }, [isOpen, buttonRef]);

  if (!isOpen) return null;

  const items = [
    { icon: PencilIcon, label: "Rename", onClick: onRename },
    { icon: DocumentTextIcon, label: "List PDFs", onClick: onListPDFs },
    { icon: PlusCircleIcon, label: "Add More PDFs", onClick: onAddPDFs },
  ];

  return (
    <>
      <div className="fixed inset-0 z-40" onClick={onClose} />
      <div
        ref={dropdownRef}
        className="fixed z-50 min-w-[188px] rounded-xl py-1.5 border"
        style={{
          background: "var(--bg-card)",
          borderColor: "var(--border-soft)",
          boxShadow: "0 4px 20px rgba(0,0,0,0.12)",
        }}
      >
        {items.map((item, i) => (
          <button
            key={i}
            onClick={() => { item.onClick(); onClose(); }}
            className="w-full flex items-center gap-3 px-3 py-2.5 text-[13px] transition-colors"
            style={{ color: "var(--text-secondary)" }}
            onMouseEnter={e => {
              e.currentTarget.style.background = "var(--bg-surface)";
              e.currentTarget.style.color = "var(--text-primary)";
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = "transparent";
              e.currentTarget.style.color = "var(--text-secondary)";
            }}
          >
            <item.icon className="h-4 w-4 flex-shrink-0" strokeWidth={1.5} />
            <span>{item.label}</span>
          </button>
        ))}

        <div className="my-1" style={{ borderTop: "0.5px solid var(--border-soft)" }} />

        <button
          onClick={() => { onDelete(); onClose(); }}
          className="w-full flex items-center gap-3 px-3 py-2.5 text-[13px] transition-colors"
          style={{ color: "var(--text-danger)" }}
          onMouseEnter={e => (e.currentTarget.style.background = "var(--danger-light)")}
          onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
        >
          <TrashIcon className="h-4 w-4 flex-shrink-0" strokeWidth={1.5} />
          <span>Delete</span>
        </button>
      </div>
    </>
  );
}