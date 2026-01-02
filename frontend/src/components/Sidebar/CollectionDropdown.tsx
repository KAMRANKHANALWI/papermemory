// src/components/Sidebar/CollectionDropdown.tsx
"use client";

import { useEffect, useRef } from "react";
import {
  PencilIcon,
  DocumentTextIcon,
  PlusCircleIcon,
  TrashIcon,
} from "@heroicons/react/24/outline";

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
  isOpen,
  onClose,
  collectionName,
  onRename,
  onListPDFs,
  onAddPDFs,
  onDelete,
  buttonRef,
}: CollectionDropdownProps) {
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close on click outside
  useEffect(() => {
    if (!isOpen || !buttonRef) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node) &&
        buttonRef.current &&
        !buttonRef.current.contains(event.target as Node)
      ) {
        onClose();
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleEscape);

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [isOpen, onClose, buttonRef]);

  // Calculate position
  useEffect(() => {
    if (!isOpen || !buttonRef || !buttonRef.current || !dropdownRef.current) return;

    const buttonRect = buttonRef.current.getBoundingClientRect();
    const dropdownRect = dropdownRef.current.getBoundingClientRect();
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;

    // Position to the right of the button
    let left = buttonRect.right + 8;
    let top = buttonRect.top;

    // If dropdown goes off screen horizontally, position to the left
    if (left + dropdownRect.width > viewportWidth - 8) {
      left = buttonRect.left - dropdownRect.width - 8;
    }

    // If dropdown goes off screen vertically, adjust
    if (top + dropdownRect.height > viewportHeight - 8) {
      top = Math.max(8, viewportHeight - dropdownRect.height - 8);
    }

    dropdownRef.current.style.left = `${left}px`;
    dropdownRef.current.style.top = `${top}px`;
  }, [isOpen, buttonRef]);

  if (!isOpen) return null;

  const menuItems = [
    {
      icon: PencilIcon,
      label: "Rename",
      onClick: () => {
        onRename();
        onClose();
      },
      danger: false,
    },
    {
      icon: DocumentTextIcon,
      label: "List PDFs",
      onClick: () => {
        onListPDFs();
        onClose();
      },
      danger: false,
    },
    {
      icon: PlusCircleIcon,
      label: "Add More PDFs",
      onClick: () => {
        onAddPDFs();
        onClose();
      },
      danger: false,
    },
  ];

  return (
    <>
      {/* Backdrop - subtle */}
      <div className="fixed inset-0 z-40" onClick={onClose} />

      {/* Dropdown Menu */}
      <div
        ref={dropdownRef}
        className="fixed z-50 min-w-[200px] bg-white rounded-xl shadow-lg border border-gray-200 py-1.5 animate-in fade-in-0 zoom-in-95 duration-100"
        style={{ transformOrigin: "top left" }}
      >
        {/* Menu Items */}
        {menuItems.map((item, index) => (
          <button
            key={index}
            onClick={item.onClick}
            className="w-full flex items-center gap-3 px-3 py-2.5 text-[14px] text-gray-700 hover:bg-gray-50 transition-colors duration-100"
          >
            <item.icon className="h-4 w-4 flex-shrink-0" strokeWidth={2} />
            <span>{item.label}</span>
          </button>
        ))}

        {/* Divider */}
        <div className="my-1.5 border-t border-gray-100" />

        {/* Delete Button - Danger Zone */}
        <button
          onClick={() => {
            onDelete();
            onClose();
          }}
          className="w-full flex items-center gap-3 px-3 py-2.5 text-[14px] text-red-600 hover:bg-red-50 transition-colors duration-100"
        >
          <TrashIcon className="h-4 w-4 flex-shrink-0" strokeWidth={2} />
          <span>Delete</span>
        </button>
      </div>
    </>
  );
}