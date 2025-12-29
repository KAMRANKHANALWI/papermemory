// src/components/UI/Toast.tsx
"use client";

import { useEffect } from "react";
import { Toast as ToastType } from "@/contexts/ToastContext";
import { XMarkIcon } from "@heroicons/react/24/outline";
import {
  CheckCircleIcon,
  ExclamationCircleIcon,
  InformationCircleIcon,
  ExclamationTriangleIcon,
} from "@heroicons/react/24/solid";

interface ToastProps {
  toast: ToastType;
  onClose: () => void;
}

export default function Toast({ toast, onClose }: ToastProps) {
  useEffect(() => {
    if (toast.duration && toast.duration > 0) {
      const timer = setTimeout(onClose, toast.duration);
      return () => clearTimeout(timer);
    }
  }, [toast.duration, onClose]);

  const styles = {
    success: {
      bg: "bg-green-50 border-green-200",
      icon: "text-green-500",
      IconComponent: CheckCircleIcon,
    },
    error: {
      bg: "bg-red-50 border-red-200",
      icon: "text-red-500",
      IconComponent: ExclamationCircleIcon,
    },
    warning: {
      bg: "bg-yellow-50 border-yellow-200",
      icon: "text-yellow-500",
      IconComponent: ExclamationTriangleIcon,
    },
    info: {
      bg: "bg-blue-50 border-blue-200",
      icon: "text-blue-500",
      IconComponent: InformationCircleIcon,
    },
  };

  const style = styles[toast.type];
  const IconComponent = style.IconComponent;

  return (
    <div
      className={`${style.bg} border rounded-lg shadow-lg p-4 flex items-start space-x-3 min-w-[320px] max-w-md animate-slide-in`}
    >
      <IconComponent className={`h-5 w-5 ${style.icon} flex-shrink-0 mt-0.5`} />
      <div className="flex-1 text-sm text-gray-800">{toast.message}</div>
      <button
        onClick={onClose}
        className="text-gray-400 hover:text-gray-600 transition-colors flex-shrink-0"
      >
        <XMarkIcon className="h-5 w-5" />
      </button>
    </div>
  );
}
