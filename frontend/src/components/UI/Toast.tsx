// src/components/UI/Toast.tsx
"use client";

import { useEffect } from "react";
import { Toast as ToastType } from "@/contexts/ToastContext";
import {
  XMarkIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  InformationCircleIcon,
  ExclamationTriangleIcon,
} from "@heroicons/react/24/solid";

interface ToastProps {
  toast: ToastType;
  onClose: () => void;
}

const toastConfig = {
  success: {
    icon: CheckCircleIcon,
    bg: "#EAF3DE",
    border: "#97C459",
    iconColor: "#3B6D11",
    textColor: "#27500A",
  },
  error: {
    icon: ExclamationCircleIcon,
    bg: "var(--danger-light)",
    border: "var(--danger)",
    iconColor: "var(--danger)",
    textColor: "var(--text-danger)",
  },
  warning: {
    icon: ExclamationTriangleIcon,
    bg: "var(--amber-light)",
    border: "var(--amber)",
    iconColor: "var(--amber)",
    textColor: "#633806",
  },
  info: {
    icon: InformationCircleIcon,
    bg: "var(--accent-light)",
    border: "var(--accent)",
    iconColor: "var(--accent)",
    textColor: "var(--text-accent)",
  },
};

export default function Toast({ toast, onClose }: ToastProps) {
  useEffect(() => {
    if (toast.duration && toast.duration > 0) {
      const t = setTimeout(onClose, toast.duration);
      return () => clearTimeout(t);
    }
  }, [toast.duration, onClose]);

  const cfg = toastConfig[toast.type];
  const Icon = cfg.icon;

  return (
    <div
      className="flex items-start gap-3 px-4 py-3 rounded-xl min-w-[300px] max-w-sm animate-slide-in border"
      style={{
        background: cfg.bg,
        borderColor: cfg.border,
        boxShadow: "0 4px 16px rgba(0,0,0,0.1)",
      }}
    >
      <Icon
        className="h-5 w-5 flex-shrink-0 mt-0.5"
        style={{ color: cfg.iconColor }}
      />
      <div className="flex-1 text-[13px]" style={{ color: cfg.textColor }}>
        {toast.message}
      </div>
      <button onClick={onClose} style={{ color: cfg.iconColor }}>
        <XMarkIcon className="h-4 w-4" />
      </button>

      {toast.type === "info" && toast.duration === 0 && (
        <div
          className="absolute bottom-0 left-0 right-0 h-0.5 mx-3 rounded-full animate-pulse"
          style={{ background: "var(--accent)" }}
        />
      )}
    </div>
  );
}
