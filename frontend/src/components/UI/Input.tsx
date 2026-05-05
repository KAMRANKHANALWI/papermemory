// src/components/UI/Input.tsx
"use client";

import { InputHTMLAttributes, forwardRef } from "react";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, helperText, className = "", style, ...props }, ref) => (
    <div className="w-full">
      {label && (
        <label className="block text-[12px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>
          {label}
        </label>
      )}
      <input
        ref={ref}
        className={`w-full px-3 py-2 rounded-lg border text-[13px] outline-none transition-all disabled:opacity-50 disabled:cursor-not-allowed ${className}`}
        style={{
          background: "var(--bg-input)",
          borderColor: error ? "var(--danger)" : "var(--border-soft)",
          color: "var(--text-primary)",
          ...style,
        }}
        onFocus={e => {
          if (!error) e.target.style.borderColor = "var(--accent)";
        }}
        onBlur={e => {
          e.target.style.borderColor = error ? "var(--danger)" : "var(--border-soft)";
        }}
        {...props}
      />
      {error && <p className="mt-1 text-[11px]" style={{ color: "var(--danger)" }}>{error}</p>}
      {helperText && !error && <p className="mt-1 text-[11px]" style={{ color: "var(--text-muted)" }}>{helperText}</p>}
    </div>
  )
);

Input.displayName = "Input";
export default Input;