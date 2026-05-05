// src/components/UI/Button.tsx
"use client";

import { ButtonHTMLAttributes, forwardRef } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "danger" | "ghost";
  size?: "sm" | "md" | "lg";
  isLoading?: boolean;
}

const variantStyles: Record<string, React.CSSProperties> = {
  primary:   { background: "var(--accent)", color: "#fff" },
  secondary: { background: "var(--bg-surface)", color: "var(--text-primary)", border: "0.5px solid var(--border-soft)" },
  danger:    { background: "var(--danger)", color: "#fff" },
  ghost:     { background: "transparent", color: "var(--text-secondary)" },
};

const hoverStyles: Record<string, React.CSSProperties> = {
  primary:   { background: "var(--accent-mid)" },
  secondary: { background: "var(--bg-base)" },
  danger:    { background: "#B03B1A" },
  ghost:     { background: "var(--bg-surface)" },
};

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ children, variant = "primary", size = "md", isLoading = false, disabled, className = "", style, ...props }, ref) => {
    const sizeClass = { sm: "px-3 py-1.5 text-[12px]", md: "px-4 py-2 text-[13px]", lg: "px-6 py-3 text-[15px]" }[size];

    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        className={`inline-flex items-center justify-center font-medium rounded-lg transition-all focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed ${sizeClass} ${className}`}
        style={{ ...variantStyles[variant], ...style }}
        onMouseEnter={e => {
          if (!disabled && !isLoading) Object.assign(e.currentTarget.style, hoverStyles[variant]);
        }}
        onMouseLeave={e => {
          Object.assign(e.currentTarget.style, variantStyles[variant]);
        }}
        {...props}
      >
        {isLoading ? (
          <>
            <svg className="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Loading...
          </>
        ) : children}
      </button>
    );
  }
);

Button.displayName = "Button";
export default Button;