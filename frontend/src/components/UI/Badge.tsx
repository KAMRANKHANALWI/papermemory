// src/components/UI/Badge.tsx
"use client";

interface BadgeProps {
  children: React.ReactNode;
  variant?: "default" | "success" | "warning" | "danger" | "info";
  size?: "sm" | "md" | "lg";
}

export default function Badge({ children, variant = "default", size = "md" }: BadgeProps) {
  const variantStyles: Record<string, React.CSSProperties> = {
    default: { background: "var(--bg-surface)", color: "var(--text-secondary)" },
    success: { background: "#EAF3DE", color: "#3B6D11" },
    warning: { background: "var(--amber-light)", color: "var(--amber)" },
    danger:  { background: "var(--danger-light)", color: "var(--danger)" },
    info:    { background: "var(--accent-light)", color: "var(--accent)" },
  };

  const sizeStyles = {
    sm: "px-1.5 py-0.5 text-[10px]",
    md: "px-2 py-0.5 text-[11px]",
    lg: "px-2.5 py-1 text-[12px]",
  };

  return (
    <span
      className={`inline-flex items-center font-medium rounded-full ${sizeStyles[size]}`}
      style={variantStyles[variant]}
    >
      {children}
    </span>
  );
}