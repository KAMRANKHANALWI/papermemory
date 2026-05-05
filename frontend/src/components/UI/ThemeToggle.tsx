// src/components/UI/ThemeToggle.tsx
"use client";

import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);

  // On mount, read saved preference
  useEffect(() => {
    const saved = localStorage.getItem("theme");
    const prefersDark = window.matchMedia(
      "(prefers-color-scheme: dark)",
    ).matches;
    const dark = saved === "dark" || (!saved && prefersDark);
    setIsDark(dark);
    document.documentElement.classList.toggle("dark", dark);
  }, []);

  const toggle = () => {
    const next = !isDark;
    setIsDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
  };

  return (
    <button
  onClick={toggle}
  className="relative w-8 h-4 rounded-full transition-all duration-300 flex items-center px-0.5 flex-shrink-0"
  style={{ background: isDark ? "var(--accent)" : "var(--border-soft)" }}
>
  <span
    className="w-3 h-3 rounded-full flex items-center justify-center transition-all duration-300"
    style={{
      background: "var(--bg-card)",
      transform: isDark ? "translateX(16px)" : "translateX(0px)",
    }}
  >
    {isDark ? (
      <svg viewBox="0 0 24 24" className="w-2 h-2" fill="currentColor" style={{ color: "var(--text-primary)" }}>
        <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>
      </svg>
    ) : (
      <svg viewBox="0 0 24 24" className="w-2 h-2" fill="none" stroke="currentColor" strokeWidth={2} style={{ color: "#BA7517" }}>
        <circle cx="12" cy="12" r="5"/>
        <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
        <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
      </svg>
    )}
  </span>
</button>
  );
}
