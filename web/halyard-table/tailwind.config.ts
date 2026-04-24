import type { Config } from "tailwindcss";

// Mirrors the palette defined in
// infra/local/atlas-chat/schematic.html so the Halyard Table and the
// ops dashboard share a visual language. If the schematic ever adopts
// a design system, this file becomes the migration source.
const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0a0e17",
        surface: "#131a2b",
        "surface-2": "#1a2240",
        border: "#2a3555",
        accent: "#4a9eff",
        "accent-dim": "rgba(74, 158, 255, 0.15)",
        text: "#e0e6f0",
        "text-dim": "#7a8ba8",
        "text-muted": "#4a5568",
        ok: "#4ade80",
        warn: "#f59e0b",
        err: "#f87171",
        orange: "#fb923c",
      },
      fontFamily: {
        mono: [
          "Cascadia Code",
          "Fira Code",
          "JetBrains Mono",
          "ui-monospace",
          "monospace",
        ],
        sans: [
          "Segoe UI",
          "system-ui",
          "-apple-system",
          "sans-serif",
        ],
      },
    },
  },
  plugins: [],
};

export default config;
