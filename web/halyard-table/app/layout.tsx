import type { Metadata, Viewport } from "next";
import "./globals.css";
import "@livekit/components-styles";

export const metadata: Metadata = {
  title: "Halyard Table",
  description:
    "Live-play runtime for the Beyond the Heliopause Call of Cthulhu campaign.",
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: "#0a0e17",
  width: "device-width",
  initialScale: 1,
};

/**
 * Root layout.
 *
 * Kept deliberately lean — the only globals are fonts, the dark
 * theme, and the LiveKit component stylesheet. Session and keeper
 * routes own their own chrome below.
 */
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-text antialiased">
        {children}
      </body>
    </html>
  );
}
