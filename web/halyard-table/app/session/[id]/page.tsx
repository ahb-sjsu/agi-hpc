"use client";

import { LiveKitRoom } from "@livekit/components-react";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import CharacterSheetDrawer from "@/components/CharacterSheetDrawer";
import MediaControls from "@/components/MediaControls";
import ScreenShareOverlay from "@/components/ScreenShareOverlay";
import TableGrid from "@/components/TableGrid";
import { mintToken } from "@/lib/livekit";

interface Params {
  id: string;
}

/**
 * Session route.
 *
 * - Reads identity/name from query string (landing page sets
 *   them; fresh browsers skipping the landing page just have
 *   "guest" identities).
 * - Mints a LiveKit JWT and connects to the room named after
 *   the session id.
 * - Renders the main three-panel layout: video grid on the
 *   left, AI chat on the right, safety bar at the bottom.
 * - Hotkey ``c`` toggles the character sheet drawer.
 *
 * Next.js 14 passes ``params`` as a plain object to route
 * components (the Promise-wrapped form is Next.js 15+). The old
 * code typed it as Promise and fed it through React.use(), which
 * threw "expected a Promise" at runtime. Direct prop access is
 * the 14.x shape.
 */
export default function SessionPage({
  params,
}: {
  params: Params;
}) {
  const { id: sessionId } = params;
  // useSearchParams requires a Suspense boundary during SSR —
  // wrap the body in a Suspense fallback.
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center">
          <p className="font-mono text-text-dim">Loading {sessionId}…</p>
        </div>
      }
    >
      <SessionBody sessionId={sessionId} />
    </Suspense>
  );
}

function SessionBody({ sessionId }: { sessionId: string }) {
  const search = useSearchParams();
  const displayName = search.get("name") ?? "guest";
  const pcId = search.get("pc");
  const password = search.get("pw") ?? "";

  const [token, setToken] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { token, url } = await mintToken({
          sessionId,
          identity: pcId ?? displayName.toLowerCase().replace(/\s+/g, "-"),
          displayName,
          password,
        });
        if (cancelled) return;
        setToken(token);
        setUrl(url);
      } catch (e) {
        if (!cancelled) setErr(String((e as Error).message ?? e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sessionId, displayName, pcId, password]);

  // Global 'c' hotkey to toggle the sheet drawer.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Skip if the user's typing in a field.
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      ) {
        return;
      }
      if (e.key === "c" || e.key === "C") {
        setDrawerOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (err) {
    return (
      <div
        role="alert"
        className="min-h-screen flex items-center justify-center px-4"
      >
        <div className="max-w-md bg-surface border border-err rounded-lg p-6">
          <h2 className="font-mono text-err mb-2">Could not join session.</h2>
          <p className="text-sm text-text-dim font-mono mb-4">{err}</p>
          <p className="text-xs text-text-muted font-mono">
            Check that halyard-keeper-backend is reachable at the
            configured token-mint URL, and that the session id is
            valid.
          </p>
        </div>
      </div>
    );
  }

  if (!token || !url) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="font-mono text-text-dim">Joining {sessionId}…</p>
      </div>
    );
  }

  return (
    <LiveKitRoom
      serverUrl={url}
      token={token}
      connect
      video
      audio
      className="min-h-screen flex flex-col"
    >
      <TopBar sessionId={sessionId} displayName={displayName} />
      <TableGrid sessionId={sessionId} />
      <CharacterSheetDrawer
        sessionId={sessionId}
        pcId={pcId}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      />
      <ScreenShareOverlay />
    </LiveKitRoom>
  );
}

function TopBar({
  sessionId,
  displayName,
}: {
  sessionId: string;
  displayName: string;
}) {
  return (
    <header className="flex items-center gap-4 px-4 py-2 border-b border-border bg-surface">
      <h1 className="font-mono text-accent text-sm">Halyard Table</h1>
      <span className="text-text-dim font-mono text-xs">
        session: <span className="text-text">{sessionId}</span>
      </span>
      <div className="ml-auto flex items-center gap-3">
        <MediaControls />
        <span className="text-text-dim font-mono text-xs">
          you: <span className="text-text">{displayName}</span>
        </span>
        <kbd className="text-text-muted font-mono text-[10px] border border-border rounded px-1.5 py-0.5">
          c · sheet
        </kbd>
      </div>
    </header>
  );
}

