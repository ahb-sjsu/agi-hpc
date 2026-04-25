"use client";

import { useLocalParticipant } from "@livekit/components-react";
import { useState } from "react";

/**
 * ScreenShareControl — toggle button for the GM's screen share.
 *
 * Mounted inside the GM's PlayerSlot via TableGrid's ``extraControl``
 * prop, and only when the LOCAL participant is the GM. Players see
 * the GM's video tile without this button.
 *
 * Click handler calls ``setScreenShareEnabled`` on the local
 * participant — LiveKit handles the browser permission prompt and
 * the actual track publication. When the share goes live, the
 * ScreenShareOverlay (mounted at the session-page level) listens
 * for the new track and takes over the viewport for everyone.
 */
export default function ScreenShareControl() {
  const { localParticipant, isScreenShareEnabled } = useLocalParticipant();
  const [busy, setBusy] = useState(false);

  const toggle = async () => {
    if (!localParticipant || busy) return;
    setBusy(true);
    try {
      await localParticipant.setScreenShareEnabled(!isScreenShareEnabled);
    } catch (err) {
      console.error("setScreenShareEnabled failed:", err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <button
      type="button"
      onClick={toggle}
      disabled={busy || !localParticipant}
      aria-label={
        isScreenShareEnabled ? "Stop screen share" : "Start screen share"
      }
      aria-pressed={isScreenShareEnabled}
      title={
        isScreenShareEnabled
          ? "Stop sharing your screen"
          : "Share your screen — visible full-window to all players"
      }
      className={[
        "flex items-center gap-1 px-2 py-1",
        "text-[10px] font-mono uppercase tracking-wider",
        "border rounded",
        "focus:outline-none focus:ring-2 focus:ring-offset-0",
        isScreenShareEnabled
          ? "border-warn bg-warn/15 text-warn hover:bg-warn/25 focus:ring-warn"
          : "border-accent bg-accent/15 text-accent hover:bg-accent/25 focus:ring-accent",
        "disabled:opacity-50 disabled:cursor-not-allowed",
      ].join(" ")}
    >
      <ScreenIcon active={isScreenShareEnabled} />
      <span>{isScreenShareEnabled ? "stop" : "share"}</span>
    </button>
  );
}

function ScreenIcon({ active }: { active: boolean }) {
  if (active) {
    return (
      <svg
        aria-hidden="true"
        width="12"
        height="12"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <line x1="2" y1="2" x2="22" y2="22" />
        <rect x="2" y="3" width="20" height="14" rx="2" />
      </svg>
    );
  }
  return (
    <svg
      aria-hidden="true"
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <line x1="8" y1="21" x2="16" y2="21" />
      <line x1="12" y1="17" x2="12" y2="21" />
    </svg>
  );
}
