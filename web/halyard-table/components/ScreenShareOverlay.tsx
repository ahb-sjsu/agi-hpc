"use client";

import {
  VideoTrack,
  useLocalParticipant,
  useTracks,
} from "@livekit/components-react";
import { Track } from "livekit-client";
import { useState } from "react";

/**
 * ScreenShareOverlay — full-viewport takeover when any participant
 * is sharing their screen. The Halyard Table treats screen share
 * as a GM-only "show this to everyone" mechanism; when the GM
 * starts sharing, the share dominates the window and the table
 * grid hides behind it.
 *
 * Behaviors:
 *  - The local sharer (i.e. the GM) sees a "stop sharing" button
 *    in the corner of the overlay.
 *  - Non-sharing participants see only the overlay; closing it
 *    isn't an option (the share IS the scene right now). When the
 *    sharer stops, the overlay tears down on its own.
 *  - If multiple shares somehow exist, we render the first one and
 *    log a warning. Practically: only the GM has the toggle button,
 *    so this should never happen at this table.
 *
 * Why a full-viewport overlay vs. inserting into the grid? The
 * shared screen carries text the Keeper wants the players to read
 * (PDFs, image handouts, references). Cell-sized rendering fights
 * that. The overlay is the right shape for the use case.
 */
export default function ScreenShareOverlay() {
  const screenShares = useTracks([Track.Source.ScreenShare]);
  const { localParticipant } = useLocalParticipant();
  const [stopping, setStopping] = useState(false);

  const active = screenShares.find((t) => !!t.publication?.track);
  if (!active) return null;

  const isLocalSharer =
    !!localParticipant &&
    active.participant.identity === localParticipant.identity;

  const stopShare = async () => {
    if (!localParticipant || stopping) return;
    setStopping(true);
    try {
      await localParticipant.setScreenShareEnabled(false);
    } catch (err) {
      console.error("stop share failed:", err);
    } finally {
      setStopping(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Screen share from ${
        active.participant.name ?? active.participant.identity
      }`}
      className="fixed inset-0 z-50 bg-bg flex flex-col"
    >
      <header className="flex items-center gap-3 px-4 py-2 border-b border-border bg-surface">
        <span className="font-mono text-accent text-sm">SCREEN SHARE</span>
        <span className="font-mono text-text-dim text-xs">
          from {active.participant.name ?? active.participant.identity}
        </span>
        {isLocalSharer && (
          <button
            type="button"
            onClick={stopShare}
            disabled={stopping}
            className="ml-auto flex items-center gap-1.5 px-2 py-1 text-xs font-mono uppercase border rounded border-err bg-err/15 text-err hover:bg-err/25 focus:outline-none focus:ring-2 focus:ring-err disabled:opacity-50"
          >
            stop sharing
          </button>
        )}
      </header>
      <div className="flex-1 min-h-0 relative bg-black">
        <VideoTrack
          trackRef={active}
          className="absolute inset-0 w-full h-full object-contain"
        />
      </div>
    </div>
  );
}
