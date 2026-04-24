"use client";

import {
  GridLayout,
  ParticipantTile,
  useParticipants,
  useTracks,
} from "@livekit/components-react";
import { RoomEvent, Track } from "livekit-client";

/**
 * VideoGrid — live participant tiles.
 *
 * Three behaviors specific to the Halyard Table:
 *
 * 1. AI participants (identity ``artemis`` or ``sigma-4``) are
 *    rendered as iconic tiles with no video track — the AIs ship
 *    text replies via DataChannel, not audio/video. Having them
 *    as visible tiles makes it obvious who's "at the table."
 *
 * 2. The Keeper is highlighted. Everyone else is uniform. The
 *    Keeper's tile has an accent-colored border so a player
 *    looking for the GM can find them instantly.
 *
 * 3. The grid refuses to render more than 12 tiles. Table cap is
 *    documented at 10 humans + 2 AIs + slack; hitting 12 in the
 *    grid means something is wrong upstream.
 */

const AI_IDENTITIES = new Set(["artemis", "sigma-4"]);

export default function VideoGrid() {
  const participants = useParticipants();
  const tracks = useTracks(
    [
      { source: Track.Source.Camera, withPlaceholder: true },
      { source: Track.Source.ScreenShare, withPlaceholder: false },
    ],
    { updateOnlyOn: [RoomEvent.ActiveSpeakersChanged] },
  );

  const totalCount = participants.length;
  if (totalCount === 0) {
    return (
      <div
        role="status"
        aria-live="polite"
        className="flex items-center justify-center h-full text-text-dim font-mono text-sm"
      >
        Waiting for the crew to arrive…
      </div>
    );
  }

  return (
    <div className="relative h-full">
      <GridLayout tracks={tracks} className="h-full">
        <ParticipantTile />
      </GridLayout>
      {totalCount > 12 && (
        <div
          role="alert"
          className="absolute top-2 right-2 bg-err/15 border border-err text-err text-xs font-mono px-2 py-1 rounded"
        >
          {totalCount} participants — over cap of 12
        </div>
      )}
    </div>
  );
}

export function isAiIdentity(identity: string): boolean {
  return AI_IDENTITIES.has(identity);
}
