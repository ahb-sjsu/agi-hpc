"use client";

import {
  ParticipantTile,
  useEnsureTrackRef,
  useMultibandTrackVolume,
  useTracks,
  type TrackReferenceOrPlaceholder,
} from "@livekit/components-react";
import { Track, type Participant } from "livekit-client";

/**
 * PlayerSlot — fixed-position cell for one participant.
 *
 * The 4x3 grid layout assigns specific roles to fixed cells:
 * P1..P7 are player slots, GM is the keeper slot, ARTEMIS/SIGMA
 * are AI slots. This component renders ONE such cell, given the
 * participant chosen for it (or a placeholder if empty).
 *
 * Why not LiveKit's GridLayout? GridLayout reflows tiles
 * dynamically. We want fixed positions: P1 is always P1, even if
 * someone disconnects. That stability matters for a tabletop
 * setting where the GM is calling on players by their on-screen
 * position.
 */
export default function PlayerSlot({
  participant,
  label,
  highlight = false,
  fallbackText,
}: {
  participant: Participant | null;
  label: string;
  highlight?: boolean;
  fallbackText?: string;
}) {
  if (!participant) {
    return (
      <EmptySlot
        label={label}
        text={fallbackText ?? "(open)"}
        highlight={highlight}
      />
    );
  }
  return (
    <PopulatedSlot
      participant={participant}
      label={label}
      highlight={highlight}
    />
  );
}

function EmptySlot({
  label,
  text,
  highlight,
}: {
  label: string;
  text: string;
  highlight: boolean;
}) {
  return (
    <div
      className={[
        "relative h-full w-full rounded-md overflow-hidden",
        "bg-surface border",
        highlight ? "border-accent" : "border-border",
        "flex items-center justify-center",
      ].join(" ")}
      aria-label={`${label}: empty slot`}
    >
      <span className="text-text-muted font-mono text-xs">{text}</span>
      <SlotLabel label={label} highlight={highlight} />
    </div>
  );
}

function PopulatedSlot({
  participant,
  label,
  highlight,
}: {
  participant: Participant;
  label: string;
  highlight: boolean;
}) {
  // Find the camera trackRef for this specific participant. If
  // none exists yet, we still render a tile (LiveKit handles the
  // "no camera" state with an avatar placeholder).
  const cameraRefs = useTracks([
    { source: Track.Source.Camera, withPlaceholder: true },
  ]);
  const ownCamera = cameraRefs.find(
    (t) => t.participant.identity === participant.identity,
  );

  if (!ownCamera) {
    return (
      <EmptySlot
        label={label}
        text={participant.name ?? participant.identity}
        highlight={highlight}
      />
    );
  }

  return (
    <div
      className={[
        "relative h-full w-full rounded-md overflow-hidden",
        "border",
        highlight ? "border-accent" : "border-border",
      ].join(" ")}
    >
      <ParticipantTile trackRef={ownCamera} />
      <MicMeter participant={participant} />
      <SlotLabel label={label} highlight={highlight} />
    </div>
  );
}

function SlotLabel({
  label,
  highlight,
}: {
  label: string;
  highlight: boolean;
}) {
  return (
    <span
      className={[
        "pointer-events-none absolute top-1.5 right-1.5",
        "px-1.5 py-0.5 rounded",
        "font-mono text-[10px] uppercase tracking-wider",
        highlight
          ? "bg-accent/80 text-bg"
          : "bg-bg/70 text-text-dim border border-border",
      ].join(" ")}
    >
      {label}
    </span>
  );
}

function MicMeter({ participant }: { participant: Participant }) {
  const micTracks = useTracks([Track.Source.Microphone]);
  const ownMic = micTracks.find(
    (t) => t.participant.identity === participant.identity,
  );
  const muted = !ownMic || !ownMic.publication || ownMic.publication.isMuted;

  return (
    <div
      aria-live="off"
      className="pointer-events-none absolute bottom-2 left-2 flex items-center gap-1.5 rounded bg-bg/60 px-1.5 py-1 backdrop-blur-sm"
      title={muted ? "microphone muted" : "microphone active"}
    >
      <MicGlyph muted={muted} />
      {ownMic && !muted ? (
        <Bars trackRef={ownMic} />
      ) : (
        <span className="text-[9px] font-mono uppercase tracking-wider text-text-muted">
          muted
        </span>
      )}
    </div>
  );
}

function Bars({ trackRef }: { trackRef: TrackReferenceOrPlaceholder }) {
  const volumes = useMultibandTrackVolume(trackRef, {
    bands: 5,
    updateInterval: 100,
  });
  return (
    <div
      role="meter"
      aria-label="microphone level"
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuenow={Math.max(...volumes, 0)}
      className="flex items-end gap-[2px] h-3"
    >
      {volumes.map((v, i) => {
        const scaled = Math.min(1, Math.pow(v, 0.6));
        const heightPct = Math.max(12, scaled * 100);
        const tone =
          scaled < 0.15 ? "bg-text-muted" : scaled < 0.6 ? "bg-ok" : "bg-warn";
        return (
          <span
            key={i}
            className={`inline-block w-[2px] rounded-sm ${tone}`}
            style={{ height: `${heightPct}%` }}
          />
        );
      })}
    </div>
  );
}

function MicGlyph({ muted }: { muted: boolean }) {
  if (muted) {
    return (
      <svg
        viewBox="0 0 24 24"
        width="12"
        height="12"
        aria-hidden="true"
        className="text-err"
      >
        <path
          fill="currentColor"
          d="M12 2a3 3 0 0 0-3 3v4.7L5.3 6a9 9 0 0 0 2 10.8l1.4-1.4A7 7 0 0 1 6.9 8.3L9 10.4V11a3 3 0 0 0 4.9 2.3l2 2a5 5 0 0 1-2.9.8 5 5 0 0 1-5-5H6a7 7 0 0 0 6 7v2h-3v2h8v-2h-3v-2a7 7 0 0 0 2.6-.7l2.1 2.1L20 17.3 3.7 1zM12 14a3 3 0 0 1-3-3v-.2l3 3z"
        />
      </svg>
    );
  }
  return (
    <svg
      viewBox="0 0 24 24"
      width="12"
      height="12"
      aria-hidden="true"
      className="text-text-dim"
    >
      <path
        fill="currentColor"
        d="M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3zm5 9a5 5 0 0 1-10 0H5a7 7 0 0 0 6 7v2H7v2h10v-2h-4v-2a7 7 0 0 0 6-7z"
      />
    </svg>
  );
}
