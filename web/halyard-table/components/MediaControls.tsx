"use client";

import { useLocalParticipant } from "@livekit/components-react";
import { useEffect, useState } from "react";

/**
 * MediaControls — mic mute and camera on/off toggles for the
 * local participant. Shown in the session top bar.
 *
 * Reads live state from the local participant via LiveKit's
 * ``useLocalParticipant`` hook so the buttons reflect the actual
 * track state even when something else (Keeper, OS prompt) flips
 * it. Button click calls ``setMicrophoneEnabled`` /
 * ``setCameraEnabled`` directly — these are the canonical
 * participant-level toggles in livekit-client and they manage
 * track publication, permission prompts, and reuse.
 */
export default function MediaControls() {
  const { localParticipant } = useLocalParticipant();
  const [micOn, setMicOn] = useState(true);
  const [camOn, setCamOn] = useState(true);
  const [busy, setBusy] = useState(false);

  // Mirror the participant's actual track state so the buttons
  // never lie. ``isMicrophoneEnabled`` / ``isCameraEnabled`` are
  // booleans that flip when tracks are muted/unpublished.
  useEffect(() => {
    if (!localParticipant) return;
    const sync = () => {
      setMicOn(localParticipant.isMicrophoneEnabled);
      setCamOn(localParticipant.isCameraEnabled);
    };
    sync();
    localParticipant.on("trackMuted", sync);
    localParticipant.on("trackUnmuted", sync);
    localParticipant.on("trackPublished", sync);
    localParticipant.on("trackUnpublished", sync);
    return () => {
      localParticipant.off("trackMuted", sync);
      localParticipant.off("trackUnmuted", sync);
      localParticipant.off("trackPublished", sync);
      localParticipant.off("trackUnpublished", sync);
    };
  }, [localParticipant]);

  const toggleMic = async () => {
    if (!localParticipant || busy) return;
    setBusy(true);
    try {
      await localParticipant.setMicrophoneEnabled(!micOn);
    } finally {
      setBusy(false);
    }
  };

  const toggleCam = async () => {
    if (!localParticipant || busy) return;
    setBusy(true);
    try {
      await localParticipant.setCameraEnabled(!camOn);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      role="toolbar"
      aria-label="Media controls"
      className="flex items-center gap-1.5"
    >
      <ControlButton
        label={micOn ? "Mute microphone" : "Unmute microphone"}
        active={micOn}
        onClick={toggleMic}
        disabled={busy}
      >
        {micOn ? <MicIcon /> : <MicOffIcon />}
        <span className="hidden sm:inline">{micOn ? "mic" : "muted"}</span>
      </ControlButton>
      <ControlButton
        label={camOn ? "Stop camera" : "Start camera"}
        active={camOn}
        onClick={toggleCam}
        disabled={busy}
      >
        {camOn ? <CamIcon /> : <CamOffIcon />}
        <span className="hidden sm:inline">{camOn ? "video" : "off"}</span>
      </ControlButton>
    </div>
  );
}

function ControlButton({
  label,
  active,
  onClick,
  disabled,
  children,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      aria-pressed={active}
      title={label}
      className={[
        "flex items-center gap-1.5 px-2 py-1",
        "text-xs font-mono uppercase",
        "border rounded",
        "focus:outline-none focus:ring-2 focus:ring-offset-0",
        active
          ? "border-border bg-surface-2 text-text hover:border-accent focus:ring-accent"
          : "border-err bg-err/15 text-err hover:bg-err/25 focus:ring-err",
        "disabled:opacity-50 disabled:cursor-not-allowed",
      ].join(" ")}
    >
      {children}
    </button>
  );
}

// Inline SVGs avoid a runtime icon dependency; sized for the bar.
function MicIcon() {
  return (
    <svg
      aria-hidden="true"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 10v2a7 7 0 0 0 14 0v-2" />
      <path d="M12 19v3" />
    </svg>
  );
}

function MicOffIcon() {
  return (
    <svg
      aria-hidden="true"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="2" y1="2" x2="22" y2="22" />
      <path d="M9 9v3a3 3 0 0 0 5.12 2.12" />
      <path d="M15 10V5a3 3 0 0 0-6 0v1" />
      <path d="M19 10v2a7 7 0 0 1-.9 3.4" />
      <path d="M12 19v3" />
    </svg>
  );
}

function CamIcon() {
  return (
    <svg
      aria-hidden="true"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="2" y="6" width="14" height="12" rx="2" />
      <path d="M22 8 16 12 22 16Z" />
    </svg>
  );
}

function CamOffIcon() {
  return (
    <svg
      aria-hidden="true"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="2" y1="2" x2="22" y2="22" />
      <path d="M22 8 16 12 22 16Z" />
      <path d="M2 6h10v6" />
      <path d="M16 18H4a2 2 0 0 1-2-2V8" />
    </svg>
  );
}
