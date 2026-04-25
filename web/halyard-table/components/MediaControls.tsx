"use client";

import { useLocalParticipant } from "@livekit/components-react";
import { useState } from "react";

/**
 * MediaControls — mic mute and camera on/off toggles for the
 * local participant.
 *
 * In ``@livekit/components-react`` v2, ``useLocalParticipant``
 * exposes ``isMicrophoneEnabled`` / ``isCameraEnabled`` as
 * reactive values that update on track publish/mute. We rely on
 * those directly instead of subscribing to participant events
 * ourselves — that earlier pattern was the source of the buttons
 * looking active but not toggling visually.
 *
 * Click handler calls ``setMicrophoneEnabled`` /
 * ``setCameraEnabled`` on the local participant, which is the
 * canonical livekit-client method that handles publication,
 * permission prompts, and track reuse.
 */
export default function MediaControls() {
  const {
    localParticipant,
    isMicrophoneEnabled,
    isCameraEnabled,
  } = useLocalParticipant();
  const [busy, setBusy] = useState(false);

  const toggleMic = async () => {
    if (!localParticipant || busy) return;
    setBusy(true);
    try {
      await localParticipant.setMicrophoneEnabled(!isMicrophoneEnabled);
    } catch (err) {
      console.error("setMicrophoneEnabled failed:", err);
    } finally {
      setBusy(false);
    }
  };

  const toggleCam = async () => {
    if (!localParticipant || busy) return;
    setBusy(true);
    try {
      await localParticipant.setCameraEnabled(!isCameraEnabled);
    } catch (err) {
      console.error("setCameraEnabled failed:", err);
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
        label={isMicrophoneEnabled ? "Mute microphone" : "Unmute microphone"}
        active={isMicrophoneEnabled}
        onClick={toggleMic}
        disabled={busy || !localParticipant}
      >
        {isMicrophoneEnabled ? <MicIcon /> : <MicOffIcon />}
        <span className="hidden sm:inline">
          {isMicrophoneEnabled ? "mic" : "muted"}
        </span>
      </ControlButton>
      <ControlButton
        label={isCameraEnabled ? "Stop camera" : "Start camera"}
        active={isCameraEnabled}
        onClick={toggleCam}
        disabled={busy || !localParticipant}
      >
        {isCameraEnabled ? <CamIcon /> : <CamOffIcon />}
        <span className="hidden sm:inline">
          {isCameraEnabled ? "video" : "off"}
        </span>
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
