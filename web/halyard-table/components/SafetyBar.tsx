"use client";

import { useCallback, useState } from "react";

/**
 * SafetyBar — always-visible row with the table's safety tools.
 *
 * X-card   → publishes ``safety_flag: x-card`` on the next turn,
 *            forcing both AIs to silence and asking the Keeper to
 *            pause the scene.
 * Pause    → a softer ask; scene continues but the AIs hold.
 * Open Door → marks the player as leaving; UI hides their video
 *            tile locally and the Keeper is notified.
 *
 * The component delegates the actual safety signal (NATS publish,
 * LiveKit DataChannel message, Keeper notification) to an
 * injected handler — the session page wires it up.
 */

export type SafetyAction = "x-card" | "pause" | "open-door";

interface SafetyBarProps {
  /**
   * Called with the requested action. Implementations should be
   * idempotent: the user may click multiple times.
   */
  onAction: (action: SafetyAction) => void;
}

export default function SafetyBar({ onAction }: SafetyBarProps) {
  const [lastAction, setLastAction] = useState<SafetyAction | null>(null);

  const fire = useCallback(
    (action: SafetyAction) => {
      setLastAction(action);
      onAction(action);
    },
    [onAction],
  );

  return (
    <div
      role="toolbar"
      aria-label="Safety tools"
      className="flex items-center gap-2 px-4 py-2 border-t border-border bg-surface/80 backdrop-blur"
    >
      <span className="text-xs text-text-muted font-mono mr-2 hidden sm:inline">
        Safety:
      </span>
      <SafetyButton
        label="X-Card"
        tone="err"
        active={lastAction === "x-card"}
        onClick={() => fire("x-card")}
        shortLabel="X"
        help="Pause the scene. AIs go silent. No explanation owed."
      />
      <SafetyButton
        label="Pause"
        tone="warn"
        active={lastAction === "pause"}
        onClick={() => fire("pause")}
        shortLabel="||"
        help="Hold the AIs; scene continues at the Keeper's pace."
      />
      <SafetyButton
        label="Open Door"
        tone="accent"
        active={lastAction === "open-door"}
        onClick={() => fire("open-door")}
        shortLabel="↴"
        help="Step away. No reason owed."
      />
      {lastAction && (
        <span
          role="status"
          aria-live="polite"
          className="ml-auto text-xs text-text-dim font-mono"
        >
          signal sent: {lastAction}
        </span>
      )}
    </div>
  );
}

interface SafetyButtonProps {
  label: string;
  shortLabel: string;
  tone: "err" | "warn" | "accent";
  active: boolean;
  onClick: () => void;
  help: string;
}

function SafetyButton({
  label,
  shortLabel,
  tone,
  active,
  onClick,
  help,
}: SafetyButtonProps) {
  const toneClass =
    tone === "err"
      ? "hover:bg-err/15 hover:border-err focus:border-err"
      : tone === "warn"
        ? "hover:bg-warn/15 hover:border-warn focus:border-warn"
        : "hover:bg-accent-dim hover:border-accent focus:border-accent";
  return (
    <button
      type="button"
      onClick={onClick}
      title={help}
      aria-label={`${label}: ${help}`}
      aria-pressed={active}
      className={`flex items-center gap-1 px-3 py-1.5 rounded-md border border-border bg-surface-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-offset-0 transition-colors ${toneClass} ${
        active ? "ring-2 ring-accent" : ""
      }`}
    >
      <span aria-hidden="true" className="w-4 text-center">
        {shortLabel}
      </span>
      <span>{label}</span>
    </button>
  );
}
