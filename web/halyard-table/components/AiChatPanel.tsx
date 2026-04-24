"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useDataChannel } from "@livekit/components-react";

import { parseEnvelope } from "@/lib/livekit";
import type { ArtemisSay, Sigma4Say } from "@/lib/types";

type AiSay = ArtemisSay | Sigma4Say;

interface AiChatPanelProps {
  /**
   * Which AI this panel is for. Drives the envelope filter and the
   * pane's label; each AI gets its own panel so the Keeper can
   * mute one without losing the other.
   */
  ai: "artemis" | "sigma4";
  /** Max lines to retain. Older messages scroll out. */
  maxLines?: number;
}

/**
 * AiChatPanel — DataChannel-backed chat pane for one AI NPC.
 *
 * Listens on every DataChannel frame, filters to ``envelope.kind``
 * matching ``<ai>.say``, and renders each one as a stage-direction
 * line. Auto-scrolls to the newest message unless the operator
 * has scrolled up — common chat-log convention so a user reading
 * back doesn't get yanked back to the bottom.
 */
export default function AiChatPanel({
  ai,
  maxLines = 200,
}: AiChatPanelProps) {
  const [lines, setLines] = useState<AiSay[]>([]);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const stickRef = useRef(true);

  const targetKind = ai === "artemis" ? "artemis.say" : "sigma4.say";
  const label = ai === "artemis" ? "ARTEMIS" : "SIGMA-4";

  // Subscribe to the whole DataChannel — LiveKit's `useDataChannel`
  // invokes the callback for every publish_data frame.
  useDataChannel((msg) => {
    const env = parseEnvelope(msg.payload);
    if (!env || env.kind !== targetKind) return;
    setLines((prev) => {
      const next = [...prev, env as AiSay];
      return next.length > maxLines
        ? next.slice(next.length - maxLines)
        : next;
    });
  });

  // Autoscroll unless the user has scrolled up.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (stickRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [lines]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < 32;
    stickRef.current = nearBottom;
  };

  const empty = useMemo(() => lines.length === 0, [lines.length]);

  return (
    <section
      aria-labelledby={`ai-chat-${ai}-heading`}
      className="flex flex-col h-full border border-border rounded-md bg-surface overflow-hidden"
    >
      <header className="px-3 py-2 border-b border-border flex items-center gap-2">
        <h2
          id={`ai-chat-${ai}-heading`}
          className="text-sm font-mono text-accent tracking-wider"
        >
          {label}
        </h2>
        <span
          className="text-xs text-text-muted"
          aria-live="polite"
        >
          {lines.length} msg{lines.length === 1 ? "" : "s"}
        </span>
      </header>
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        role="log"
        aria-label={`${label} messages`}
        aria-live="polite"
        className="flex-1 overflow-y-auto px-3 py-2 space-y-2 text-sm"
      >
        {empty ? (
          <p className="text-text-muted italic">
            ({label} is listening.)
          </p>
        ) : (
          lines.map((m) => (
            <div key={`${m.ts}-${m.turn_id ?? ""}`} className="font-mono">
              <span className="text-text-dim mr-2">
                {formatTs(m.ts)}
              </span>
              <span>{m.text}</span>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function formatTs(ts: number): string {
  try {
    const d = new Date(ts * 1000);
    const h = d.getHours().toString().padStart(2, "0");
    const m = d.getMinutes().toString().padStart(2, "0");
    const s = d.getSeconds().toString().padStart(2, "0");
    return `${h}:${m}:${s}`;
  } catch {
    return "--:--:--";
  }
}
