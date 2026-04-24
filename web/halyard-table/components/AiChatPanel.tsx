"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useDataChannel, useRoomContext } from "@livekit/components-react";

import { TOKEN_MINT_URL } from "@/lib/env";
import { parseEnvelope, publishEnvelope } from "@/lib/livekit";
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
 * Local chat-log entry. Unlike ``AiSay`` from the wire format,
 * this includes the locally-typed prompts from the player so
 * the panel reads as a conversation.
 */
interface ChatLine {
  kind: "user.say" | "artemis.say" | "sigma4.say";
  text: string;
  speaker?: string;
  ts: number;
}

/**
 * AiChatPanel — DataChannel-backed chat pane for one AI NPC,
 * plus a dev-affordance text input that POSTs to the stub turn
 * endpoint when the player types.
 *
 * Long-term design: the AIs listen via Whisper on the LiveKit
 * audio tracks and reply via DataChannel envelope. While that
 * ingestion chain is pending, the text box below posts directly
 * to the halyard-keeper stub endpoint so dry-run sessions can
 * still see in-persona responses.
 *
 * Responses arriving via DataChannel (from a real AI agent) are
 * also displayed; the two paths don't conflict.
 */
export default function AiChatPanel({
  ai,
  maxLines = 200,
}: AiChatPanelProps) {
  const [lines, setLines] = useState<ChatLine[]>([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const stickRef = useRef(true);
  const room = useRoomContext();

  const targetKind = ai === "artemis" ? "artemis.say" : "sigma4.say";
  const label = ai === "artemis" ? "ARTEMIS" : "SIGMA-4";

  // Derive the stub-turn endpoint from the token-mint URL so both
  // stay same-origin behind Caddy (swap ``/livekit/token`` →
  // ``/ai/<which>/stub-turn``).
  const stubUrl = useMemo(
    () => TOKEN_MINT_URL.replace(/\/livekit\/token$/, `/ai/${ai}/stub-turn`),
    [ai],
  );

  // Subscribe to the DataChannel — real AI agents publish here.
  useDataChannel((msg) => {
    const env = parseEnvelope(msg.payload);
    if (!env || env.kind !== targetKind) return;
    const say = env as AiSay;
    setLines((prev) =>
      trimLines(
        [...prev, { kind: say.kind, text: say.text, ts: say.ts }],
        maxLines,
      ),
    );
  });

  // Autoscroll unless the user has scrolled up.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (stickRef.current) el.scrollTop = el.scrollHeight;
  }, [lines]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    stickRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 32;
  };

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const text = draft.trim();
    if (!text || sending) return;
    setSending(true);

    const now = Date.now() / 1000;
    // Show the user's message in the panel immediately.
    setLines((prev) =>
      trimLines(
        [
          ...prev,
          { kind: "user.say", text, speaker: "you", ts: now },
        ],
        maxLines,
      ),
    );
    setDraft("");

    // Also broadcast the user's text over DataChannel so other
    // participants see it as an in-room prompt. The AIs (when
    // they come online via voice/whisper) would normally pick up
    // the spoken version — this is the text-path equivalent.
    try {
      await publishEnvelope(room, {
        kind: "scene.trigger",
        scene_id: `user-say:${ai}`,
        note: text,
        ts: now,
      });
    } catch {
      /* non-fatal */
    }

    try {
      const resp = await fetch(stubUrl, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text, speaker: "player" }),
      });
      if (!resp.ok) {
        throw new Error(`${resp.status} ${resp.statusText}`);
      }
      const body = (await resp.json()) as {
        text: string;
        ts: number;
        kind: string;
      };
      setLines((prev) =>
        trimLines(
          [
            ...prev,
            {
              kind: body.kind as ChatLine["kind"],
              text: body.text,
              ts: body.ts,
            },
          ],
          maxLines,
        ),
      );
    } catch (err) {
      setLines((prev) =>
        trimLines(
          [
            ...prev,
            {
              kind: targetKind,
              text: `(no response — ${String((err as Error).message ?? err)})`,
              ts: Date.now() / 1000,
            },
          ],
          maxLines,
        ),
      );
    } finally {
      setSending(false);
    }
  }

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
        className="flex-1 min-h-0 overflow-y-auto px-3 py-2 space-y-2 text-sm"
      >
        {empty ? (
          <p className="text-text-muted italic">
            ({label} is listening. Type below to address them.)
          </p>
        ) : (
          lines.map((m, i) => (
            <ChatLineRow key={`${m.ts}-${i}`} line={m} aiLabel={label} />
          ))
        )}
      </div>

      <form
        onSubmit={send}
        className="flex items-center gap-2 px-2 py-2 border-t border-border bg-surface-2"
      >
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={`Say something to ${label}…`}
          aria-label={`Message to ${label}`}
          disabled={sending}
          maxLength={1024}
          className="flex-1 bg-bg border border-border rounded px-2 py-1 text-sm font-mono focus:outline-none focus:border-accent disabled:opacity-50"
          autoComplete="off"
        />
        <button
          type="submit"
          disabled={sending || !draft.trim()}
          className="px-3 py-1 text-xs font-mono border border-border rounded bg-accent-dim hover:bg-accent-dim hover:border-accent disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {sending ? "…" : "send"}
        </button>
      </form>
    </section>
  );
}

function trimLines(lines: ChatLine[], max: number): ChatLine[] {
  if (lines.length <= max) return lines;
  return lines.slice(lines.length - max);
}

function ChatLineRow({
  line,
  aiLabel,
}: {
  line: ChatLine;
  aiLabel: string;
}) {
  const isUser = line.kind === "user.say";
  return (
    <div className="font-mono leading-snug">
      <span className="text-text-dim mr-2 text-[10px]">{formatTs(line.ts)}</span>
      <span
        className={
          isUser
            ? "text-orange mr-1 text-xs"
            : "text-accent mr-1 text-xs"
        }
      >
        {isUser ? "you:" : `${aiLabel}:`}
      </span>
      <span className={isUser ? "text-text-dim" : ""}>{line.text}</span>
    </div>
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
