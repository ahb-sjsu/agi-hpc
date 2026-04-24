"use client";

/**
 * React hook: subscribe to halyard-state's per-session WS stream
 * and maintain a local map of ``{pc_id → CharacterSheet}``.
 *
 * The WS is cheap — one connection per session per browser. Every
 * sheet update arrives as a single event and we merge it in.
 *
 * Also exposes an HTTP helper to bootstrap sheets on first load
 * so the drawer has content before the first patch arrives.
 */

import { useEffect, useMemo, useState } from "react";
import useWebSocket, { ReadyState } from "react-use-websocket";

import { STATE_HTTP, STATE_WS } from "./env";
import type {
  CharacterSheet,
  PatchEnvelope,
  StateEvent,
} from "./types";

export type ReadyLabel = "connecting" | "open" | "closing" | "closed" | "gone";

interface UseSessionSheetsResult {
  sheets: Record<string, CharacterSheet>;
  pcIds: string[];
  readyState: ReadyLabel;
  refetch: () => Promise<void>;
}

/**
 * Subscribe to halyard-state for one session.
 *
 * ``sessionId`` is the LiveKit room name and the state-service
 * session key — same string end-to-end per HALYARD_TABLE.md §2.6.
 */
export function useSessionSheets(sessionId: string): UseSessionSheetsResult {
  const [sheets, setSheets] = useState<Record<string, CharacterSheet>>({});
  const [pcIds, setPcIds] = useState<string[]>([]);

  const wsUrl = useMemo(
    () => `${STATE_WS}/ws/sheets/${encodeURIComponent(sessionId)}`,
    [sessionId],
  );

  const { lastMessage, readyState } = useWebSocket(wsUrl, {
    shouldReconnect: () => true,
    reconnectAttempts: Infinity,
    reconnectInterval: (attempt) => Math.min(30_000, 500 * 2 ** attempt),
  });

  const refetch = async () => {
    const resp = await fetch(
      `${STATE_HTTP}/api/sheets/${encodeURIComponent(sessionId)}`,
    );
    if (!resp.ok) return;
    const body = (await resp.json()) as { pc_ids: string[] };
    setPcIds(body.pc_ids);
    // Fetch each sheet individually — cheap and keeps the hook
    // independent of a bulk endpoint that may never materialize.
    const pairs = await Promise.all(
      body.pc_ids.map(async (pc): Promise<[string, CharacterSheet] | null> => {
        const r = await fetch(
          `${STATE_HTTP}/api/sheets/${encodeURIComponent(sessionId)}/${encodeURIComponent(pc)}`,
        );
        if (!r.ok) return null;
        return [pc, (await r.json()) as CharacterSheet];
      }),
    );
    setSheets((prev) => {
      const next = { ...prev };
      for (const pair of pairs) {
        if (pair) next[pair[0]] = pair[1];
      }
      return next;
    });
  };

  // Initial bootstrap.
  useEffect(() => {
    void refetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // WS message handling.
  useEffect(() => {
    if (!lastMessage?.data) return;
    try {
      const parsed = JSON.parse(lastMessage.data as string) as StateEvent;
      if (parsed.kind === "session.hello") {
        setPcIds(parsed.pc_ids);
        return;
      }
      if (parsed.kind === "sheet.update") {
        const update = parsed;
        setSheets((prev) => ({ ...prev, [update.pc_id]: update.sheet }));
        setPcIds((prev) =>
          prev.includes(update.pc_id) ? prev : [...prev, update.pc_id],
        );
      }
    } catch {
      // Drop malformed frames silently — halyard-state is the source of truth.
    }
  }, [lastMessage]);

  return {
    sheets,
    pcIds,
    readyState: readyLabel(readyState),
    refetch,
  };
}

function readyLabel(rs: ReadyState): ReadyLabel {
  switch (rs) {
    case ReadyState.CONNECTING:
      return "connecting";
    case ReadyState.OPEN:
      return "open";
    case ReadyState.CLOSING:
      return "closing";
    case ReadyState.CLOSED:
      return "closed";
    default:
      return "gone";
  }
}

/** Apply a patch to one sheet via the state REST API. */
export async function patchSheet(
  sessionId: string,
  pcId: string,
  envelope: PatchEnvelope,
): Promise<CharacterSheet> {
  const resp = await fetch(
    `${STATE_HTTP}/api/sheets/${encodeURIComponent(sessionId)}/${encodeURIComponent(pcId)}/patch`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(envelope),
    },
  );
  if (!resp.ok) {
    let body: unknown = await resp.text();
    try {
      body = JSON.parse(body as string);
    } catch {
      /* fallthrough */
    }
    throw new Error(
      `patch failed (${resp.status}): ${JSON.stringify(body)}`,
    );
  }
  return (await resp.json()) as CharacterSheet;
}
