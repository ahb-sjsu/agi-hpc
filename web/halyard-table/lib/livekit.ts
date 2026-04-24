/**
 * LiveKit connection helpers.
 *
 * Responsible for: minting-request hand-off to the Keeper backend,
 * DataChannel envelope parsing, and the one room-level hook the
 * session view uses. The actual React tiles come from
 * ``@livekit/components-react`` — we don't reinvent that layer.
 */

import type { Room, RoomConnectOptions } from "livekit-client";

import { LIVEKIT_URL, TOKEN_MINT_URL } from "./env";
import type { Envelope } from "./types";

// ─────────────────────────────────────────────────────────────────
// Token fetching
// ─────────────────────────────────────────────────────────────────

export interface MintTokenArgs {
  sessionId: string;
  identity: string;
  displayName?: string;
}

export interface MintedToken {
  token: string;
  /** LiveKit WS URL the client should connect to. */
  url: string;
}

/**
 * Fetch a LiveKit participant JWT from the Keeper backend.
 *
 * Errors are thrown rather than swallowed — the caller typically
 * surfaces these on the landing page ("session not found",
 * "unauthorized").
 */
export async function mintToken(
  args: MintTokenArgs,
): Promise<MintedToken> {
  const resp = await fetch(TOKEN_MINT_URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      session_id: args.sessionId,
      identity: args.identity,
      name: args.displayName ?? args.identity,
    }),
  });
  if (!resp.ok) {
    const txt = await resp.text().catch(() => resp.statusText);
    throw new Error(`token mint failed (${resp.status}): ${txt}`);
  }
  const body = (await resp.json()) as { token: string; url?: string };
  return { token: body.token, url: body.url ?? LIVEKIT_URL };
}

// ─────────────────────────────────────────────────────────────────
// DataChannel envelopes
// ─────────────────────────────────────────────────────────────────

/**
 * Parse one DataChannel payload as an Envelope. Returns ``null``
 * on malformed JSON or missing ``kind``.
 *
 * Kept separate from the React layer so session components can
 * unit-test routing without mounting real LiveKit hooks.
 */
export function parseEnvelope(data: Uint8Array | ArrayBuffer): Envelope | null {
  try {
    const text = new TextDecoder("utf-8").decode(data);
    const parsed = JSON.parse(text) as unknown;
    if (
      parsed &&
      typeof parsed === "object" &&
      "kind" in parsed &&
      typeof (parsed as { kind: unknown }).kind === "string"
    ) {
      return parsed as Envelope;
    }
    return null;
  } catch {
    return null;
  }
}

/** Publish a DataChannel envelope to the whole room (RELIABLE). */
export async function publishEnvelope(
  room: Room,
  env: Envelope,
): Promise<void> {
  const payload = new TextEncoder().encode(JSON.stringify(env));
  // LiveKit 2.x publishData: reliable goes in options; the legacy
  // `kind: DataPacket_Kind.RELIABLE` form is removed.
  await room.localParticipant.publishData(payload, { reliable: true });
}

// ─────────────────────────────────────────────────────────────────
// Connect options
// ─────────────────────────────────────────────────────────────────

export const DEFAULT_CONNECT_OPTIONS: RoomConnectOptions = {
  // Retry budget keeps the session alive across brief network hiccups
  // — common over a consumer internet connection + TURN relay.
  autoSubscribe: true,
  // Fast ICE gathering matters more than perfect candidate selection
  // for a small table; once we have more than 5 participants this
  // trade-off is worth revisiting.
  rtcConfig: {
    iceTransportPolicy: "all",
  },
};
