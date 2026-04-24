/**
 * Client-side environment — where to find the backend services.
 *
 * Because the web client ships standalone and the backends live
 * behind atlas-caddy, the browser needs to know which origin to
 * hit. In dev we point everything at localhost; in production
 * the Caddy site is the single public origin and the state
 * service is reached at ``/state/*`` (reverse-proxied).
 *
 * **NEXT_PUBLIC_* access must be literal** — Next.js substitutes
 * the string ``process.env.NEXT_PUBLIC_FOO`` at build time, but
 * does NOT substitute dynamic forms like ``process.env[name]``
 * or ``process.env["NEXT_PUBLIC_FOO"]``. A previous revision of
 * this file used the dynamic form and silently fell through to
 * localhost defaults in the browser. Keep the literal accesses.
 */

function trimOrigin(raw: string, fallback: string): string {
  return (raw || fallback).replace(/\/+$/, "");
}

/** LiveKit WebSocket URL. Browser connects here with a JWT. */
export const LIVEKIT_URL = trimOrigin(
  process.env.NEXT_PUBLIC_LIVEKIT_URL ?? "",
  "ws://127.0.0.1:7880",
);

/** Base URL of halyard-state (REST). */
export const STATE_HTTP = trimOrigin(
  process.env.NEXT_PUBLIC_STATE_HTTP ?? "",
  "http://127.0.0.1:8090",
);

/**
 * WebSocket URL of halyard-state. Derived from the HTTP origin
 * rather than a separate env var so the two can't drift.
 */
export const STATE_WS = STATE_HTTP.replace(/^http/, "ws");

/**
 * Token-minting endpoint. Owned by halyard-keeper-backend in the
 * final design.
 */
export const TOKEN_MINT_URL = trimOrigin(
  process.env.NEXT_PUBLIC_TOKEN_MINT_URL ?? "",
  "http://127.0.0.1:8091/api/livekit/token",
);
