/**
 * Client-side environment — where to find the backend services.
 *
 * Because the web client ships standalone and the backends live
 * behind atlas-caddy, the browser needs to know which origin to
 * hit. In dev we point everything at localhost; in production
 * the Caddy site is the single public origin and the state
 * service is reached at ``/state/*`` (reverse-proxied).
 */

function readOrigin(
  name: string,
  fallback: string,
): string {
  // Next.js exposes NEXT_PUBLIC_* vars to the browser at build time.
  // Server-side code that needs these should read process.env as
  // usual and not go through this helper.
  const raw =
    (typeof process !== "undefined" &&
      (process.env as Record<string, string | undefined>)[name]) ||
    "";
  return (raw || fallback).replace(/\/+$/, "");
}

/** LiveKit WebSocket URL. Browser connects here with a JWT. */
export const LIVEKIT_URL = readOrigin(
  "NEXT_PUBLIC_LIVEKIT_URL",
  "ws://127.0.0.1:7880",
);

/** Base URL of halyard-state (REST). */
export const STATE_HTTP = readOrigin(
  "NEXT_PUBLIC_STATE_HTTP",
  "http://127.0.0.1:8090",
);

/**
 * WebSocket URL of halyard-state. Derived from the HTTP origin
 * rather than a separate env var so the two can't drift.
 */
export const STATE_WS = STATE_HTTP.replace(/^http/, "ws");

/**
 * Token-minting endpoint. Owned by halyard-keeper-backend in the
 * final design (Sprint 6); defaults here to localhost so dev
 * setups can stub it with a static file or a tiny proxy.
 */
export const TOKEN_MINT_URL = readOrigin(
  "NEXT_PUBLIC_TOKEN_MINT_URL",
  "http://127.0.0.1:8091/api/livekit/token",
);
