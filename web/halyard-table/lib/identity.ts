/**
 * Identity helpers shared across components.
 *
 * The Halyard Table uses LiveKit ``identity`` strings to route
 * participants into named slots in the 4×3 grid (ARTEMIS, SIGMA,
 * GM) and to gate UI affordances (e.g. only the GM can start a
 * screen share).
 */

export const AI_IDENTITIES = new Set(["artemis", "sigma-4", "sigma4"]);

/** True for the GM/Keeper slot (case-insensitive match). */
export function isGmIdentity(identity: string | null | undefined): boolean {
  if (!identity) return false;
  const id = identity.toLowerCase();
  return (
    id === "gm" ||
    id === "keeper" ||
    id.startsWith("gm-") ||
    id.startsWith("keeper-")
  );
}

/** True for the in-fiction AI NPC slots. */
export function isAiIdentity(identity: string | null | undefined): boolean {
  if (!identity) return false;
  return AI_IDENTITIES.has(identity.toLowerCase());
}
