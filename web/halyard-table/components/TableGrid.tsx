"use client";

import {
  useLocalParticipant,
  useParticipants,
} from "@livekit/components-react";
import { useMemo } from "react";

import { AI_IDENTITIES, isGmIdentity } from "@/lib/identity";

import AiChatPanel from "./AiChatPanel";
import MapPanel from "./MapPanel";
import PlayerSlot from "./PlayerSlot";
import ScreenShareControl from "./ScreenShareControl";
import StatusPanel from "./StatusPanel";

/**
 * TableGrid — fixed 4×3 layout for the Halyard Table.
 *
 *   ARTEMIS | Status | SIGMA
 *      P1   |  Map   |  P2
 *      P3   |  GM    |  P4
 *      P5   |  P6    |  P7
 *
 * The center column is always Status / Map / GM / one extra player
 * slot. The named NPC chat panels (ARTEMIS, SIGMA) hold the top
 * corners; when the AI voice/avatar pipeline lands those cells will
 * gain a video tile alongside the chat pane.
 *
 * Player-slot policy:
 *   - Identity ``gm`` (or starting with ``gm-``, or ``keeper``)
 *     goes to the GM cell.
 *   - AI identities (``artemis``, ``sigma-4``) are intercepted by
 *     the named cells, never spilled into P-slots.
 *   - Remaining participants are sorted by identity (deterministic)
 *     and slotted into P1..P7 in that order.
 *
 * Screen-share is GM-only. The ``ScreenShareControl`` is mounted
 * inside the GM slot and only visible when the local participant
 * IS the GM — players see the GM's video tile without the toggle.
 */

export default function TableGrid({ sessionId }: { sessionId: string }) {
  const participants = useParticipants();
  const { localParticipant } = useLocalParticipant();
  const localIsGm = isGmIdentity(localParticipant?.identity);

  const { gm, players } = useMemo(() => {
    let gm = null as ReturnType<typeof useParticipants>[number] | null;
    const players: ReturnType<typeof useParticipants> = [];
    const sorted = [...participants].sort((a, b) =>
      a.identity.localeCompare(b.identity),
    );
    for (const p of sorted) {
      const id = p.identity.toLowerCase();
      if (AI_IDENTITIES.has(id)) continue;
      if (!gm && isGmIdentity(id)) {
        gm = p;
        continue;
      }
      players.push(p);
    }
    return { gm, players };
  }, [participants]);

  const playerAt = (i: number) => players[i] ?? null;

  return (
    <div
      className="flex-1 grid grid-cols-3 grid-rows-4 gap-1.5 p-1.5 min-h-0"
      role="region"
      aria-label="Table grid"
    >
      {/* Row 1: ARTEMIS | Status | SIGMA */}
      <AiChatPanel ai="artemis" />
      <StatusPanel sessionId={sessionId} />
      <AiChatPanel ai="sigma4" />

      {/* Row 2: P1 | Map | P2 */}
      <PlayerSlot
        participant={playerAt(0)}
        label="P1"
        fallbackText="(seat open)"
      />
      <MapPanel />
      <PlayerSlot
        participant={playerAt(1)}
        label="P2"
        fallbackText="(seat open)"
      />

      {/* Row 3: P3 | GM | P4 */}
      <PlayerSlot
        participant={playerAt(2)}
        label="P3"
        fallbackText="(seat open)"
      />
      <PlayerSlot
        participant={gm}
        label="GM"
        highlight
        fallbackText="(no Keeper joined)"
        extraControl={localIsGm ? <ScreenShareControl /> : null}
      />
      <PlayerSlot
        participant={playerAt(3)}
        label="P4"
        fallbackText="(seat open)"
      />

      {/* Row 4: P5 | P6 | P7 */}
      <PlayerSlot
        participant={playerAt(4)}
        label="P5"
        fallbackText="(seat open)"
      />
      <PlayerSlot
        participant={playerAt(5)}
        label="P6"
        fallbackText="(seat open)"
      />
      <PlayerSlot
        participant={playerAt(6)}
        label="P7"
        fallbackText="(seat open)"
      />
    </div>
  );
}
