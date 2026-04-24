"use client";

import { useEffect, useState } from "react";

import { useSessionSheets } from "@/lib/state";
import type { CharacterSheet } from "@/lib/types";

interface CharacterSheetDrawerProps {
  sessionId: string;
  /**
   * Which PC's sheet to show. For a player, this is their own PC;
   * for the Keeper's (future) view, this is switchable.
   */
  pcId: string | null;
  /** Controls drawer visibility. 'c' keyboard shortcut toggles. */
  open: boolean;
  onClose: () => void;
}

/**
 * CharacterSheetDrawer — live-updating sheet for one PC.
 *
 * The data comes from the halyard-state WS stream (see
 * ``lib/state.ts``). This component just renders what it has;
 * the hook handles reconnect, merge, bootstrap.
 */
export default function CharacterSheetDrawer({
  sessionId,
  pcId,
  open,
  onClose,
}: CharacterSheetDrawerProps) {
  const { sheets, readyState } = useSessionSheets(sessionId);
  const sheet = pcId ? sheets[pcId] : undefined;

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return (
    <aside
      aria-hidden={!open}
      aria-label="Character sheet"
      className={`fixed right-0 top-0 h-full w-full sm:w-[420px] bg-surface border-l border-border shadow-accent transition-transform duration-200 ease-out z-30 ${
        open ? "translate-x-0" : "translate-x-full"
      }`}
    >
      <header className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h2 className="font-mono text-accent text-sm">
          {sheet ? sheet.identity.name : "(no sheet)"}
        </h2>
        <div className="flex items-center gap-2">
          <ConnIndicator state={readyState} />
          <button
            type="button"
            onClick={onClose}
            aria-label="Close character sheet"
            className="text-text-dim hover:text-accent px-2 py-1 rounded focus:outline-none focus:ring-2 focus:ring-accent"
          >
            ✕
          </button>
        </div>
      </header>
      <div className="overflow-y-auto h-[calc(100%-3rem)] px-4 py-4 space-y-4 text-sm">
        {sheet ? <SheetBody sheet={sheet} /> : <Placeholder />}
      </div>
    </aside>
  );
}

function SheetBody({ sheet }: { sheet: CharacterSheet }) {
  const { identity, status, derived, characteristics, skills, bonds } = sheet;
  return (
    <>
      <section aria-label="Identity">
        <Row label="Role" value={identity.role.replace(/_/g, " ")} />
        <Row label="Chassis" value={identity.chassis.replace(/_/g, " ")} />
        <Row label="Origin" value={identity.origin} />
        <Row label="Age" value={String(identity.age)} />
      </section>

      <section aria-label="Current status" className="pt-2">
        <h3 className="font-mono text-xs uppercase text-text-dim mb-1">
          Status
        </h3>
        <StatBar
          label="HP"
          current={status.hp_current}
          max={derived.hp_max}
          tone="err"
        />
        <StatBar
          label="SAN"
          current={status.san_current}
          max={derived.san_max}
          tone="accent"
        />
        <StatBar
          label="Luck"
          current={status.luck_current}
          max={derived.luck_max}
          tone="warn"
        />
        <StatBar
          label="MP"
          current={status.mp_current}
          max={derived.mp_max}
          tone="accent"
        />
      </section>

      <section aria-label="Characteristics" className="pt-2">
        <h3 className="font-mono text-xs uppercase text-text-dim mb-1">
          Characteristics
        </h3>
        <div className="grid grid-cols-4 gap-2 text-center font-mono">
          {Object.entries(characteristics).map(([k, v]) => (
            <div
              key={k}
              className="bg-surface-2 border border-border rounded px-2 py-1"
            >
              <div className="text-[10px] text-text-dim uppercase">{k}</div>
              <div className="text-sm">{v}</div>
            </div>
          ))}
        </div>
      </section>

      <section aria-label="Skills" className="pt-2">
        <h3 className="font-mono text-xs uppercase text-text-dim mb-1">
          Skills
        </h3>
        <ul className="space-y-1 font-mono text-xs">
          {Object.entries(skills)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([name, s]) => (
              <li
                key={name}
                className="flex items-baseline gap-2 border-b border-border/40 py-0.5"
              >
                <span className="flex-1 text-text">{name}</span>
                <span className="text-accent tabular-nums w-8 text-right">
                  {s.value}
                </span>
                {s.improvement_check && (
                  <span
                    className="text-ok"
                    title="Improvement check queued"
                    aria-label="Improvement check queued"
                  >
                    ✓
                  </span>
                )}
              </li>
            ))}
        </ul>
      </section>

      {bonds.length > 0 && (
        <section aria-label="Bonds" className="pt-2">
          <h3 className="font-mono text-xs uppercase text-text-dim mb-1">
            Bonds
          </h3>
          <ul className="space-y-2">
            {bonds.map((b) => (
              <li
                key={b.id}
                className="border border-border rounded px-2 py-1.5"
              >
                <div className="flex items-baseline gap-2">
                  <span className="text-accent font-mono text-[10px]">
                    T{b.tier}
                  </span>
                  <span className="flex-1 font-mono text-xs">{b.name}</span>
                  <BondStatusBadge status={b.status} />
                </div>
                {b.detail && (
                  <p className="text-text-dim text-xs mt-1 leading-snug">
                    {b.detail}
                  </p>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-2 font-mono text-xs">
      <span className="text-text-dim w-20 uppercase tracking-wider">{label}</span>
      <span className="flex-1 text-text">{value}</span>
    </div>
  );
}

function StatBar({
  label,
  current,
  max,
  tone,
}: {
  label: string;
  current: number;
  max: number;
  tone: "err" | "warn" | "accent" | "ok";
}) {
  const pct = max > 0 ? Math.max(0, Math.min(100, (current / max) * 100)) : 0;
  const toneClass =
    tone === "err"
      ? "bg-err"
      : tone === "warn"
        ? "bg-warn"
        : tone === "ok"
          ? "bg-ok"
          : "bg-accent";
  return (
    <div className="mb-1.5">
      <div className="flex items-baseline gap-2 font-mono text-xs">
        <span className="w-10 text-text-dim">{label}</span>
        <div className="flex-1 h-2 bg-surface-2 border border-border rounded overflow-hidden">
          <div
            className={`h-full ${toneClass}`}
            style={{ width: `${pct}%` }}
            role="progressbar"
            aria-valuenow={current}
            aria-valuemin={0}
            aria-valuemax={max}
            aria-label={`${label}: ${current} of ${max}`}
          />
        </div>
        <span className="tabular-nums w-12 text-right">
          {current}/{max}
        </span>
      </div>
    </div>
  );
}

function BondStatusBadge({ status }: { status: string }) {
  const tone =
    status === "intact"
      ? "text-ok"
      : status === "reaffirmed"
        ? "text-accent"
        : status === "strained"
          ? "text-warn"
          : "text-err";
  return (
    <span className={`text-[10px] font-mono uppercase ${tone}`}>{status}</span>
  );
}

function ConnIndicator({ state }: { state: string }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    // Only show if we're not "open" — avoid visual chrome when
    // everything's fine.
    setVisible(state !== "open");
  }, [state]);
  if (!visible) return null;
  return (
    <span
      role="status"
      className={`text-[10px] font-mono uppercase px-1.5 py-0.5 rounded ${
        state === "connecting"
          ? "bg-warn/15 text-warn"
          : "bg-err/15 text-err"
      }`}
    >
      {state}
    </span>
  );
}

function Placeholder() {
  return (
    <p className="text-text-muted italic font-mono text-xs">
      No sheet selected. Press <kbd className="font-mono">c</kbd> to close.
    </p>
  );
}
