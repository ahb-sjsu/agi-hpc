"use client";

import { useEffect, useMemo, useState } from "react";

/**
 * StatusPanel — top-center cell of the table grid.
 *
 * Three tabs:
 *   - "info"     — in-fiction date/time, current location.
 *   - "files"    — downloadable artifacts (handouts, dossiers).
 *   - "scene"    — Keeper-broadcast announcements + NPC list.
 *
 * Content is intentionally minimal in this iteration. The
 * artifact/announcement feeds will be wired to a backend route in
 * a later sprint; for now they render a small static set so the
 * Keeper can see the panel's shape during play.
 */

type Tab = "info" | "files" | "scene";

interface ArtifactRef {
  title: string;
  url: string;
  kind: "pdf" | "image" | "audio";
  bytes?: number;
}

const ARTIFACTS: ArtifactRef[] = [
  // Wired through atlas-caddy at /artifacts/* → halyard-state's
  // static directory in a later sprint. For now these are
  // placeholders the Keeper can swap by editing this file.
];

const ANNOUNCEMENTS: { ts: string; text: string }[] = [
  // Populated by the Keeper console in a later sprint.
];

const NPCS: { name: string; role: string; status?: string }[] = [
  { name: "Capt. Iona Marsh", role: "Commanding officer" },
  { name: "Dr. Halverson", role: "Senior science officer" },
  { name: "Chief Engineer Desai", role: "Engineering" },
  { name: "Dr. Ekene Wuyt", role: "Xenobiology" },
];

export default function StatusPanel({ sessionId }: { sessionId: string }) {
  const [tab, setTab] = useState<Tab>("info");

  return (
    <section
      aria-labelledby="status-panel-heading"
      className="flex flex-col h-full border border-border rounded-md bg-surface overflow-hidden"
    >
      <header className="px-3 py-2 border-b border-border flex items-center gap-2">
        <h2
          id="status-panel-heading"
          className="text-sm font-mono text-accent tracking-wider"
        >
          STATUS
        </h2>
        <nav className="ml-auto flex items-center gap-1" role="tablist">
          {(["info", "files", "scene"] as const).map((t) => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className={[
                "px-2 py-0.5 rounded font-mono text-[11px] uppercase",
                tab === t
                  ? "bg-accent-dim text-accent border border-accent"
                  : "text-text-dim border border-border hover:border-accent",
              ].join(" ")}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto px-3 py-2 text-sm font-mono">
        {tab === "info" && <InfoTab sessionId={sessionId} />}
        {tab === "files" && <FilesTab />}
        {tab === "scene" && <SceneTab />}
      </div>
    </section>
  );
}

function InfoTab({ sessionId }: { sessionId: string }) {
  const fictionClock = useFictionClock();
  return (
    <dl className="space-y-1.5 text-xs">
      <Row label="session" value={sessionId} />
      <Row label="ship date" value={fictionClock.date} />
      <Row label="ship time" value={fictionClock.time} mono />
      <Row label="location" value="MKS Halyard · Bridge" />
      <Row label="current scene" value="Cruise — outbound burn nominal" />
      <p className="pt-2 text-text-muted">
        The map cell shows the floorplan of this scene&rsquo;s local
        area. The Keeper can swap maps from the GM console.
      </p>
    </dl>
  );
}

function FilesTab() {
  if (ARTIFACTS.length === 0) {
    return (
      <p className="text-text-muted text-xs italic">
        (no artifacts available — the Keeper hasn&rsquo;t shared
        anything in this scene)
      </p>
    );
  }
  return (
    <ul className="space-y-1.5 text-xs">
      {ARTIFACTS.map((a) => (
        <li
          key={a.url}
          className="flex items-center gap-2 border border-border rounded px-2 py-1.5"
        >
          <KindBadge kind={a.kind} />
          <span className="flex-1 truncate text-text">{a.title}</span>
          {a.bytes !== undefined && (
            <span className="text-text-muted text-[10px]">
              {formatBytes(a.bytes)}
            </span>
          )}
          <a
            href={a.url}
            download
            className="text-accent text-[11px] uppercase hover:underline"
            aria-label={`Download ${a.title}`}
          >
            get
          </a>
        </li>
      ))}
    </ul>
  );
}

function SceneTab() {
  return (
    <div className="space-y-3 text-xs">
      <section aria-label="Announcements">
        <h3 className="text-text-dim uppercase text-[10px] tracking-wider mb-1">
          announcements
        </h3>
        {ANNOUNCEMENTS.length === 0 ? (
          <p className="text-text-muted italic">(no announcements)</p>
        ) : (
          <ul className="space-y-1">
            {ANNOUNCEMENTS.map((a, i) => (
              <li key={i}>
                <span className="text-text-muted mr-2">{a.ts}</span>
                <span>{a.text}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
      <section aria-label="NPCs">
        <h3 className="text-text-dim uppercase text-[10px] tracking-wider mb-1">
          npcs in scene
        </h3>
        <ul className="space-y-1">
          {NPCS.map((n) => (
            <li key={n.name} className="flex items-baseline gap-2">
              <span className="text-text">{n.name}</span>
              <span className="text-text-muted text-[10px]">{n.role}</span>
              {n.status && (
                <span className="ml-auto text-warn text-[10px]">
                  {n.status}
                </span>
              )}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function Row({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline gap-2">
      <dt className="text-text-muted uppercase text-[10px] w-20 shrink-0">
        {label}
      </dt>
      <dd className={mono ? "text-text tabular-nums" : "text-text"}>
        {value}
      </dd>
    </div>
  );
}

function KindBadge({ kind }: { kind: ArtifactRef["kind"] }) {
  return (
    <span
      className={[
        "px-1.5 py-0.5 rounded text-[9px] font-mono uppercase border",
        kind === "pdf"
          ? "border-accent text-accent"
          : kind === "image"
            ? "border-warn text-warn"
            : "border-ok text-ok",
      ].join(" ")}
    >
      {kind}
    </span>
  );
}

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(0)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

/**
 * In-fiction clock. The campaign is set in 2348; this maps real
 * UTC to a fictional shipboard chronology. The mapping is a
 * straight delta (real_year + 322) so the day-of-year and
 * time-of-day track real life — useful for the Keeper because
 * "it's morning here" maps to "it's morning on the ship."
 */
function useFictionClock() {
  const [now, setNow] = useState<Date>(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  return useMemo(() => {
    const fictionalYear = now.getUTCFullYear() + 322;
    const months = [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec",
    ];
    const date = `${now.getUTCDate().toString().padStart(2, "0")} ${
      months[now.getUTCMonth()]
    } ${fictionalYear}`;
    const time = `${String(now.getUTCHours()).padStart(2, "0")}:${String(
      now.getUTCMinutes(),
    ).padStart(2, "0")}:${String(now.getUTCSeconds()).padStart(2, "0")} UTC`;
    return { date, time };
  }, [now]);
}
