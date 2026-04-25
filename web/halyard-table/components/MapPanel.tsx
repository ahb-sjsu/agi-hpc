"use client";

import { useState } from "react";

/**
 * MapPanel — middle-center cell. Shows the floorplan or area map
 * for the current scene.
 *
 * The Keeper picks which map is active from the GM console; the
 * client subscribes to the map id over DataChannel and swaps
 * source. For now this is a static placeholder — the
 * scene-broadcast wiring lands in a later sprint.
 *
 * Map images live under ``/public/maps/`` in the web bundle; the
 * Keeper can drop new floorplans there and reference them by
 * filename. For now we list a small set inline so the panel has
 * something to render even before the GM console is online.
 */

interface MapEntry {
  id: string;
  title: string;
  src: string | null;
  scale?: string;
}

const MAPS: MapEntry[] = [
  {
    id: "halyard-bridge",
    title: "MKS Halyard — Bridge",
    src: null,
    scale: "1m grid",
  },
  {
    id: "halyard-habring",
    title: "MKS Halyard — Hab ring (Deck 2)",
    src: null,
    scale: "1m grid",
  },
  {
    id: "halyard-engineering",
    title: "MKS Halyard — Engineering (Deck 4)",
    src: null,
    scale: "1m grid",
  },
  { id: "ceres-port", title: "Ceres — Outer Port concourse", src: null },
  { id: "nithon-surface", title: "Nithon — Anomaly approach", src: null },
];

export default function MapPanel() {
  const [activeId, setActiveId] = useState<string>(MAPS[0].id);
  const active = MAPS.find((m) => m.id === activeId) ?? MAPS[0];

  return (
    <section
      aria-labelledby="map-panel-heading"
      className="flex flex-col h-full border border-border rounded-md bg-surface overflow-hidden"
    >
      <header className="px-3 py-2 border-b border-border flex items-center gap-2">
        <h2
          id="map-panel-heading"
          className="text-sm font-mono text-accent tracking-wider"
        >
          MAP
        </h2>
        <select
          value={activeId}
          onChange={(e) => setActiveId(e.target.value)}
          aria-label="Select map"
          className="ml-auto bg-bg border border-border rounded px-1.5 py-0.5 text-[11px] font-mono focus:outline-none focus:border-accent max-w-[60%]"
        >
          {MAPS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.title}
            </option>
          ))}
        </select>
      </header>

      <div className="flex-1 min-h-0 relative bg-bg">
        {active.src ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={active.src}
            alt={active.title}
            className="absolute inset-0 w-full h-full object-contain"
          />
        ) : (
          <Placeholder title={active.title} scale={active.scale} />
        )}
      </div>
    </section>
  );
}

function Placeholder({
  title,
  scale,
}: {
  title: string;
  scale?: string;
}) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-4">
      <svg
        width="48"
        height="48"
        viewBox="0 0 24 24"
        aria-hidden="true"
        className="text-text-muted mb-2"
      >
        <path
          fill="currentColor"
          d="M12 2 4 6v12l8 4 8-4V6l-8-4zm0 2.2L18 7l-6 3-6-3 6-2.8zM6 8.7l5 2.5v8.2l-5-2.5V8.7zm12 8.2-5 2.5v-8.2l5-2.5v8.2z"
        />
      </svg>
      <p className="font-mono text-text-dim text-sm">{title}</p>
      {scale && (
        <p className="font-mono text-text-muted text-[10px] mt-1">
          scale: {scale}
        </p>
      )}
      <p className="font-mono text-text-muted text-[10px] mt-3 italic">
        (floorplan asset not yet uploaded)
      </p>
    </div>
  );
}
