"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

/**
 * Landing page.
 *
 * Collects: session id, display name, PC id (for the sheet
 * drawer). The actual token minting happens server-side once
 * halyard-keeper-backend is up; for Sprint 4 we route the user to
 * the session page and mint at join time.
 */
export default function Landing() {
  const router = useRouter();
  const [sessionId, setSessionId] = useState("halyard-s01");
  const [displayName, setDisplayName] = useState("");
  const [pcId, setPcId] = useState("");
  const [err, setErr] = useState<string | null>(null);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!sessionId.match(/^[a-z0-9-]{3,64}$/)) {
      setErr("session id: 3–64 chars of a–z 0–9 -");
      return;
    }
    if (!displayName.trim()) {
      setErr("display name required");
      return;
    }
    const q = new URLSearchParams({
      name: displayName.trim(),
      ...(pcId ? { pc: pcId.trim() } : {}),
    });
    router.push(`/session/${encodeURIComponent(sessionId)}?${q.toString()}`);
  }

  return (
    <main className="min-h-screen flex items-center justify-center px-6">
      <form
        onSubmit={submit}
        className="w-full max-w-md bg-surface border border-border rounded-lg p-6 shadow-accent"
      >
        <h1 className="text-xl font-mono text-accent mb-1">
          Halyard Table
        </h1>
        <p className="text-text-dim text-xs mb-6 font-mono">
          Beyond the Heliopause · live-play runtime
        </p>

        <Field
          label="Session"
          name="session"
          value={sessionId}
          onChange={setSessionId}
          help="Keeper-provided session id (e.g. halyard-s01)."
          required
        />
        <Field
          label="Display name"
          name="name"
          value={displayName}
          onChange={setDisplayName}
          help="How other players see you in the call."
          required
        />
        <Field
          label="PC id (optional)"
          name="pc"
          value={pcId}
          onChange={setPcId}
          help="Your character's id. If set, your sheet drawer auto-opens to it."
        />

        {err && (
          <p
            role="alert"
            className="text-err font-mono text-xs mb-3"
          >
            {err}
          </p>
        )}

        <button
          type="submit"
          className="w-full bg-accent text-bg font-mono py-2 rounded hover:bg-accent/80 focus:outline-none focus:ring-2 focus:ring-accent"
        >
          Join session
        </button>

        <p className="text-text-muted text-[10px] mt-4 font-mono leading-relaxed">
          By joining you consent to the table&apos;s safety charter.
          Any of the voices you hear at this table may be AI.
        </p>
      </form>
    </main>
  );
}

interface FieldProps {
  label: string;
  name: string;
  value: string;
  onChange: (v: string) => void;
  help?: string;
  required?: boolean;
}
function Field({ label, name, value, onChange, help, required }: FieldProps) {
  return (
    <label htmlFor={name} className="block mb-4">
      <span className="font-mono text-xs text-text-dim uppercase tracking-wider">
        {label}
      </span>
      <input
        id={name}
        name={name}
        value={value}
        required={required}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full bg-surface-2 border border-border rounded px-3 py-1.5 font-mono text-sm focus:outline-none focus:border-accent"
        autoComplete="off"
      />
      {help && (
        <span className="block mt-1 text-[10px] text-text-muted font-mono">
          {help}
        </span>
      )}
    </label>
  );
}
