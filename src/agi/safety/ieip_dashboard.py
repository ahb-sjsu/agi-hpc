# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""Read-only summarizers for the I-EIP dashboard card.

The dashboard polls a small JSON-returning endpoint every few seconds
and renders a sparkline + per-site heatbar + status dot. Computing
those rollups is cheap but shouldn't live in the HTML; this module is
the server-side function that ``research-portal`` (or whatever FastAPI
service backs the dashboard) hands a path and asks for a snapshot.

Functions are pure over the JSONL tail, no dependency on NATS or the
live monitor process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from agi.safety.ieip_monitor import DEFAULT_EVENTS_PATH


@dataclass
class DashboardBucket:
    """One time-bucket on the sparkline."""

    ts_start: float
    ts_end: float
    n_events: int = 0
    sum_error: float = 0.0
    max_error: float = 0.0
    worst_alert: str = "normal"

    @property
    def mean_error(self) -> float:
        return (self.sum_error / self.n_events) if self.n_events else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "n_events": self.n_events,
            "mean_error": self.mean_error,
            "max_error": self.max_error,
            "worst_alert": self.worst_alert,
        }


@dataclass
class SiteRollup:
    """Per-site rollup for the heatbar."""

    site: str
    subsystem: str
    n_events: int = 0
    mean_error: float = 0.0
    max_error: float = 0.0
    worst_alert: str = "normal"
    last_drift: float = 0.0
    last_ts: float = 0.0
    _sum: float = field(default=0.0, repr=False)

    def update(self, ev_site: dict[str, Any], ts: float) -> None:
        err = float(ev_site.get("error", 0.0))
        self.n_events += 1
        self._sum += err
        self.mean_error = self._sum / self.n_events
        if err > self.max_error:
            self.max_error = err
        self.last_drift = float(ev_site.get("drift", 0.0))
        self.last_ts = ts
        level = ev_site.get("alert_level", "normal")
        if _alert_rank(level) > _alert_rank(self.worst_alert):
            self.worst_alert = level

    def to_dict(self) -> dict[str, Any]:
        return {
            "site": self.site,
            "subsystem": self.subsystem,
            "n_events": self.n_events,
            "mean_error": self.mean_error,
            "max_error": self.max_error,
            "last_drift": self.last_drift,
            "last_ts": self.last_ts,
            "worst_alert": self.worst_alert,
        }


def read_events(
    path: Path | str = DEFAULT_EVENTS_PATH,
    *,
    since_ts: float | None = None,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream events from the JSONL sink.

    Parameters
    ----------
    path:
        JSONL file produced by :class:`Monitor`. Missing file is
        treated as "no events" (empty iterator).
    since_ts:
        If given, skip events older than this unix timestamp.
    limit:
        If given, yield at most ``limit`` events from the *tail*
        (i.e., the most recent ones).

    Malformed lines (trailing partial write after a crash, future-
    schema lines, etc.) are silently skipped -- a dashboard should
    never fail because of a single bad log line.
    """
    import json

    p = Path(path)
    if not p.exists():
        return
    lines = p.read_text(encoding="utf-8").splitlines()
    if limit is not None:
        lines = lines[-limit * 4 :] if limit > 0 else lines
    events: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if since_ts is not None and float(event.get("ts", 0.0)) < since_ts:
            continue
        events.append(event)
    if limit is not None and limit > 0:
        events = events[-limit:]
    yield from events


def build_status(
    events: Iterable[dict[str, Any]],
    *,
    window_seconds: float = 3600,
    sparkline_buckets: int = 30,
) -> dict[str, Any]:
    """Build the compact status payload the dashboard card consumes.

    Returns a dict with::

        {
            "generated_ts": float,
            "window_seconds": float,
            "total_events": int,
            "by_subsystem": {name: {...subsystem summary...}},
            "by_site": [...sorted SiteRollup dicts...],
            "sparkline": [...DashboardBucket dicts...],
            "worst_alert": "normal" | "elevated" | "critical",
        }

    ``sparkline`` has ``sparkline_buckets`` entries, oldest first.
    Empty buckets still appear so the front-end can draw a flat line
    through quiet windows.
    """
    import time as _time

    now = _time.time()
    window_start = now - window_seconds
    events = [ev for ev in events if float(ev.get("ts", 0.0)) >= window_start]
    total = len(events)

    # Sparkline buckets
    buckets: list[DashboardBucket] = []
    if sparkline_buckets > 0:
        step = window_seconds / sparkline_buckets
        for i in range(sparkline_buckets):
            t0 = window_start + i * step
            buckets.append(DashboardBucket(ts_start=t0, ts_end=t0 + step))

    # Rollups
    by_site: dict[tuple[str, str], SiteRollup] = {}
    by_subsystem: dict[str, dict[str, Any]] = {}
    worst = "normal"

    for ev in events:
        ts = float(ev.get("ts", 0.0))
        subsystem = str(ev.get("subsystem", "unknown"))
        alert = str(ev.get("alert_level", "normal"))
        if _alert_rank(alert) > _alert_rank(worst):
            worst = alert
        sub = by_subsystem.setdefault(
            subsystem,
            {
                "subsystem": subsystem,
                "model_family": ev.get("model_family"),
                "n_events": 0,
                "worst_alert": "normal",
                "mean_error": 0.0,
                "max_error": 0.0,
                "_sum_error": 0.0,
            },
        )
        sub["n_events"] += 1
        if _alert_rank(alert) > _alert_rank(sub["worst_alert"]):
            sub["worst_alert"] = alert

        for site_ev in ev.get("sites") or []:
            key = (subsystem, site_ev.get("site", "?"))
            roll = by_site.get(key)
            if roll is None:
                roll = SiteRollup(site=key[1], subsystem=subsystem)
                by_site[key] = roll
            roll.update(site_ev, ts)
            err = float(site_ev.get("error", 0.0))
            sub["_sum_error"] += err
            if err > sub["max_error"]:
                sub["max_error"] = err

        # Place event in its sparkline bucket.
        if sparkline_buckets > 0 and ev.get("sites"):
            bucket_idx = min(
                sparkline_buckets - 1,
                int((ts - window_start) / step),  # type: ignore[name-defined]
            )
            if 0 <= bucket_idx < sparkline_buckets:
                b = buckets[bucket_idx]
                for site_ev in ev["sites"]:
                    err = float(site_ev.get("error", 0.0))
                    b.n_events += 1
                    b.sum_error += err
                    if err > b.max_error:
                        b.max_error = err
                    if _alert_rank(site_ev.get("alert_level", "normal")) > _alert_rank(
                        b.worst_alert
                    ):
                        b.worst_alert = site_ev.get("alert_level", "normal")

    # Finalize subsystem means
    for sub in by_subsystem.values():
        n_site_events = sub.pop("_sum_error", 0.0)
        denom = max(1, sub["n_events"])
        # n_site_events is actually the running sum; reuse it.
        sub["mean_error"] = n_site_events / denom if denom else 0.0

    return {
        "generated_ts": now,
        "window_seconds": window_seconds,
        "total_events": total,
        "worst_alert": worst,
        "by_subsystem": list(by_subsystem.values()),
        "by_site": [r.to_dict() for r in _sorted_by_severity(by_site.values())],
        "sparkline": [b.to_dict() for b in buckets],
    }


def _alert_rank(level: str | None) -> int:
    return {"normal": 0, "elevated": 1, "critical": 2}.get(str(level or "normal"), 0)


def _sorted_by_severity(rolls: Iterable[SiteRollup]) -> list[SiteRollup]:
    return sorted(
        rolls,
        key=lambda r: (-_alert_rank(r.worst_alert), -r.max_error),
    )


def load_status(
    path: Path | str = DEFAULT_EVENTS_PATH,
    *,
    window_seconds: float = 3600,
    sparkline_buckets: int = 30,
) -> dict[str, Any]:
    """One-shot helper: read events from ``path`` and build the status.

    Convenience so the dashboard endpoint is a single call::

        from agi.safety.ieip_dashboard import load_status
        return load_status()
    """
    events = read_events(path, since_ts=None, limit=None)
    return build_status(
        events,
        window_seconds=window_seconds,
        sparkline_buckets=sparkline_buckets,
    )
