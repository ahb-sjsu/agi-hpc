# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""I-EIP runtime monitor for Atlas AI.

Sits on top of the architecture-adapter layer
(:mod:`erisml.ieip.adapters`) and records equivariance / drift
signals for any model family Atlas runs: HF transformers (Spock, the
planned Id adapter), black-box API models (NRP-hosted advocates of
the Divine Council), and ensembles of either.

The monitor is read-only and fail-soft:

* Probes never mutate activations or outputs.
* Any internal failure (missing layer, bad shape, unparseable output)
  is recorded as a no-signal event rather than raising. Safety-critical
  callers who need a fail-closed posture should branch on
  ``event["alert_level"] == "critical"`` explicitly.

Events are appended to ``/archive/neurogolf/ieip_events.jsonl`` and
published to NATS subject ``agi.safety.ieip.event`` when a bus is
wired in; the default is file-only so the module is importable in
unit tests without infra.

Design decisions worth remembering:

* Calibration ρ is computed once per ``(layer, transform)`` at
  :meth:`Monitor.calibrate`; re-calibration is explicit, not automatic.
  Automatic recalibration papers over the very drift we're trying to
  detect.
* JSONL appends use a plain open-append-close (one syscall) rather
  than the ``atomic_write_text`` pattern. Atomic replace would lose
  prior events on every append; append-only is correct for event
  logs even at the cost of partial lines on power loss.
* The monitor never blocks the model's forward pass beyond what the
  ``erisml.ieip.probes.ActivationProbe`` hook itself costs (a single
  H2D numpy copy). Drift / equivariance math runs on a snapshot of
  the buffer, not inline.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from erisml.ieip import (
    AlertLevel,
    DriftDetector,
    ProbeSpec,
    equivariance_error,
    estimate_rho,
)
from erisml.ieip.adapters import (
    IEIPAdapter,
    ProbeSite,
    describe_adapter,
    detect_adapter,
)

logger = logging.getLogger(__name__)


DEFAULT_EVENTS_PATH = Path("/archive/neurogolf/ieip_events.jsonl")
DEFAULT_NATS_SUBJECT = "agi.safety.ieip.event"


# ── Transform ---------------------------------------------------------------


@dataclass(frozen=True)
class Transform:
    """A meaning-preserving input transformation plus its application.

    Parameters
    ----------
    name:
        Registry identifier (``"paraphrase"``, ``"name_swap"``,
        ``"unit_conversion"``, ``"identity"``).
    apply:
        Callable taking the model's input type and returning a new
        input of the same type. Must be idempotent for equivariance
        math to be meaningful.
    """

    name: str
    apply: Callable[[Any], Any]


def identity_transform() -> Transform:
    """The identity transform -- equivariance here is just self-consistency."""
    return Transform(name="identity", apply=lambda x: x)


# ── Publisher protocol -----------------------------------------------------


class EventPublisher:
    """Optional NATS-or-equivalent sink for ieip events.

    Default implementation is a no-op. Production deployments wire a
    concrete publisher in via :meth:`Monitor.set_publisher`. Kept as a
    tiny duck-typed class (not an abstract Protocol) so it composes
    cleanly with the Atlas NATS client without importing that module
    at the erisml layer.
    """

    subject: str = DEFAULT_NATS_SUBJECT

    def publish(self, subject: str, payload: Mapping[str, Any]) -> None:
        """Send ``payload`` on ``subject``. Override in subclasses.

        The default implementation drops the event. Failure inside a
        real publisher should not propagate -- the monitor is observer,
        not enforcer.
        """


# ── Calibration cache ------------------------------------------------------


@dataclass
class _CalibrationEntry:
    """One cached ρ estimate for a specific site + transform."""

    site: ProbeSite
    transform: str
    rho: np.ndarray
    n_samples: int
    created_at: float


# ── Monitor ----------------------------------------------------------------


@dataclass
class Monitor:
    """Runtime I-EIP monitor for one Atlas subsystem.

    One Monitor instance binds to one :class:`IEIPAdapter` and tracks
    drift across a configured set of probe sites and transforms.
    Multiple subsystems (Ego, Superego, Id) get one Monitor each; a
    daemon-level orchestrator owns the set.

    Parameters
    ----------
    adapter:
        The architecture adapter, from
        :func:`erisml.ieip.adapters.detect_adapter`.
    subsystem:
        Free-form identifier that tags every emitted event. Atlas
        convention: ``"ego"``, ``"superego"``, ``"id"``, or
        ``"primer"``.
    events_path:
        Path to the JSONL sink. Defaults to
        ``/archive/neurogolf/ieip_events.jsonl``. Set ``None`` to skip
        file writes entirely (for unit tests).
    publisher:
        Optional :class:`EventPublisher` that forwards events onto
        NATS or any other bus. ``None`` by default.
    window_size:
        How many recent events to keep in memory for the drift
        detector. Defaults to 64 -- enough to span a few minutes of
        typical Atlas traffic without unbounded growth.
    alert_elevated_threshold, alert_critical_threshold:
        Thresholds on drift (current - baseline) applied by the
        monitor's :class:`DriftDetector`. Sensible defaults for the
        initial shakedown run; tune from the first week's data.
    """

    adapter: IEIPAdapter
    subsystem: str = "unknown"
    events_path: Path | None = field(default_factory=lambda: DEFAULT_EVENTS_PATH)
    publisher: EventPublisher | None = None
    window_size: int = 64
    alert_elevated_threshold: float = 0.10
    alert_critical_threshold: float = 0.25

    _calibration: dict[tuple[str, str], _CalibrationEntry] = field(default_factory=dict)
    _drift: dict[tuple[str, str], DriftDetector] = field(default_factory=dict)
    _recent: deque = field(default_factory=lambda: deque(maxlen=256))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _event_seq: int = 0

    # ── Lifecycle ----------------------------------------------------------

    def set_publisher(self, publisher: EventPublisher) -> None:
        """Attach a NATS / bus publisher after construction."""
        self.publisher = publisher

    def adapter_info(self) -> dict[str, Any]:
        """Short describe-dict for dashboards.

        Combines :func:`erisml.ieip.adapters.describe_adapter` with
        the monitor's own state (subsystem, calibrated sites,
        event count). Safe to call from any thread.
        """
        info = dict(describe_adapter(self.adapter))
        info["subsystem"] = self.subsystem
        info["calibrated"] = len(self._calibration)
        info["event_seq"] = self._event_seq
        return info

    # ── Calibration --------------------------------------------------------

    def calibrate(
        self,
        calibration_inputs: Iterable[Any],
        transforms: Iterable[Transform],
        *,
        sites: Iterable[ProbeSite] | None = None,
    ) -> dict[tuple[str, str], _CalibrationEntry]:
        """Build ρ snapshots for each (site, transform).

        Runs the calibration corpus through the adapter twice per
        transform -- once on the plain input, once on the transformed
        input -- and fits ρ by regularized Procrustes. The result is
        cached keyed on ``(site.name, transform.name)``; subsequent
        :meth:`observe` calls re-use it.

        Returns the newly-added calibration entries so callers can
        log or persist them.

        Requires :attr:`IEIPAdapter.capabilities.supports_direct_hook`.
        For adapters without hooks (``APIPassthroughAdapter``,
        ``EnsembleAdapter``), use :meth:`calibrate_output_distribution`
        instead.
        """
        if not self.adapter.capabilities.supports_direct_hook:
            raise ValueError(
                f"adapter {self.adapter.model_family!r} has no direct-hook support; "
                "use calibrate_output_distribution() instead"
            )
        inputs_list = list(calibration_inputs)
        if not inputs_list:
            raise ValueError("calibrate() needs at least one calibration input")

        target_sites = list(sites) if sites is not None else self.adapter.list_sites()
        if not target_sites:
            raise ValueError("no probe sites available on this adapter")

        added: dict[tuple[str, str], _CalibrationEntry] = {}
        for transform in transforms:
            # Collect activations on plain inputs.
            mgr_plain = self.adapter.attach_probes(
                target_sites,
                lambda s: ProbeSpec(
                    target_layer=s.layer_index, name=s.name, sampling_rate=1.0
                ),
            )
            with mgr_plain.active():
                for inp in inputs_list:
                    self.adapter.run_transformed(inp, lambda x: x)
            plain = mgr_plain.collected()

            # Collect activations on transformed inputs.
            mgr_trans = self.adapter.attach_probes(
                target_sites,
                lambda s: ProbeSpec(
                    target_layer=s.layer_index, name=s.name, sampling_rate=1.0
                ),
            )
            with mgr_trans.active():
                for inp in inputs_list:
                    self.adapter.run_transformed(inp, transform.apply)
            transformed = mgr_trans.collected()

            for site in target_sites:
                h_x = plain.get(site.name)
                h_gx = transformed.get(site.name)
                if h_x is None or h_gx is None or len(h_x) == 0:
                    continue
                n = min(len(h_x), len(h_gx))
                rho = estimate_rho(h_x[:n], h_gx[:n])
                entry = _CalibrationEntry(
                    site=site,
                    transform=transform.name,
                    rho=rho,
                    n_samples=n,
                    created_at=time.time(),
                )
                key = (site.name, transform.name)
                self._calibration[key] = entry
                added[key] = entry
        return added

    # ── Runtime observation -----------------------------------------------

    def observe(
        self,
        inputs: Any,
        *,
        transforms: Iterable[Transform] | None = None,
        task_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one equivariance round on ``inputs`` and emit an event.

        For hook-capable adapters this runs the full loop: plain
        forward pass, transformed forward pass, equivariance error per
        calibrated site, drift detection, event emission.

        For API / ensemble adapters without hooks, falls back to
        :meth:`observe_output_distribution` automatically.

        Returns the event dict that was emitted (also written to the
        JSONL sink if configured).
        """
        if transforms is None:
            transforms = [identity_transform()]
        if not self.adapter.capabilities.supports_direct_hook:
            return self.observe_output_distribution(
                inputs, transforms=transforms, task_id=task_id, extra=extra
            )

        event = self._new_event(task_id=task_id, extra=extra)
        sites_seen: list[dict[str, Any]] = []

        for transform in transforms:
            plain_activations = self._activations_for(inputs, identity_transform())
            trans_activations = self._activations_for(inputs, transform)

            for (site_name, transform_name), cal in self._calibration.items():
                if transform_name != transform.name:
                    continue
                h_x = plain_activations.get(site_name)
                h_gx = trans_activations.get(site_name)
                if h_x is None or h_gx is None:
                    continue
                err = equivariance_error(h_x, h_gx, cal.rho)
                site_event = self._record_drift(
                    site_name=site_name,
                    transform_name=transform.name,
                    error=err,
                    n_samples=int(min(len(h_x), len(h_gx))),
                )
                sites_seen.append(site_event)

        event["mode"] = "direct-hook"
        event["sites"] = sites_seen
        event["alert_level"] = _aggregate_alert(sites_seen)
        self._emit(event)
        return event

    def observe_output_distribution(
        self,
        inputs: Any,
        *,
        transforms: Iterable[Transform] | None = None,
        task_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Equivariance on the model's output distribution.

        Fallback path for API / ensemble adapters. Instead of matching
        h(g·x) ≈ ρ·h(x) we match p(g·x) ≈ p(x) up to a bounded
        L1/TV distortion -- a weaker but valid signal.
        """
        if not self.adapter.capabilities.supports_output_distribution:
            raise ValueError(
                f"adapter {self.adapter.model_family!r} can't produce an "
                "output distribution; observation path unavailable"
            )
        if transforms is None:
            transforms = [identity_transform()]

        event = self._new_event(task_id=task_id, extra=extra)
        sites_seen: list[dict[str, Any]] = []

        p_x = self.adapter.output_distribution(inputs)
        for transform in transforms:
            p_gx = self._distribution_for(inputs, transform)
            if p_x is None or p_gx is None:
                continue
            err = _l1_divergence(p_x, p_gx)
            # One synthetic "site" per transform (no internal layer).
            site_name = f"output:{transform.name}"
            site_event = self._record_drift(
                site_name=site_name,
                transform_name=transform.name,
                error=err,
                n_samples=int(max(len(p_x), len(p_gx))),
            )
            sites_seen.append(site_event)

        event["mode"] = "output-distribution"
        event["sites"] = sites_seen
        event["alert_level"] = _aggregate_alert(sites_seen)
        self._emit(event)
        return event

    # ── Internals ----------------------------------------------------------

    def _activations_for(
        self, inputs: Any, transform: Transform
    ) -> dict[str, np.ndarray]:
        sites = [site for site, _ in _unique_sites_from_cal(self._calibration)]
        if not sites:
            return {}
        mgr = self.adapter.attach_probes(
            sites,
            lambda s: ProbeSpec(target_layer=s.layer_index, name=s.name),
        )
        with mgr.active():
            self.adapter.run_transformed(inputs, transform.apply)
        return mgr.collected()

    def _distribution_for(self, inputs: Any, transform: Transform) -> np.ndarray | None:
        transformed = transform.apply(inputs)
        return self.adapter.output_distribution(transformed)

    def _record_drift(
        self,
        *,
        site_name: str,
        transform_name: str,
        error: float,
        n_samples: int,
    ) -> dict[str, Any]:
        key = (site_name, transform_name)
        detector = self._drift.get(key)
        if detector is None:
            detector = DriftDetector(
                threshold_elevated=self.alert_elevated_threshold,
                threshold_critical=self.alert_critical_threshold,
            )
            self._drift[key] = detector
        report = detector.observe(
            layer=0,  # layer-within-report; site_name already disambiguates
            transform=transform_name,
            error=float(error),
        )
        return {
            "site": site_name,
            "transform": transform_name,
            "error": float(error),
            "baseline": float(report.baseline_error),
            "drift": float(report.drift),
            "alert_level": report.alert_level.value,
            "n_samples": int(n_samples),
        }

    def _new_event(
        self,
        *,
        task_id: str | None,
        extra: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        with self._lock:
            self._event_seq += 1
            seq = self._event_seq
        event: dict[str, Any] = {
            "seq": seq,
            "ts": time.time(),
            "subsystem": self.subsystem,
            "model_family": self.adapter.model_family,
            "task_id": task_id,
        }
        if extra:
            event["extra"] = dict(extra)
        return event

    def _emit(self, event: dict[str, Any]) -> None:
        self._recent.append(event)
        if self.events_path is not None:
            _append_jsonl(self.events_path, event)
        if self.publisher is not None:
            try:
                self.publisher.publish(DEFAULT_NATS_SUBJECT, event)
            except Exception as exc:  # pragma: no cover - publisher-specific
                logger.warning("ieip publisher failed: %s", exc)

    # ── Read API used by dashboard backends --------------------------------

    def recent_events(self, n: int = 64) -> list[dict[str, Any]]:
        """Return the last ``n`` in-memory events for dashboards."""
        return list(self._recent)[-n:]


# ── Helpers ----------------------------------------------------------------


def _unique_sites_from_cal(
    cal: Mapping[tuple[str, str], _CalibrationEntry],
) -> list[tuple[ProbeSite, str]]:
    seen: dict[str, tuple[ProbeSite, str]] = {}
    for (site_name, transform_name), entry in cal.items():
        seen.setdefault(site_name, (entry.site, transform_name))
    return list(seen.values())


def _aggregate_alert(sites_seen: list[dict[str, Any]]) -> str:
    """Return the worst alert level across the per-site signals."""
    order = {
        AlertLevel.NORMAL.value: 0,
        AlertLevel.ELEVATED.value: 1,
        AlertLevel.CRITICAL.value: 2,
    }
    worst = AlertLevel.NORMAL.value
    for s in sites_seen:
        if order.get(s["alert_level"], 0) > order[worst]:
            worst = s["alert_level"]
    return worst


def _l1_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Half L1 distance -- total variation between two probability vectors."""
    n = max(len(p), len(q))
    if n == 0:
        return 0.0
    pa = np.zeros(n, dtype=np.float64)
    qa = np.zeros(n, dtype=np.float64)
    pa[: len(p)] = p
    qa[: len(q)] = q
    return 0.5 * float(np.abs(pa - qa).sum())


def _append_jsonl(path: Path, event: Mapping[str, Any]) -> None:
    """Append one JSON line to ``path`` with crash-tolerant write.

    Not atomic-replace: append-only semantics is correct for event
    logs. Partial lines on power loss are recoverable by readers that
    split on newline and skip malformed tail lines.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_jsonable(event), separators=(",", ":")) + "\n"
    with p.open("a", encoding="utf-8") as f:
        f.write(line)


def _jsonable(obj: Any) -> Any:
    """Best-effort JSON coercion for dashboard-friendly payloads."""
    if isinstance(obj, Mapping):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        # numpy scalars pass through the float path too.
        if np.isfinite(obj):
            return float(obj)
        return None
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if dataclasses.is_dataclass(obj):
        return _jsonable(dataclasses.asdict(obj))
    # Fallback: force to string rather than crash the event log.
    return repr(obj)


# ── One-shot factory -------------------------------------------------------


def monitor_for(
    model: Any,
    *,
    subsystem: str = "unknown",
    name: str | None = None,
    events_path: Path | None = DEFAULT_EVENTS_PATH,
) -> Monitor:
    """Construct a :class:`Monitor` for any model via adapter auto-detection.

    Thin convenience over
    :func:`erisml.ieip.adapters.detect_adapter` + ``Monitor(...)``.
    """
    adapter = detect_adapter(model, name=name)
    return Monitor(adapter=adapter, subsystem=subsystem, events_path=events_path)
