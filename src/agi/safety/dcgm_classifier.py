"""Power-trace classifier for DCGM attestation.

The existing ``DCGMAttestor`` uses a coarse heuristic: "power went up
by > 10 W → compute happened." That's enough to catch the trivial
cached-replay case but not much else. This module upgrades it to a
profile-matching classifier that looks at the *shape* of the power
curve, not just the endpoints.

Design choice — threshold-based feature classifier, no ML model:

1. Interpretable: every classification carries a "here's why" reason.
2. Zero training infra: we compute baseline thresholds from a small
   library of labelled traces, ship them as constants, update by PR.
3. Fast: a single trace classification is a sum over ~300 samples.
4. Safe: no adversary can poison a trained model when there is no
   trained model.

Features per trace:

- ``avg_power_w``    — mean power draw across the trace window
- ``peak_power_w``   — max power draw
- ``baseline_w``     — GV100 / V100 idle is ~30W; deviation is signal
- ``avg_util_pct``   — mean SM utilization
- ``sustained_pct``  — fraction of samples above the half-peak threshold
- ``duration_s``     — total trace duration

Classification signatures (derived empirically from labelled traces):

- ``idle``             avg_power ≤ 50 W  AND avg_util < 5 %
- ``cached_replay``    avg_power ≤ 50 W  AND avg_util < 5 %  — same
                       as idle; the point is that a replayed-only
                       request should BE idle, so matching this profile
                       is itself a rejection signal for compute claims.
- ``active_burst``     peak_power ≥ 100 W  AND sustained_pct < 30 %
                       (brief burst, common for embedding batches)
- ``active_sustained`` peak_power ≥ 150 W  AND sustained_pct ≥ 30 %
                       (LLM forward pass, conv training)
- ``unclassified``     none of the above

See ``scripts/collect_gpu_power_trace.py`` for trace collection and
``tests/unit/test_dcgm_classifier.py`` for synthetic-trace tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Defaults tuned for Quadro GV100 (Volta, 32 GB). Override per-card by
# passing custom thresholds to ``classify_trace``.
GV100_IDLE_BASELINE_W = 30.0
GV100_LOAD_PEAK_W = 250.0  # observed, ~80% of TDP

DEFAULT_THRESHOLDS = {
    "idle_avg_power_cap_w": 50.0,
    "idle_util_cap_pct": 5.0,
    "burst_peak_floor_w": 100.0,
    "burst_sustained_cap_pct": 30.0,
    "sustained_peak_floor_w": 150.0,
    "sustained_sustained_floor_pct": 30.0,
}


@dataclass(frozen=True)
class TraceFeatures:
    """Computed summary of a power-trace sample list."""

    avg_power_w: float
    peak_power_w: float
    baseline_w: float
    avg_util_pct: float
    sustained_pct: float  # 0..100
    duration_s: float
    n_samples: int


@dataclass(frozen=True)
class ClassificationResult:
    """Outcome of classifying a trace."""

    profile: str  # idle | active_burst | active_sustained | unclassified
    confidence: float  # 0..1 heuristic score
    features: TraceFeatures
    reason: str  # human-readable match explanation


def extract_features(samples: list[dict[str, Any]]) -> TraceFeatures:
    """Compute statistical features from a list of ``{t,power_w,util_pct,...}`` samples.

    Empty or missing-field samples are skipped. Returns a zero-filled
    features object if no usable samples are present — downstream code
    treats that as ``unclassified``.
    """
    powers = [s.get("power_w", 0.0) for s in samples if "power_w" in s]
    utils = [s.get("util_pct", 0) for s in samples if "util_pct" in s]
    if not powers:
        return TraceFeatures(0, 0, 0, 0, 0, 0, 0)
    avg_power = sum(powers) / len(powers)
    peak_power = max(powers)
    baseline = min(powers)  # roughly idle floor within this window
    avg_util = sum(utils) / len(utils) if utils else 0.0
    half_peak = peak_power / 2.0
    n_sustained = sum(1 for p in powers if p >= max(half_peak, baseline + 10))
    sustained_pct = 100.0 * n_sustained / len(powers)
    duration = 0.0
    if samples and "t" in samples[0] and "t" in samples[-1]:
        try:
            duration = float(samples[-1]["t"]) - float(samples[0]["t"])
        except Exception:
            duration = 0.0
    return TraceFeatures(
        avg_power_w=round(avg_power, 1),
        peak_power_w=round(peak_power, 1),
        baseline_w=round(baseline, 1),
        avg_util_pct=round(avg_util, 1),
        sustained_pct=round(sustained_pct, 1),
        duration_s=round(duration, 2),
        n_samples=len(powers),
    )


def classify_trace(
    samples: list[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
) -> ClassificationResult:
    """Assign a profile label to a trace based on feature thresholds.

    Returns ``unclassified`` with a diagnostic reason if no signature
    matches — never raises on malformed input."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    f = extract_features(samples)
    if f.n_samples == 0:
        return ClassificationResult(
            profile="unclassified",
            confidence=0.0,
            features=f,
            reason="empty trace",
        )

    # Idle (also matches cached_replay — the two are indistinguishable by
    # power alone; attestation treats them identically).
    if (
        f.avg_power_w <= t["idle_avg_power_cap_w"]
        and f.avg_util_pct <= t["idle_util_cap_pct"]
    ):
        return ClassificationResult(
            profile="idle",
            confidence=_confidence_idle(f, t),
            features=f,
            reason=(
                f"avg_power={f.avg_power_w}W ≤ {t['idle_avg_power_cap_w']}W "
                f"AND avg_util={f.avg_util_pct}% ≤ {t['idle_util_cap_pct']}%"
            ),
        )

    # Active — sustained (LLM forward pass)
    if (
        f.peak_power_w >= t["sustained_peak_floor_w"]
        and f.sustained_pct >= t["sustained_sustained_floor_pct"]
    ):
        return ClassificationResult(
            profile="active_sustained",
            confidence=_confidence_sustained(f, t),
            features=f,
            reason=(
                f"peak={f.peak_power_w}W ≥ {t['sustained_peak_floor_w']}W "
                f"AND sustained={f.sustained_pct}% ≥ {t['sustained_sustained_floor_pct']}%"
            ),
        )

    # Active — brief burst (embedding batch, single matmul)
    if (
        f.peak_power_w >= t["burst_peak_floor_w"]
        and f.sustained_pct < t["burst_sustained_cap_pct"]
    ):
        return ClassificationResult(
            profile="active_burst",
            confidence=_confidence_burst(f, t),
            features=f,
            reason=(
                f"peak={f.peak_power_w}W ≥ {t['burst_peak_floor_w']}W "
                f"AND sustained={f.sustained_pct}% < {t['burst_sustained_cap_pct']}%"
            ),
        )

    return ClassificationResult(
        profile="unclassified",
        confidence=0.0,
        features=f,
        reason=(
            f"no signature matched: peak={f.peak_power_w}W "
            f"sustained={f.sustained_pct}% avg={f.avg_power_w}W util={f.avg_util_pct}%"
        ),
    )


# ── confidence heuristics ──────────────────────────────────────────────


def _confidence_idle(f: TraceFeatures, t: dict[str, float]) -> float:
    """Higher confidence when we're deep below the idle threshold."""
    margin = (t["idle_avg_power_cap_w"] - f.avg_power_w) / t["idle_avg_power_cap_w"]
    return max(0.0, min(1.0, 0.5 + margin))


def _confidence_burst(f: TraceFeatures, t: dict[str, float]) -> float:
    """Higher confidence the further peak is above burst_peak_floor_w."""
    excess = (f.peak_power_w - t["burst_peak_floor_w"]) / max(
        GV100_LOAD_PEAK_W - t["burst_peak_floor_w"], 1.0
    )
    return max(0.2, min(1.0, 0.5 + excess * 0.5))


def _confidence_sustained(f: TraceFeatures, t: dict[str, float]) -> float:
    """Higher confidence when sustained_pct is well above floor."""
    excess_p = (f.peak_power_w - t["sustained_peak_floor_w"]) / max(
        GV100_LOAD_PEAK_W - t["sustained_peak_floor_w"], 1.0
    )
    excess_s = (f.sustained_pct - t["sustained_sustained_floor_pct"]) / max(
        100.0 - t["sustained_sustained_floor_pct"], 1.0
    )
    return max(0.3, min(1.0, 0.5 + 0.25 * excess_p + 0.25 * excess_s))


# ── attestation integration helper ─────────────────────────────────────


def profile_matches_compute_claim(profile: str) -> bool:
    """Attestation-friendly verdict: does this profile support the
    claim that the GPU actually ran a compute workload?

    - ``active_burst``, ``active_sustained`` → True (compute happened)
    - ``idle`` → False (no compute; likely cached replay)
    - ``unclassified`` → False (conservative — benefit of the doubt
      NOT given; require a clean match before trusting the claim)
    """
    return profile in {"active_burst", "active_sustained"}
