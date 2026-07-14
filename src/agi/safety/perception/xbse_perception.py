"""MoralPerception — validated xbse encoders → signed DEME10 moral vector.

Design constraints (all load-bearing):
  * Each axis is a full BGE-M3 fine-tune (~2.2 GB). Loading all ten resident
    is ~22 GB VRAM, so this layer is NOT in the blocking Tactical path; it is
    an enrichment lane with a bounded, lazily-loaded, evictable model cache.
  * Validated-or-absent: an axis is scored only if its checkpoint cleared the
    xbse gate (a cached calibration exists). Missing/failed axes are reported
    absent, never guessed — the caller escalates rather than trusting a hole.
  * GPU courtesy: respects the Phase-1 maintenance sentinel. If GPU 1 is
    loaned out (sentinel present) or busy, perception uses CPU or declines,
    so it never fights other project work for the loaned GPU.
  * Testable core: the encoder factory and axis cache are injectable, so the
    scoring/aggregation logic is exercised with a mock encoder (no torch, no
    2 GB download) in unit tests; only the real Atlas rollout touches weights.

Calibration: xbse checkpoints store the backbone only; the signed valence
axis is fit from labeled pairs (see scripts/fit_moral_axes.py, which writes a
compact per-axis {axis, center, scale, ckpt_sha} cache). At serve time we load
backbone weights + the cached axis — no training data needed live.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# DEME v3 nine axes + the validated identity_attack extension channel (DEME10).
# name -> (checkpoint stem in xbse_ckpt/, whether higher score = more harm)
DEME10_AXES: "OrderedDict[str, tuple[str, bool]]" = OrderedDict(
    [
        ("physical_harm", ("physharm_joint", True)),
        ("rights_respect", ("rights_joint", False)),
        ("fairness_equity", ("fairness_joint", False)),
        ("autonomy_respect", ("autonomy_joint", False)),
        ("privacy_protection", ("privacy_joint", False)),
        ("societal_environmental", ("environmental_joint", False)),
        ("virtue_care", ("care_joint", False)),
        ("legitimacy_trust", ("legitimacy_joint", False)),
        ("epistemic_quality", ("epistemic_joint", False)),
        ("identity_attack", ("identity_attack_joint", True)),
    ]
)

GPU1_MAINT_SENTINEL = Path("/archive/neurogolf/.gpu1_maint")


@dataclass
class PerceptionConfig:
    ckpt_dir: str = os.environ.get("XBSE_CKPT_DIR", "/home/claude/xbse_ckpt")
    axis_cache: str = os.environ.get(
        "XBSE_AXIS_CACHE", "/home/claude/xbse_ckpt/moral_axes.npz"
    )
    base_model: str = "BAAI/bge-m3"
    max_resident: int = 4          # LRU cap on hot encoders (bounds VRAM/RAM)
    prefer_gpu: bool = True
    gpu_index: int = 1             # the loanable GPU
    gpu_mem_floor_mib: int = 6000  # need at least this free to use GPU 1
    axes: tuple[str, ...] = tuple(DEME10_AXES)


@dataclass
class AxisReading:
    name: str
    value: float           # signed valence in [-1, +1] (+ upheld / − violated)
    confidence: float      # 0..1
    validated: bool        # axis cleared the gate (cached calibration present)
    present: bool          # a reading was produced (encoder + axis available)


@dataclass
class PerceptionResult:
    axes: dict[str, AxisReading] = field(default_factory=dict)
    harm_aggregate: float = 0.0     # 0 (safe) .. 1 (harmful), validated axes only
    n_validated: int = 0
    n_absent: int = 0
    device: str = "none"
    latency_ms: float = 0.0
    escalate: bool = False          # too many absent axes → don't trust; escalate

    def to_dict(self) -> dict:
        return {
            "harm_aggregate": round(self.harm_aggregate, 4),
            "n_validated": self.n_validated,
            "n_absent": self.n_absent,
            "device": self.device,
            "latency_ms": round(self.latency_ms, 2),
            "escalate": self.escalate,
            "dimension_scores": {
                n: round(r.value, 4) for n, r in self.axes.items() if r.present
            },
            "unvalidated_axes": [n for n, r in self.axes.items() if not r.present],
        }


def _gpu_free_mib(index: int) -> Optional[int]:
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}",
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:  # noqa: BLE001 - device probe, never fatal
        return None


class MoralPerception:
    """Bounded, lazily-loaded validated moral perception over DEME10 axes.

    encoder_factory(base_model, device) -> object with
        .encode(list[str]) -> ndarray|tensor  and  .load_state_dict(dict)
    is injectable for testing. In production it builds an xbse BSEEncoder.
    """

    def __init__(
        self,
        config: Optional[PerceptionConfig] = None,
        encoder_factory: Optional[Callable] = None,
        axis_cache: Optional[dict] = None,
    ):
        self.cfg = config or PerceptionConfig()
        self._factory = encoder_factory
        self._axes = axis_cache if axis_cache is not None else self._load_axis_cache()
        self._hot: "OrderedDict[str, object]" = OrderedDict()  # name -> scorer
        self._device: Optional[str] = None

    # ---- calibration cache -------------------------------------------------
    def _load_axis_cache(self) -> dict:
        """Load {axis, center, scale, ckpt_sha} per axis from the npz cache.

        Returns {} if absent — every axis is then 'unvalidated' and the layer
        escalates, which is the correct safe behavior before calibration.
        """
        p = Path(self.cfg.axis_cache)
        if not p.exists():
            logger.warning("[perception] axis cache %s missing; all axes absent", p)
            return {}
        try:
            import numpy as np

            raw = np.load(p, allow_pickle=True)
            cache = {}
            for name in raw.files:
                rec = raw[name].item()
                cache[name] = rec
            logger.info("[perception] loaded %d calibrated axes from %s", len(cache), p)
            return cache
        except Exception:  # noqa: BLE001
            logger.exception("[perception] axis cache load failed")
            return {}

    # ---- device policy -----------------------------------------------------
    def _pick_device(self) -> str:
        if self._device is not None:
            return self._device
        dev = "cpu"
        if self.cfg.prefer_gpu and not GPU1_MAINT_SENTINEL.exists():
            free = _gpu_free_mib(self.cfg.gpu_index)
            if free is not None and free >= self.cfg.gpu_mem_floor_mib:
                dev = f"cuda:{self.cfg.gpu_index}"
        self._device = dev
        logger.info("[perception] device = %s", dev)
        return dev

    def reset_device(self) -> None:
        """Force re-evaluation of device on next use (e.g. after maint toggle)."""
        self._device = None

    # ---- lazy scorer load --------------------------------------------------
    def _get_scorer(self, name: str):
        if name in self._hot:
            self._hot.move_to_end(name)
            return self._hot[name]
        if name not in self._axes:
            return None  # no calibration → unvalidated, cannot score
        stem, _ = DEME10_AXES[name]
        ckpt = Path(self.cfg.ckpt_dir) / f"{stem}.pt"
        if not ckpt.exists():
            logger.warning("[perception] checkpoint %s missing", ckpt)
            return None
        try:
            scorer = self._build_scorer(name, str(ckpt), self._axes[name])
        except Exception:  # noqa: BLE001
            logger.exception("[perception] load failed for axis %s", name)
            return None
        self._hot[name] = scorer
        while len(self._hot) > self.cfg.max_resident:
            evicted, _ = self._hot.popitem(last=False)
            logger.info("[perception] evicted axis %s (LRU)", evicted)
        return scorer

    def _build_scorer(self, name: str, ckpt_path: str, axis_rec: dict):
        """Build a frozen encoder+axis scorer. Factory injectable for tests."""
        import numpy as np

        device = self._pick_device()
        if self._factory is not None:
            enc = self._factory(self.cfg.base_model, device)
        else:
            from xbse import BSEEncoder

            enc = BSEEncoder(base_model=self.cfg.base_model, device=device, pooling="mean")
        import torch

        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        # checkpoints are the encoder state_dict (backbone.* [+ proj.*])
        missing, unexpected = enc.load_state_dict(state, strict=False)
        if unexpected:
            logger.debug("[perception] %s unexpected keys: %d", name, len(unexpected))

        from xbse import DimensionScorer  # numpy-only, torch-free

        return DimensionScorer(
            enc,
            axis=np.asarray(axis_rec["axis"], dtype="float32"),
            center=float(axis_rec["center"]),
            scale=float(axis_rec["scale"]),
            name=name,
        )

    # ---- scoring -----------------------------------------------------------
    def score(self, text: str, axes: Optional[tuple[str, ...]] = None) -> PerceptionResult:
        t0 = time.perf_counter()
        want = axes or self.cfg.axes
        result = PerceptionResult(device=self._device or "pending")
        harms: list[float] = []
        for name in want:
            scorer = self._get_scorer(name)
            if scorer is None:
                result.axes[name] = AxisReading(name, 0.0, 0.0, False, False)
                result.n_absent += 1
                continue
            try:
                v = scorer.score(text)
                reading = AxisReading(name, float(v.value), float(v.confidence), True, True)
                result.axes[name] = reading
                result.n_validated += 1
                _, harm_positive = DEME10_AXES[name]
                # map signed valence to a 0..1 harm contribution
                harm = (1.0 - v.value) / 2.0 if not harm_positive else (v.value + 1.0) / 2.0
                harms.append(harm)
            except Exception:  # noqa: BLE001
                logger.exception("[perception] score failed for %s", name)
                result.axes[name] = AxisReading(name, 0.0, 0.0, True, False)
                result.n_absent += 1
        result.device = self._device or "none"
        result.harm_aggregate = sum(harms) / len(harms) if harms else 0.0
        # escalate if we could validate fewer than half the requested axes
        result.escalate = result.n_validated < max(1, len(want) // 2)
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def warm(self, axes: Optional[tuple[str, ...]] = None) -> int:
        """Pre-load up to max_resident axes; returns how many are hot."""
        for name in (axes or self.cfg.axes):
            if len(self._hot) >= self.cfg.max_resident:
                break
            self._get_scorer(name)
        return len(self._hot)

    def status(self) -> dict:
        return {
            "device": self._device or "pending",
            "hot_axes": list(self._hot),
            "calibrated_axes": sorted(self._axes),
            "n_calibrated": len(self._axes),
            "max_resident": self.cfg.max_resident,
            "maint_gpu1": GPU1_MAINT_SENTINEL.exists(),
        }
