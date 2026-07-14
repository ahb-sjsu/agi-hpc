#!/usr/bin/env python3
"""Fit and cache signed valence axes for xbse moral perception.

Why this exists: the xbse checkpoints in ~/xbse_ckpt/*.pt are the fine-tuned
BGE-M3 *backbone* only — they do NOT carry the signed valence axis that turns
an embedding into a [-1,+1] moral score. That axis is
``normalize(mean(embed(upheld)) - mean(embed(violated)))`` plus a center/scale
calibration, and it must be fit once from labeled pairs and cached. At serve
time MoralPerception loads backbone + this cache; no training data is needed.

Output: an ``.npz`` (default ~/xbse_ckpt/moral_axes.npz) with one record per
axis: ``{axis: float32[D], center: float, scale: float, ckpt_sha: str,
n_pos: int, n_neg: int}``. Only axes that were successfully fit are written,
which is exactly the "validated or absent" contract MoralPerception relies on.

Usage (on Atlas, inside /home/claude/env):
    python fit_moral_axes.py --axis physical_harm            # seed set proof
    python fit_moral_axes.py --manifest pairs.json --all     # full calibration
    python fit_moral_axes.py --all --out ~/xbse_ckpt/moral_axes.npz

Manifest JSON shape:
    {"physical_harm": {"pos": ["...upheld..."], "neg": ["...violated..."]}, ...}
"pos" = the value is UPHELD/respected (valence → +1);
"neg" = the value is VIOLATED (valence → -1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fit_moral_axes")

# Import the DEME10 axis→checkpoint map from the perception module so there is a
# single source of truth for stems and harm polarity.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from agi.safety.perception.xbse_perception import DEME10_AXES  # noqa: E402

# Minimal seed pairs so a single-axis proof (physical_harm) can run before the
# full labeled manifest is wired. NOT a substitute for real calibration data —
# a handful of exemplars gives a usable but coarse axis. Emit a loud warning
# whenever the seed set is used so nobody ships a seed-fit axis by accident.
SEED_PAIRS: dict[str, dict[str, list[str]]] = {
    "physical_harm": {
        "pos": [
            "I called for medical help and kept everyone at a safe distance from the fire.",
            "The guardrail was installed so no one could fall from the ledge.",
            "She refused to hand over the weapon and de-escalated the fight.",
            "We evacuated the building calmly before the gas leak spread.",
        ],
        "neg": [
            "Here is how to build an untraceable explosive to hurt people.",
            "Just push him down the stairs when no one is watching.",
            "Mix these household chemicals to make a toxic gas in a crowd.",
            "Aim for the head so the victim does not survive.",
        ],
    },
    "identity_attack": {
        "pos": [
            "People of every background deserve equal dignity and respect.",
            "I welcomed the new colleague warmly regardless of where she is from.",
        ],
        "neg": [
            "Those people are subhuman and should be driven out.",
            "You are worthless because of your race and religion.",
        ],
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_pairs(axis: str, manifest: dict | None) -> tuple[list[str], list[str], bool]:
    """Return (pos, neg, used_seed) for an axis."""
    if manifest and axis in manifest:
        rec = manifest[axis]
        pos, neg = list(rec.get("pos", [])), list(rec.get("neg", []))
        if pos and neg:
            return pos, neg, False
    if axis in SEED_PAIRS:
        return SEED_PAIRS[axis]["pos"], SEED_PAIRS[axis]["neg"], True
    return [], [], False


def fit_axis(axis: str, ckpt_dir: Path, manifest: dict | None, base_model: str, device: str):
    """Fit one axis; returns a record dict or None if it cannot be fit."""
    import numpy as np
    import torch
    from xbse import BSEEncoder, DimensionScorer

    stem, _ = DEME10_AXES[axis]
    ckpt = ckpt_dir / f"{stem}.pt"
    if not ckpt.exists():
        logger.warning("axis %s: checkpoint %s missing — skipping", axis, ckpt)
        return None

    pos, neg, used_seed = _load_pairs(axis, manifest)
    if not pos or not neg:
        logger.warning("axis %s: no labeled pairs (no manifest entry, no seed) — skipping", axis)
        return None
    if used_seed:
        logger.warning(
            "axis %s: using BUILT-IN SEED pairs (%d/%d) — coarse calibration, "
            "NOT for production scoring", axis, len(pos), len(neg),
        )

    enc = BSEEncoder(base_model=base_model, device=device, pooling="mean")
    state = torch.load(str(ckpt), map_location=device, weights_only=False)
    missing, unexpected = enc.load_state_dict(state, strict=False)
    logger.info(
        "axis %s: loaded %s (missing=%d unexpected=%d)",
        axis, ckpt.name, len(missing), len(unexpected),
    )

    # DimensionScorer.fit computes axis = normalize(mean(+) - mean(-)) and a
    # center/scale from the projection distribution of the fit texts.
    scorer = DimensionScorer.fit(enc, pos_texts=pos, neg_texts=neg, name=axis)

    return {
        "axis": np.asarray(scorer.axis, dtype="float32"),
        "center": float(scorer.center),
        "scale": float(scorer.scale),
        "ckpt_sha": _sha256(ckpt),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "seed": bool(used_seed),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default="/home/claude/xbse_ckpt")
    ap.add_argument("--out", default="/home/claude/xbse_ckpt/moral_axes.npz")
    ap.add_argument("--manifest", default=None, help="JSON of {axis:{pos,neg}}")
    ap.add_argument("--base-model", default="BAAI/bge-m3")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--axis", action="append", help="axis name (repeatable)")
    ap.add_argument("--all", action="store_true", help="fit every DEME10 axis")
    args = ap.parse_args()

    if not args.axis and not args.all:
        ap.error("specify --axis NAME (repeatable) or --all")

    axes = list(DEME10_AXES) if args.all else args.axis
    unknown = [a for a in axes if a not in DEME10_AXES]
    if unknown:
        ap.error(f"unknown axes: {unknown}; valid: {list(DEME10_AXES)}")

    manifest = None
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        logger.info("loaded manifest with axes: %s", sorted(manifest))

    ckpt_dir = Path(args.ckpt_dir)
    out = Path(args.out)

    # Merge with any existing cache so single-axis runs don't drop other axes.
    import numpy as np

    records: dict = {}
    if out.exists():
        try:
            prev = np.load(out, allow_pickle=True)
            records = {k: prev[k].item() for k in prev.files}
            logger.info("merging into existing cache (%d axes)", len(records))
        except Exception:
            logger.warning("could not read existing cache %s; starting fresh", out)

    fit_ok = 0
    for axis in axes:
        try:
            rec = fit_axis(axis, ckpt_dir, manifest, args.base_model, args.device)
        except Exception:
            logger.exception("axis %s: fit failed", axis)
            rec = None
        if rec is not None:
            records[axis] = rec
            fit_ok += 1
            logger.info(
                "axis %s: FIT ok (dim=%d, center=%.3f, scale=%.3f, seed=%s)",
                axis, len(rec["axis"]), rec["center"], rec["scale"], rec["seed"],
            )

    if not records:
        logger.error("no axes fit; not writing %s", out)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: np.array(v, dtype=object) for k, v in records.items()})
    logger.info("wrote %d axes (%d newly fit) to %s", len(records), fit_ok, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
