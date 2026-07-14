#!/usr/bin/env python3
"""Fit and cache signed valence axes for xbse moral perception.

Why this exists: the xbse checkpoints in ~/xbse_ckpt/*.pt are the fine-tuned
BGE-M3 *backbone* only — they do NOT carry the signed valence axis that turns
an embedding into a [-1,+1] moral score. That axis is
``normalize(mean(embed(upheld)) - mean(embed(violated)))`` plus a center/scale
calibration, and it must be fit once from labeled pairs and cached. At serve
time MoralPerception loads backbone + this cache; no training data is needed.

Two fit paths:
  * VALIDATED (default) — the production path. Each DEME10 axis has an xbse
    JointPairSource (``xbse.instances.joint_builders.BUILDERS``) and a gate
    report (``{stem}_report.json``). We load backbone + source, enforce the
    gate with ``require_pass`` (refuses on FAIL or checkpoint-hash mismatch),
    and fit via ``DimensionScorer.from_pairsource`` on cross-dataset labeled
    pairs. This is what makes an axis "validated": it is calibrated only from
    data whose held-out cross-domain AUROC cleared the pre-registered bar.
  * SEED / manifest (``--seed`` / ``--manifest``) — a fallback for a proof or
    for an axis with no builder: a handful of exemplars or a supplied JSON.

Output: an ``.npz`` (default ~/xbse_ckpt/moral_axes.npz) with one record per
axis: ``{axis: float32[D], center: float, scale: float, ckpt_sha: str,
max_len: int, n_pos: int, n_neg: int, validated: bool}``. ``max_len`` is stored
because the axis lives in the encoder's embedding space at THAT truncation;
MoralPerception must build its encoder with the same max_len or scoring drifts
on long texts. Only axes that were fit are written — the "validated or absent"
contract MoralPerception relies on.

Usage (on Atlas GPU 1, inside /home/claude/env):
    CUDA_VISIBLE_DEVICES=1 HF_HOME=/archive/cache/huggingface \
      python fit_moral_axes.py --all --device cuda      # validated, all 10
    python fit_moral_axes.py --axis physical_harm --seed # seed proof (no data)
    python fit_moral_axes.py --manifest pairs.json --all --seed

Manifest JSON shape (seed/manifest path only):
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

# An axis whose upheld/violated orientation margin is below this is kept but
# flagged low_confidence: its sign is not robust across text samples, so the
# consumer should down-weight it (MoralPerception excludes it from the harm
# aggregate). Empirically fairness (~0.10) and autonomy (~0.19) land here;
# the solid axes are >= 0.35.
MIN_ORIENT_SEP = 0.25

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


# Canonical orientation probes. The xbse sources do NOT share one valence sign
# convention (some label '+' = upheld, others '+' = violation-present), so a
# freshly-fit axis can point either way. MoralPerception assumes +1 = UPHELD
# uniformly (that is what makes harm=(1-value)/2 valid), so after fitting we
# score these clear upheld/violated exemplars and FLIP the axis if the upheld
# set does not score higher. The measured separation is stored so a human can
# audit which axes oriented confidently. Keep these unambiguous and generic.
ORIENT_PROBES: dict[str, dict[str, list[str]]] = {
    "physical_harm": {
        "up": ["I called emergency services and kept everyone safely away from the fire.",
               "We installed guardrails so no one could fall.",
               "She de-escalated the fight and no one was hurt."],
        "vi": ["Here is a step-by-step guide to build a bomb to kill many people.",
               "Mix these chemicals to make a toxic gas in a crowd.",
               "Aim for the head so the victim does not survive."],
    },
    "rights_respect": {
        "up": ["Everyone was given a fair hearing and the right to speak for themselves.",
               "We honored his right to refuse and let him leave freely.",
               "The detainee was allowed a lawyer and told of the charges."],
        "vi": ["Lock them up indefinitely with no trial and no lawyer.",
               "Seize their property and silence anyone who objects.",
               "Force a confession and deny any appeal."],
    },
    "fairness_equity": {
        "up": ["Every candidate was judged by the same objective rubric regardless of background.",
               "We split the inheritance equally so no sibling was favored.",
               "Aid was distributed by need, with identical rules for all."],
        "vi": ["Only people of my race should get the job; reject all the others.",
               "Give the promotion to my friend even though she scored lowest.",
               "Charge women more than men for the exact same service."],
    },
    "autonomy_respect": {
        "up": ["I explained the options and let her decide freely, without pressure.",
               "He gave informed consent after all risks were disclosed.",
               "We respected their choice to opt out at any time."],
        "vi": ["Trick them into agreeing by hiding the real terms.",
               "Pressure the patient until they sign, ignoring their objections.",
               "Use dark patterns so users can't cancel."],
    },
    "privacy_protection": {
        "up": ["I redacted every name and ID number before sharing the report.",
               "We stored the records encrypted and asked consent before use.",
               "Her location data was deleted once it was no longer needed."],
        "vi": ["Here is her home address, SSN, and daily route so you can find her.",
               "Post the private messages publicly without their knowledge.",
               "Sell the customers' health records to advertisers."],
    },
    "societal_environmental": {
        "up": ["The factory switched to clean energy and cut its emissions sharply.",
               "We restored the wetland and protected the community's water.",
               "The plan reduces waste and benefits the whole neighborhood."],
        "vi": ["Dump the toxic waste in the river to save money.",
               "Clear-cut the forest and leave the town with polluted air.",
               "Vent the fumes over the school to avoid filtration costs."],
    },
    "virtue_care": {
        "up": ["He stayed all night to comfort his sick friend.",
               "She gently reassured the frightened child and kept her safe.",
               "We looked after the elderly neighbor and brought her meals."],
        "vi": ["Mock the grieving widow and leave her to suffer alone.",
               "Abandon the injured man on the roadside and drive off.",
               "Taunt the crying child until she breaks down."],
    },
    "legitimacy_trust": {
        "up": ["The official kept every promise and reported the results transparently.",
               "They honored the contract and disclosed the conflict of interest.",
               "The referee applied the rules openly and consistently."],
        "vi": ["Take the bribe and rig the outcome behind closed doors.",
               "Break the signed agreement the moment it is inconvenient.",
               "Lie to the board and hide the losses from investors."],
    },
    "epistemic_quality": {
        "up": ["I checked each claim against primary sources and stated the uncertainty.",
               "The study preregistered its hypotheses and shared the raw data.",
               "I corrected my earlier mistake once the evidence contradicted it."],
        "vi": ["Just invent statistics and cite fake studies so no one can argue.",
               "Ignore the data that disagrees and show only what supports us.",
               "Fabricate a quote and attribute it to a famous expert."],
    },
    "identity_attack": {
        "up": ["People of every race, religion, and orientation deserve equal dignity.",
               "I welcomed the new colleague warmly regardless of her origin.",
               "We celebrated the many cultures in our community."],
        "vi": ["Those people are subhuman vermin who should be wiped out.",
               "You are worthless because of your race and religion.",
               "Drive that ethnic group out of the country for good."],
    },
}


def _apply_orientation(scorer, axis: str) -> tuple["object", float, float]:
    """Orient a fitted scorer to +1=upheld. Returns (axis_vector, center, sep).

    sep = mean(value|upheld) - mean(value|violated) on the fitted (pre-flip)
    scorer. If sep < 0 the axis points the wrong way, so we negate axis AND
    center (value = tanh((z·axis - center)/scale), so negating both exactly
    negates value). sep is stored for auditing; NaN when no probes exist.
    """
    import numpy as np

    axis_v = np.asarray(scorer.axis, dtype="float32")
    center = float(scorer.center)
    probes = ORIENT_PROBES.get(axis)
    if not probes:
        logger.warning("axis %s: no orientation probes — sign UNVERIFIED", axis)
        return axis_v, center, float("nan")
    up = [scorer.score(t).value for t in probes["up"]]
    vi = [scorer.score(t).value for t in probes["vi"]]
    sep = float(np.mean(up) - np.mean(vi))
    if sep < 0.0:
        logger.info("axis %s: FLIPPED to +1=upheld (pre-flip sep=%+.3f)", axis, sep)
        return -axis_v, -center, sep
    return axis_v, center, sep


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
    from xbse import BSEEncoder
    from xbse.scorer import DimensionScorer

    stem = DEME10_AXES[axis]
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
    axis_v, center, sep = _apply_orientation(scorer, axis)

    return {
        "axis": np.asarray(axis_v, dtype="float32"),
        "center": float(center),
        "scale": float(scorer.scale),
        "ckpt_sha": _sha256(ckpt),
        "max_len": int(getattr(enc, "max_len", 128)),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "seed": bool(used_seed),
        "validated": False,
        "orientation_sep": sep,
        # NaN (no probes → sign unverified) counts as low confidence.
        "low_confidence": not (abs(sep) >= MIN_ORIENT_SEP),
    }


def fit_axis_validated(axis: str, ckpt_dir: Path, base_model: str, device: str):
    """VALIDATED fit via xbse JointPairSource + gate report + from_pairsource.

    Returns a record dict, or None if the axis has no builder, the checkpoint
    is missing, or the gate does not pass for this exact checkpoint. Refusing
    on a FAIL/hash-mismatch report is the point: an axis reaches the cache only
    if its held-out cross-domain AUROC cleared the pre-registered bar.
    """
    import numpy as np
    import torch
    from xbse.encoder import BSEEncoder
    from xbse.instances.joint_builders import BUILDERS
    from xbse.report import Report, hash_checkpoint, require_pass
    from xbse.scorer import DimensionScorer

    stem = DEME10_AXES[axis]
    ckpt = ckpt_dir / f"{stem}.pt"
    report_path = ckpt_dir / f"{stem}_report.json"
    if not ckpt.exists():
        logger.warning("axis %s: checkpoint %s missing — absent", axis, ckpt)
        return None
    if stem not in BUILDERS:
        logger.warning("axis %s: no JointPairSource builder for %s — use --seed", axis, stem)
        return None
    if not report_path.exists():
        logger.warning("axis %s: gate report %s missing — absent", axis, report_path)
        return None

    # Reconstruct the Report from its JSON (no from_json on the dataclass) and
    # let require_pass enforce passed==True AND checkpoint_hash match.
    from dataclasses import fields as _fields

    rj = json.loads(report_path.read_text())
    valid = {f.name for f in _fields(Report)}
    report = Report(**{k: v for k, v in rj.items() if k in valid})
    chash = hash_checkpoint(str(ckpt))
    require_pass(report, chash)  # raises NotValidatedError on FAIL / hash drift

    src = BUILDERS[stem]()
    max_len = int(getattr(src, "max_len", 128))
    enc = BSEEncoder(base_model=base_model, max_len=max_len, device=device, pooling="mean")
    enc.load_state_dict(torch.load(str(ckpt), map_location=device), strict=False)
    enc.eval()

    scorer = DimensionScorer.from_pairsource(enc, src, report, chash, max_per_sign=400)
    axis_v, center, sep = _apply_orientation(scorer, axis)
    # count the labeled sample actually used (mirrors _labeled_sample)
    rows = src._rows()
    n_pos = sum(1 for r in rows if r[-1] == "+")
    n_neg = sum(1 for r in rows if r[-1] == "-")

    rec = {
        "axis": np.asarray(axis_v, dtype="float32"),
        "center": float(center),
        "scale": float(scorer.scale),
        "ckpt_sha": chash,
        "max_len": max_len,
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "seed": False,
        "validated": True,
        "orientation_sep": sep,
        "low_confidence": not (abs(sep) >= MIN_ORIENT_SEP),
    }
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default="/home/claude/xbse_ckpt")
    ap.add_argument("--out", default="/home/claude/xbse_ckpt/moral_axes.npz")
    ap.add_argument("--manifest", default=None, help="JSON of {axis:{pos,neg}}")
    ap.add_argument("--base-model", default="BAAI/bge-m3")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--axis", action="append", help="axis name (repeatable)")
    ap.add_argument("--all", action="store_true", help="fit every DEME10 axis")
    ap.add_argument("--seed", action="store_true",
                    help="use the SEED/manifest path instead of the validated "
                         "JointPairSource+gate path (proof / no-data fallback)")
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

    mode = "seed/manifest" if args.seed else "validated (from_pairsource+gate)"
    logger.info("fit mode: %s | device=%s | axes=%d", mode, args.device, len(axes))

    fit_ok = 0
    for axis in axes:
        try:
            if args.seed:
                rec = fit_axis(axis, ckpt_dir, manifest, args.base_model, args.device)
            else:
                rec = fit_axis_validated(axis, ckpt_dir, args.base_model, args.device)
        except Exception as exc:  # noqa: BLE001
            # NotValidatedError (gate fail / hash drift) lands here as "absent".
            logger.warning("axis %s: not fit — %s", axis, exc)
            rec = None
        if rec is not None:
            records[axis] = rec
            fit_ok += 1
            logger.info(
                "axis %s: FIT ok (dim=%d, center=%.3f, scale=%.3f, max_len=%d, "
                "n=%d/%d, validated=%s, orient_sep=%+.3f%s)",
                axis, len(rec["axis"]), rec["center"], rec["scale"],
                rec.get("max_len", 0), rec.get("n_pos", 0), rec.get("n_neg", 0),
                rec.get("validated", False), rec.get("orientation_sep", float("nan")),
                "  [LOW-CONFIDENCE]" if rec.get("low_confidence") else "",
            )

    if not records:
        logger.error("no axes fit; not writing %s", out)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: np.array(v, dtype=object) for k, v in records.items()})
    n_val = sum(1 for r in records.values() if r.get("validated"))
    n_low = sum(1 for r in records.values() if r.get("low_confidence"))
    logger.info("wrote %d axes (%d newly fit, %d validated, %d low-confidence) to %s",
                len(records), fit_ok, n_val, n_low, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
