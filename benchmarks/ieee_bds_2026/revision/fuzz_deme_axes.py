#!/usr/bin/env python3
"""
Structural-fuzzing campaign over the DEME v3 moral-vector axes.

Question: which of the 9 DEME axes does the GenevaEMV3 typed-verdict surface
*functionally* depend on — in what combinations, and with what flip thresholds?
This is the interventional complement to the panel FA / bifactor analysis
(which characterizes variance structure, not behavioral dependence).

Model under fuzz: a vectorized, exact reimplementation of
GenevaEMV3.evaluate_vector()'s facts -> 9-axis vector -> verdict rule,
parameterized per axis:
  - inactive (param >= 1e5): axis neutralized (harm=0, others=1) and its
    veto channel disabled — the causal ablation.
  - active with gain w: the axis's deviation from neutral is scaled by w
    (harm: v' = clip(w*v); others: v' = clip(1 - w*(1-v))). w = 1 is the
    production kernel. Boolean-driven hard zeros (rights violation,
    discrimination, explicit rule) keep their veto iff w >= 1.

Cases: the 31-scenario x 3-condition x 6-model extracted-facts corpus from
deme_verdicts_raw.json (the 161-case verdict-flip study of the BDS paper).
Reference behavior = full kernel (all w = 1) on the same cases; the script
also validates the reimplementation against the stored verdicts.

evaluate_fn readout:
  mae    = mean |ordinal verdict shift| vs the full kernel, over all cases
  errors = per-condition mean signed shift + euphemism flip-rate excess
           (does ablation change the measured exploit?)

Run on Atlas (CPU, numpy + structural-fuzzing only). Seed-free: the model
is deterministic; only subset-search sampling uses the framework's RNG.
"""
import json
import os
from dataclasses import asdict, is_dataclass

import numpy as np
from structural_fuzzing import run_campaign

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "deme_verdicts_raw.json")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

AXES = [
    "physical_harm", "rights_respect", "fairness_equity", "autonomy_respect",
    "privacy_protection", "societal_environmental", "virtue_care",
    "legitimacy_trust", "epistemic_quality",
]
ORD = {"forbid": 0, "avoid": 1, "neutral": 2, "prefer": 3, "strongly_prefer": 4}
CONDS = ["neutral", "euphemistic", "dramatic"]
INACTIVE_THRESH = 1e5

# ---------------------------------------------------------------- load cases
raw = json.load(open(RAW, encoding="utf-8"))
h, u = [], []
b_vr, b_d, b_er, b_c, b_ev, b_pi, b_rdc = [], [], [], [], [], [], []
cond_ix, pair_key, stored_ord = [], [], []
for sid, conds in raw.items():
    for ci, cond in enumerate(CONDS):
        for model, rec in (conds.get(cond) or {}).items():
            if not rec or rec.get("verdict") not in ORD or not rec.get("facts"):
                continue
            f = rec["facts"]
            try:
                eh = min(1.0, max(0.0, float(f.get("expected_harm", 0.0))))
                un = min(1.0, max(0.0, float(f.get("uncertainty", 0.0))))
            except (TypeError, ValueError):
                continue
            h.append(eh); u.append(un)
            b_vr.append(bool(f.get("violates_rights", False)))
            b_d.append(bool(f.get("discriminates_protected", False)))
            b_er.append(bool(f.get("violates_explicit_rule", False)))
            b_c.append(f.get("has_valid_consent", True) is not False)
            b_ev.append(bool(f.get("exploits_vulnerable", False)))
            b_pi.append(bool(f.get("exacerbates_power_imbalance", False)))
            b_rdc.append(bool(f.get("role_duty_conflict", False)))
            cond_ix.append(ci)
            pair_key.append((sid, model))
            stored_ord.append(ORD[rec["verdict"]])

h = np.array(h); u = np.array(u)
b_vr = np.array(b_vr); b_d = np.array(b_d); b_er = np.array(b_er)
b_c = np.array(b_c); b_ev = np.array(b_ev); b_pi = np.array(b_pi)
b_rdc = np.array(b_rdc)
cond_ix = np.array(cond_ix)
stored_ord = np.array(stored_ord)
n_cases = len(h)

# Full-kernel base axis values, exactly as GenevaEMV3.evaluate_vector builds
# them from the extractor's EthicalFacts (evidence_quality defaults "medium",
# hence the 0.8 factor on epistemic).
base = np.ones((9, n_cases))
base[0] = h
base[1] = np.where(b_vr, 0.0, 1.0)
base[2] = np.where(b_d, 0.0, 1.0) * np.where(b_ev, 0.5, 1.0) * np.where(b_pi, 0.8, 1.0)
base[3] = np.where(b_c, 1.0, 0.5)
base[7] = np.where(b_er, 0.0, 1.0) * np.where(b_rdc, 0.7, 1.0)
base[8] = (1.0 - u) * 0.8

# Euphemism-flip pairing: (scenario, model) present in both neutral and euph
ix_by_key = {}
for i, (key, ci) in enumerate(zip(pair_key, cond_ix)):
    ix_by_key.setdefault(key, {})[ci] = i
pairs = [(v[0], v[1]) for v in ix_by_key.values() if 0 in v and 1 in v]
pair_neu = np.array([p[0] for p in pairs])
pair_euph = np.array([p[1] for p in pairs])


def kernel(params):
    """Vectorized parameterized GenevaEMV3 verdict over all cases."""
    p = np.asarray(params, dtype=float)
    act = p < INACTIVE_THRESH
    w = np.where(act, p, 1.0)

    v = np.empty_like(base)
    # harm axis: neutral = 0
    v[0] = np.clip(w[0] * base[0], 0.0, 1.0) if act[0] else 0.0
    # remaining axes: neutral = 1
    for i in range(1, 9):
        v[i] = np.clip(1.0 - w[i] * (1.0 - base[i]), 0.0, 1.0) if act[i] else 1.0

    hard = w >= 1.0 - 1e-12
    veto = ((b_vr & act[1] & hard[1])
            | (b_d & act[2] & hard[2])
            | (b_er & act[7] & hard[7]))

    gate_hi = (v[1] > 0.7) & (v[2] > 0.7) & (v[7] > 0.7)
    gate_mid = (v[1] > 0.5) & (v[2] > 0.5) & (v[7] > 0.5)
    return np.where(veto, 0,
                    np.where(gate_hi, np.where(v[0] < 0.2, 4, 3),
                             np.where(gate_mid, 2, 1)))


ref = kernel(np.ones(9))
match_rate = float((ref == stored_ord).mean())


def flip_rate(verd):
    return float(((verd[pair_neu] <= 1) & (verd[pair_euph] >= 3)).mean())


flip_ref = flip_rate(ref)


def evaluate_fn(params):
    verd = kernel(params)
    dev = verd - ref
    mae = float(np.abs(dev).mean())
    errors = {
        f"{c}_shift": float(dev[cond_ix == ci].mean())
        for ci, c in enumerate(CONDS)
    }
    errors["euph_flip_excess"] = flip_rate(verd) - flip_ref
    return mae, errors


def _ser(obj):
    if is_dataclass(obj):
        return {k: _ser(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _ser(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ser(v) for v in obj]
    return obj


def main():
    print(f"cases: {n_cases} ({len(pairs)} neutral/euph pairs)")
    print(f"reimplementation vs stored verdicts: {match_rate:.3f} match")
    print(f"full-kernel euphemism flip rate: {flip_ref:.3f}")
    assert match_rate > 0.95, "kernel reimplementation drifted from stored verdicts"

    report = run_campaign(
        dim_names=AXES,
        evaluate_fn=evaluate_fn,
        max_subset_dims=4,
        n_grid=20,
        n_random=1500,
        n_mri_perturbations=300,
        run_baselines=True,
        verbose=True,
    )

    summary = report.summary()
    print(summary)
    with open(os.path.join(OUT, "fuzz_deme_axes_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"cases={n_cases} match_rate={match_rate:.4f} "
                 f"flip_ref={flip_ref:.4f}\n\n")
        fh.write(summary)

    out = {
        "n_cases": n_cases,
        "n_pairs": len(pairs),
        "reimpl_match_rate": match_rate,
        "full_kernel_euph_flip_rate": flip_ref,
        "best_subsets_by_size": {},
        "sensitivity": [_ser(s) for s in report.sensitivity_results],
        "mri": _ser(report.mri_result),
        "adversarial": [_ser(a) for a in report.adversarial_results],
        "composition": _ser(report.composition_result),
        "pareto": [_ser(p) for p in report.pareto_results],
    }
    by_size = {}
    for r in report.subset_results:
        k = len(r.dim_names)
        if k not in by_size or r.mae < by_size[k].mae:
            by_size[k] = r
    out["best_subsets_by_size"] = {str(k): _ser(v) for k, v in sorted(by_size.items())}
    with open(os.path.join(OUT, "fuzz_deme_axes_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote out/fuzz_deme_axes_report.txt and out/fuzz_deme_axes_results.json")


if __name__ == "__main__":
    main()
