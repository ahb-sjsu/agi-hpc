"""Factor analysis of the judge panel's OWN 31x7 harm-score matrix.

Answers the referee question the bifactor slide invites: the xbse bifactor
readout was measured on the learned-encoder instantiation — does the LLM
judge panel's 7-D space show the same one-strong-factor structure, measured
directly on the panel's scores?

Input: harm_validation_raw.json (the camera-ready validation panel capture:
31 scenarios x 6 models x 3 reps x 7-dim harm vector, base arm).
Output: printed table + panel_fa_result.json. Pure numpy; no fitting beyond
eigendecomposition; Horn's parallel analysis for factor count.
"""
import json
import sys

import numpy as np

DIMS = ["physical", "emotional", "financial", "autonomy", "trust", "social", "identity"]

src = sys.argv[1] if len(sys.argv) > 1 else "harm_validation_raw.json"
raw = json.load(open(src, encoding="utf-8"))["base"]

# consensus matrix: scenario x 7, mean over models and reps
scen_ids = sorted(raw.keys())
rows, pooled = [], []
for sid in scen_ids:
    vecs = []
    for model, reps in raw[sid].items():
        for rep, leaf in reps.items():
            v = leaf.get("vec")
            if v is not None and len(v) == 7:
                vecs.append(v)
                pooled.append(v)
    rows.append(np.mean(np.asarray(vecs, dtype=float), axis=0))
M = np.asarray(rows)              # 31 x 7 consensus
P = np.asarray(pooled, dtype=float)  # ~558 x 7 pooled (model x rep kept separate)
print(f"consensus matrix {M.shape}, pooled {P.shape}")


def pca_report(X, label, n_perm=2000, seed=0):
    Xc = X - X.mean(0)
    sd = Xc.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Z = Xc / sd                       # correlation-matrix PCA
    C = np.corrcoef(Z, rowvar=False)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    share = w / w.sum()
    pc1 = V[:, 0] * np.sign(V[:, 0].sum())  # orient positively
    # per-dim G-share analogue: squared loading on PC1 (variance of the dim
    # reproduced by the first component of the panel's own score space)
    load2 = pc1 ** 2 * w[0]
    # Horn's parallel analysis: permute each column independently
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, len(DIMS)))
    Zp = Z.copy()
    for i in range(n_perm):
        for j in range(Z.shape[1]):
            rng.shuffle(Zp[:, j])
        wn = np.linalg.eigvalsh(np.corrcoef(Zp, rowvar=False))[::-1]
        null[i] = wn
    thresh95 = np.percentile(null, 95, axis=0)
    n_keep = int(np.sum(w > thresh95))
    print(f"\n== {label} ==")
    print("eigenvalues:", np.round(w, 3).tolist())
    print("variance share:", np.round(share, 3).tolist())
    print(f"PC1 share = {share[0]:.3f}; parallel-analysis keeps {n_keep} factor(s) "
          f"(95th-pct null eig1 = {thresh95[0]:.3f})")
    print(f"{'dim':<10} {'PC1 loading':>12} {'PC1 R^2':>9}")
    for d, l, r2 in zip(DIMS, pc1, load2):
        print(f"{d:<10} {l:>12.3f} {r2:>9.3f}")
    return {
        "eigenvalues": np.round(w, 4).tolist(),
        "variance_share": np.round(share, 4).tolist(),
        "pc1_share": round(float(share[0]), 4),
        "parallel_analysis_factors": n_keep,
        "pa_null95_eig1": round(float(thresh95[0]), 4),
        "pc1_loadings": {d: round(float(l), 4) for d, l in zip(DIMS, pc1)},
        "pc1_r2": {d: round(float(r), 4) for d, r in zip(DIMS, load2)},
        "n_rows": int(X.shape[0]),
    }


out = {
    "consensus": pca_report(M, "consensus (31 scenarios, panel mean)"),
    "pooled": pca_report(P, "pooled (scenario x model x rep rows)"),
}

# ---- per-model FA: does the one-strong-factor structure hold inside each
# judge, or only in the aggregate? Each model gets its own 31x7 matrix
# (reps averaged), its own PCA + parallel analysis, and a Tucker congruence
# coefficient of its PC1 against the consensus PC1.
models = sorted({m for sid in scen_ids for m in raw[sid]})
cons_pc1 = np.array([out["consensus"]["pc1_loadings"][d] for d in DIMS])
per_model = {}
for model in models:
    rows_m = []
    for sid in scen_ids:
        reps = raw[sid].get(model, {})
        vecs = [leaf["vec"] for leaf in reps.values()
                if leaf.get("vec") is not None and len(leaf["vec"]) == 7]
        if vecs:
            rows_m.append(np.mean(np.asarray(vecs, dtype=float), axis=0))
    Xm = np.asarray(rows_m)
    rep = pca_report(Xm, f"model: {model} ({Xm.shape[0]} scenarios)")
    pc1_m = np.array([rep["pc1_loadings"][d] for d in DIMS])
    phi = float(np.dot(pc1_m, cons_pc1)
                / np.sqrt(np.dot(pc1_m, pc1_m) * np.dot(cons_pc1, cons_pc1)))
    rep["tucker_congruence_vs_consensus"] = round(phi, 4)
    print(f"  Tucker congruence vs consensus PC1: {phi:.3f}")
    per_model[model] = rep
out["per_model"] = per_model
shares = [r["pc1_share"] for r in per_model.values()]
phis = [r["tucker_congruence_vs_consensus"] for r in per_model.values()]
facs = [r["parallel_analysis_factors"] for r in per_model.values()]
out["per_model_summary"] = {
    "pc1_share_min": round(min(shares), 4), "pc1_share_max": round(max(shares), 4),
    "congruence_min": round(min(phis), 4), "congruence_max": round(max(phis), 4),
    "pa_factors": {m: per_model[m]["parallel_analysis_factors"] for m in per_model},
}

json.dump(out, open("panel_fa_result.json", "w"), indent=1)
print("\nSAVED panel_fa_result.json")
print("DONE PANEL_FA pc1_consensus=%.3f pc1_pooled=%.3f factors=%d/%d "
      "permodel_pc1=[%.3f,%.3f] congruence=[%.3f,%.3f]" % (
          out["consensus"]["pc1_share"], out["pooled"]["pc1_share"],
          out["consensus"]["parallel_analysis_factors"],
          out["pooled"]["parallel_analysis_factors"],
          min(shares), max(shares), min(phis), max(phis)))
