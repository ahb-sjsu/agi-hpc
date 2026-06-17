#!/usr/bin/env python3
"""
Harm-Space Validation Study  (IEEE BigDataService 2026, paper 0204 revision)

Answers reviewer #1 ("validate the harm space; inter-model / human agreement on
the vectors") with an inter-model agreement study over an open, multi-family
model panel served by the NRP managed LLM API (https://ellm.nrp-nautilus.io/v1).

For each (model, scenario) we elicit the paper's 7-D harm vector
(physical, emotional, financial, autonomy, trust, social, identity; each 0-10)
via schema-constrained JSON, with R replications per cell (test-retest).

Outputs per-dimension:
  * ICC(2,k)  -- two-way random, absolute-agreement, average-measures
                 (inter-model reliability)
  * Krippendorff's alpha (interval)            (inter-model reliability)
  * within-model test-retest reliability       (stochastic floor)

Also runs the salience case study on the 6 gold scenarios: euphemistic /
dramatic rewrites vs neutral, panel-averaged harm displacement (0-70).

Self-contained: stdlib + numpy only. Run on Atlas (has /home/claude/.llmtoken).
"""
import json, os, re, sys, time, math, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ---------------------------------------------------------------- config
BASE   = "https://ellm.nrp-nautilus.io/v1"
TOKEN  = open(os.path.expanduser("~/.llmtoken")).read().strip()
MODELS = ["qwen3", "qwen3-small", "glm-5", "gpt-oss", "minimax-m2", "gemma"]  # 6 distinct, 5 families
DIMS   = ["physical", "emotional", "financial", "autonomy", "trust", "social", "identity"]
REPS   = 3                       # replications per (model, scenario) for test-retest
WORKERS= 8
HERE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("BDS_OUT", os.path.join(HERE, "out")); os.makedirs(OUTDIR, exist_ok=True)

SYS = (
    "You are a careful moral-harm rater. For the scenario, rate the HARM along "
    "seven independent dimensions, each an integer 0-10 (0 = none, 10 = extreme):\n"
    "  physical  - bodily harm or risk\n"
    "  emotional - psychological / emotional harm\n"
    "  financial - monetary / material harm\n"
    "  autonomy  - violation of agency / self-determination / consent\n"
    "  trust     - betrayal or breach of trust\n"
    "  social    - damage to relationships / community standing\n"
    "  identity  - harm to dignity / sense of self / character\n"
    "Then give a verdict (one of NTA, YTA, ESH, NAH) and confidence 0-10.\n"
    "Respond with ONLY a single JSON object, keys exactly: "
    "physical, emotional, financial, autonomy, trust, social, identity, verdict, confidence. "
    "No commentary, no markdown."
)

# ---------------------------------------------------------------- LLM call
def _clean_json(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S|re.I)
    txt = re.sub(r"```(?:json)?", "", txt)
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m: return None
    blob = re.sub(r",\s*([}\]])", r"\1", m.group(0))  # trailing commas
    try: return json.loads(blob)
    except Exception: return None

MAXTOK = 1600   # reasoning models (qwen3-397B, glm-5, gpt-oss) need room past CoT

def _vec_from(obj):
    if not obj or not all(k in obj for k in DIMS): return None
    try: vec=[float(obj[k]) for k in DIMS]
    except Exception: return None
    return vec if all(0<=v<=10 for v in vec) else None

def call(model, scenario_text, temperature, max_retry=4):
    body = {
        "model": model, "temperature": temperature, "max_tokens": MAXTOK,
        "messages": [{"role":"system","content":SYS},
                     {"role":"user","content":"Scenario:\n"+scenario_text}],
        # disable chain-of-thought where supported (harmless/ignored otherwise) so the
        # token budget goes to the JSON answer, not reasoning. We still parse `reasoning`
        # as a fallback for models that emit the JSON only inside their CoT.
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "low",
    }
    payload = json.dumps(body).encode()
    for attempt in range(max_retry):
        try:
            req = urllib.request.Request(BASE+"/chat/completions", data=payload,
                  headers={"Authorization":"Bearer "+TOKEN,"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read().decode())
            msg = (d.get("choices") or [{}])[0].get("message", {}) or {}
            vec = _vec_from(_clean_json(msg.get("content"))) \
                  or _vec_from(_clean_json(msg.get("reasoning")))
            if vec is not None:
                obj = _clean_json(msg.get("content")) or _clean_json(msg.get("reasoning")) or {}
                return {"vec":vec, "verdict":str(obj.get("verdict","")),
                        "confidence":obj.get("confidence")}
            # malformed / empty -> retry
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            time.sleep(2*(attempt+1))
        except Exception:
            time.sleep(1.5*(attempt+1))
    return None

# ---------------------------------------------------------------- stats
def icc_2k(M):
    """ICC(2,k) absolute agreement, average measures. M: n_subjects x k_raters."""
    M = np.asarray(M, float)
    n, k = M.shape
    if n < 2 or k < 2: return float("nan")
    grand = M.mean()
    rows  = M.mean(axis=1); cols = M.mean(axis=0)
    SST = ((M-grand)**2).sum()
    SSR = k*((rows-grand)**2).sum();  MSR = SSR/(n-1)
    SSC = n*((cols-grand)**2).sum();  MSC = SSC/(k-1)
    SSE = SST-SSR-SSC;                MSE = SSE/((n-1)*(k-1))
    denom = MSR + (MSC-MSE)/n
    return float("nan") if denom==0 else (MSR-MSE)/denom

def krippendorff_interval(M):
    """Interval Krippendorff alpha for COMPLETE rectangular data.
    M: n_units x k_raters. Canonical form alpha = 1 - Do/De with squared-diff
    metric (Krippendorff 2011). Vectorised via sum-of-squared-pairwise identity
    Sum_{a!=b}(x_a-x_b)^2 = 2*N*Sum(x^2) - 2*(Sum x)^2.
    """
    M = np.asarray(M, float)
    N, k = M.shape
    if N < 2 or k < 2: return float("nan")
    # observed: within-unit pairwise, normalised by N*k*(k-1)
    rs1 = M.sum(axis=1); rs2 = (M**2).sum(axis=1)
    do_num = (2*k*rs2 - 2*rs1**2).sum()
    Do = do_num / (N*k*(k-1))
    # expected: across all pooled values
    flat = M.flatten(); n_tot = flat.size
    S1 = flat.sum(); S2 = (flat**2).sum()
    de_num = 2*n_tot*S2 - 2*S1**2
    De = de_num / (n_tot*(n_tot-1))
    return float(1.0 - Do/De) if De > 0 else float("nan")

def _crosscheck(M):
    """Optional library cross-check; silent if libs absent."""
    out = {}
    try:
        import krippendorff as _k
        out["krip_lib"] = round(float(_k.alpha(reliability_data=np.asarray(M,float).T,
                                                level_of_measurement="interval")), 3)
    except Exception: pass
    try:
        import pingouin as pg, pandas as pd
        n,k = np.asarray(M).shape
        rows=[]
        for i in range(n):
            for j in range(k):
                rows.append({"target":i,"rater":j,"score":float(M[i][j])})
        df=pd.DataFrame(rows)
        icc=pg.intraclass_corr(data=df,targets="target",raters="rater",nan_policy="omit")
        v=icc[icc["Type"]=="ICC2k"]["ICC"]
        if len(v): out["icc2k_pingouin"]=round(float(v.iloc[0]),3)
    except Exception: pass
    return out

# ---------------------------------------------------------------- run
def main():
    scen = json.load(open(os.path.join(HERE,"scenarios.json"), encoding="utf-8"))
    print(f"[run] {len(scen)} scenarios x {len(MODELS)} models x {REPS} reps "
          f"= {len(scen)*len(MODELS)*REPS} base calls", flush=True)

    # build job list: base ratings (rep 0 temp=0; reps>0 temp=0.7)
    jobs = []
    for s in scen:
        for model in MODELS:
            for r in range(REPS):
                jobs.append(("base", s["id"], model, r, 0.0 if r==0 else 0.7, s["text"]))
    # case study: gold euphemistic / dramatic (single, temp=0)
    gold = [s for s in scen if s["source"]=="GOLD_SET"]
    for s in gold:
        for model in MODELS:
            for cond in ("euphemistic","dramatic"):
                if s.get(cond):
                    jobs.append((cond, s["id"], model, 0, 0.0, s[cond]))
    print(f"[run] total jobs incl. case study = {len(jobs)}", flush=True)

    raw = {}
    done = 0; fail = 0
    def work(job):
        kind,sid,model,rep,temp,text = job
        res = call(model, text, temp)
        return job, res
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for fu in as_completed(futs):
            job, res = fu.result()
            kind,sid,model,rep,temp,text = job
            raw.setdefault(kind, {}).setdefault(sid, {}).setdefault(model, {})[rep] = res
            done += 1
            if res is None: fail += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} done ({fail} failed)", flush=True)
    print(f"[run] complete: {done} done, {fail} failed", flush=True)
    json.dump(raw, open(os.path.join(OUTDIR,"harm_validation_raw.json"),"w"), indent=1)

    # -------- assemble matrices: per dimension, scenario x model (mean over reps)
    base = raw["base"]
    sids = [s["id"] for s in scen if s["id"] in base]
    analysis = {"n_scenarios":len(sids),"models":MODELS,"dims":DIMS,
                "per_dimension":{}, "retest":{}, "case_study":{}}
    # inter-model ICC + alpha per dim
    for di,dim in enumerate(DIMS):
        Mat = []
        keep_sids=[]
        for sid in sids:
            row=[]; ok=True
            for model in MODELS:
                reps = base[sid].get(model,{})
                vals = [reps[r]["vec"][di] for r in reps if reps[r]]
                if not vals: ok=False; break
                row.append(np.mean(vals))
            if ok:
                Mat.append(row); keep_sids.append(sid)
        Mat=np.array(Mat)
        analysis["per_dimension"][dim] = {
            "n": int(Mat.shape[0]),
            "icc_2k": round(icc_2k(Mat),3),
            "krippendorff_interval": round(krippendorff_interval(Mat),3),
            "model_means": {m: round(float(Mat[:,j].mean()),2) for j,m in enumerate(MODELS)},
        }
    # overall (pool dims as one matrix scenario*dim x model)
    big=[]
    for sid in sids:
        for di in range(len(DIMS)):
            row=[]; ok=True
            for model in MODELS:
                reps=base[sid].get(model,{})
                vals=[reps[r]["vec"][di] for r in reps if reps[r]]
                if not vals: ok=False;break
                row.append(np.mean(vals))
            if ok: big.append(row)
    big=np.array(big)
    analysis["overall"]={"icc_2k":round(icc_2k(big),3),
                         "krippendorff_interval":round(krippendorff_interval(big),3),
                         "n_rows":int(big.shape[0]),
                         "library_crosscheck":_crosscheck(big)}
    # within-model test-retest: correlation of rep0 vs mean(rep1..) across all scenario*dim
    for model in MODELS:
        a=[]; b=[]
        for sid in sids:
            reps=base[sid].get(model,{})
            if 0 in reps and reps[0] and any(reps.get(r) for r in reps if r>0):
                later=[reps[r]["vec"] for r in reps if r>0 and reps[r]]
                if not later: continue
                v0=reps[0]["vec"]; vl=np.mean(later,axis=0)
                a.extend(v0); b.extend(list(vl))
        if len(a)>3:
            r=float(np.corrcoef(a,b)[0,1])
            analysis["retest"][model]={"pearson_r":round(r,3),"n":len(a)}
    # -------- case study: panel-averaged total-harm displacement on gold
    for cond in ("euphemistic","dramatic"):
        if cond not in raw: continue
        deltas=[]
        for sid in raw[cond]:
            # neutral total (panel mean), perturbed total (panel mean)
            neut=[]; pert=[]
            for model in MODELS:
                nb=base.get(sid,{}).get(model,{})
                nvals=[sum(nb[r]["vec"]) for r in nb if nb[r]]
                pc=raw[cond][sid].get(model,{})
                pvals=[sum(pc[r]["vec"]) for r in pc if pc[r]]
                if nvals and pvals:
                    neut.append(np.mean(nvals)); pert.append(np.mean(pvals))
            if neut and pert:
                deltas.append(np.mean(pert)-np.mean(neut))
        if deltas:
            analysis["case_study"][cond]={
                "mean_total_harm_delta": round(float(np.mean(deltas)),2),
                "min": round(float(np.min(deltas)),2),
                "max": round(float(np.max(deltas)),2),
                "n_gold": len(deltas)}
    json.dump(analysis, open(os.path.join(OUTDIR,"harm_validation_analysis.json"),"w"), indent=2)
    print("\n=== ANALYSIS ===")
    print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    main()
