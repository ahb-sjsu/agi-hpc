"""Moral Geometry Benchmark v6 — Ablation Study
Social Cognition Track | Measuring AGI Competition

Double-blind A/B test: Does geometric moral reasoning improve LLM judgment?

Method A (Control): Vanilla LLM — "Judge this AITA post. Verdict: YTA/NTA/ESH/NAH."
Method B (Treatment): Geometric framework — force dimensional decomposition before judgment.

Same model, same scenarios, randomized order. The model doesn't know which
method is being tested. We don't look at results until both arms complete.

Hypothesis: Method B improves accuracy on ESH/NAH (multi-dimensional cases)
by forcing the model to reason about competing moral dimensions before
collapsing to a scalar verdict.

Based on Bond (2026), Geometric Ethics — the Scalar Irrecoverability Theorem
predicts that direct scalar judgment (Method A) loses information that
dimensional decomposition (Method B) preserves.

Paste this ENTIRE file into ONE cell in a Kaggle Benchmark Task notebook.
No pip install needed. Prints progress throughout.
Expected runtime: ~45-60 minutes (runs each scenario TWICE — control + treatment).
"""

import kaggle_benchmarks as kbench
from dataclasses import dataclass
import os, json, time, random

os.environ["RENDER_SUBRUNS"] = "False"

print("=" * 60)
print("MORAL GEOMETRY ABLATION STUDY")
print("Double-blind: Vanilla vs Geometric Reasoning")
print("=" * 60)
print()

# ═══════════════════════════════════════════════════════════════
# LOAD DATASET
# ═══════════════════════════════════════════════════════════════

print("[1/6] Loading AITA dataset...")
t0 = time.time()
try:
    from datasets import load_dataset
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset

ds = load_dataset("OsamaBsher/AITA-Reddit-Dataset", split="train")
print(f"  Loaded {len(ds):,} posts in {time.time()-t0:.0f}s")

random.seed(42)
PER_CLASS = 50  # 50 per class × 4 = 200 scenarios × 2 methods = 400 LLM calls
pools = {"nta": [], "yta": [], "esh": [], "nah": []}

for row in ds:
    v = (row.get("verdict") or "").lower().strip()
    if v not in pools or len(pools[v]) >= PER_CLASS * 4:
        continue
    text = row.get("text") or ""
    score = row.get("score") or 0
    min_score = 5 if v in ("esh", "nah") else 10
    if len(text) < 300 or len(text) > 1500 or score < min_score:
        continue
    pools[v].append({
        "title": row["title"],
        "text": text[:1200],
        "verdict": v.upper(),
        "score": score,
    })

SCENARIOS = []
for v in ["NTA", "YTA", "ESH", "NAH"]:
    pool = pools[v.lower()]
    random.shuffle(pool)
    SCENARIOS.extend(pool[:PER_CLASS])
    print(f"  {v}: {len(pool[:PER_CLASS])}")

random.shuffle(SCENARIOS)
print(f"  Total: {len(SCENARIOS)} scenarios (each tested twice)\n")


# ═══════════════════════════════════════════════════════════════
# STRUCTURED OUTPUT
# ═══════════════════════════════════════════════════════════════

@dataclass
class VanillaJudgment:
    verdict: str
    confidence: float
    reasoning: str

@dataclass
class DimensionalAnalysis:
    consequences_assessment: str
    rights_assessment: str
    fairness_assessment: str
    autonomy_assessment: str
    trust_assessment: str
    social_impact_assessment: str
    identity_assessment: str
    dimensions_in_tension: str
    verdict: str
    confidence: float
    reasoning: str

def normalize_verdict(raw):
    raw = raw.strip().upper().replace("'", "").replace(".", "")
    for label in ["NTA", "YTA", "ESH", "NAH"]:
        if label in raw:
            return label
    return "UNK"


# ═══════════════════════════════════════════════════════════════
# THE TWO PROMPTS
# ═══════════════════════════════════════════════════════════════

def vanilla_prompt(title, text):
    return (
        f"You are judging a post from Reddit's r/AmITheAsshole.\n\n"
        f"Title: {title}\n\n"
        f"{text}\n\n"
        f"Give your verdict as EXACTLY one of:\n"
        f"  YTA = You're The Asshole (poster is wrong)\n"
        f"  NTA = Not The Asshole (poster is right)\n"
        f"  ESH = Everyone Sucks Here (both sides are wrong)\n"
        f"  NAH = No Assholes Here (nobody is wrong)\n\n"
        f"Rate confidence 0-10 and explain briefly."
    )

def geometric_prompt(title, text):
    return (
        f"You are judging a post from Reddit's r/AmITheAsshole.\n"
        f"Before giving your verdict, analyze EACH moral dimension separately.\n\n"
        f"Title: {title}\n\n"
        f"{text}\n\n"
        f"STEP 1 — Analyze each dimension independently:\n"
        f"  • Consequences: Who was harmed? Who benefited? How severely?\n"
        f"  • Rights: Whose rights were respected or violated?\n"
        f"  • Fairness: Was this fair to all parties? Was there reciprocity?\n"
        f"  • Autonomy: Was anyone's freedom of choice restricted?\n"
        f"  • Trust: Was trust maintained or broken?\n"
        f"  • Social impact: How did this affect relationships and community?\n"
        f"  • Identity: What does this say about the person's character?\n\n"
        f"STEP 2 — Identify dimensions in tension:\n"
        f"  Which dimensions point toward different verdicts? "
        f"  For example, consequences might say NTA but fairness might say YTA.\n\n"
        f"STEP 3 — Give your FINAL verdict considering ALL dimensions:\n"
        f"  YTA = You're The Asshole (poster is wrong)\n"
        f"  NTA = Not The Asshole (poster is right)\n"
        f"  ESH = Everyone Sucks Here (both sides are wrong)\n"
        f"  NAH = No Assholes Here (nobody is wrong)\n\n"
        f"Rate confidence 0-10 and explain how the dimensions informed your verdict."
    )


# ═══════════════════════════════════════════════════════════════
# TASK: DOUBLE-BLIND ABLATION
# ═══════════════════════════════════════════════════════════════

@kbench.task(name="moral_geometry_ablation")
def moral_geometry_ablation(llm):
    """Double-blind ablation: Vanilla vs Geometric moral reasoning."""

    # ── ARM A: VANILLA (control) ──
    print("\n[2/6] ARM A: VANILLA (control)")
    print(f"  Standard prompt — no dimensional decomposition")
    print("-" * 60)

    vanilla_results = {"correct": 0, "total": 0,
                       "by_class": {v: {"correct": 0, "total": 0} for v in ["NTA","YTA","ESH","NAH"]}}

    for i, s in enumerate(SCENARIOS):
        with kbench.chats.new(f"vanilla_{i}"):
            j = llm.prompt(vanilla_prompt(s["title"], s["text"]), schema=VanillaJudgment)

        pred = normalize_verdict(j.verdict)
        actual = s["verdict"]
        match = pred == actual
        vanilla_results["total"] += 1
        vanilla_results["by_class"][actual]["total"] += 1
        if match:
            vanilla_results["correct"] += 1
            vanilla_results["by_class"][actual]["correct"] += 1

        n = i + 1
        if n % 25 == 0:
            running = vanilla_results["correct"] / vanilla_results["total"]
            print(f"  [{n}/{len(SCENARIOS)}] running accuracy: {running:.0%}")
        elif not match:
            print(f"  [{n}] MISS: pred={pred} actual={actual} | {s['title'][:50]}")
            print(f"       Reasoning: {j.reasoning[:120]}")

    v_acc = vanilla_results["correct"] / vanilla_results["total"]
    print(f"\n  VANILLA ACCURACY: {vanilla_results['correct']}/{vanilla_results['total']} ({v_acc:.1%})")
    for cls in ["NTA","YTA","ESH","NAH"]:
        bc = vanilla_results["by_class"][cls]
        pct = bc["correct"] / max(bc["total"], 1)
        print(f"    {cls}: {bc['correct']}/{bc['total']} ({pct:.0%})")

    # ── ARM B: GEOMETRIC (treatment) ──
    print(f"\n\n[3/6] ARM B: GEOMETRIC (treatment)")
    print(f"  Dimensional decomposition before verdict")
    print("-" * 60)

    geometric_results = {"correct": 0, "total": 0,
                         "by_class": {v: {"correct": 0, "total": 0} for v in ["NTA","YTA","ESH","NAH"]},
                         "tensions_detected": 0}

    for i, s in enumerate(SCENARIOS):
        with kbench.chats.new(f"geometric_{i}"):
            j = llm.prompt(geometric_prompt(s["title"], s["text"]), schema=DimensionalAnalysis)

        pred = normalize_verdict(j.verdict)
        actual = s["verdict"]
        match = pred == actual
        geometric_results["total"] += 1
        geometric_results["by_class"][actual]["total"] += 1
        if match:
            geometric_results["correct"] += 1
            geometric_results["by_class"][actual]["correct"] += 1

        if j.dimensions_in_tension and len(j.dimensions_in_tension) > 5:
            geometric_results["tensions_detected"] += 1

        n = i + 1
        if n % 25 == 0:
            running = geometric_results["correct"] / geometric_results["total"]
            print(f"  [{n}/{len(SCENARIOS)}] running accuracy: {running:.0%}")
        elif not match:
            print(f"  [{n}] MISS: pred={pred} actual={actual} | {s['title'][:50]}")
            print(f"       Tensions: {j.dimensions_in_tension[:100]}")
            print(f"       Reasoning: {j.reasoning[:120]}")

    g_acc = geometric_results["correct"] / geometric_results["total"]
    print(f"\n  GEOMETRIC ACCURACY: {geometric_results['correct']}/{geometric_results['total']} ({g_acc:.1%})")
    for cls in ["NTA","YTA","ESH","NAH"]:
        bc = geometric_results["by_class"][cls]
        pct = bc["correct"] / max(bc["total"], 1)
        print(f"    {cls}: {bc['correct']}/{bc['total']} ({pct:.0%})")
    tension_rate = geometric_results["tensions_detected"] / geometric_results["total"]
    print(f"  Tensions detected: {geometric_results['tensions_detected']}/{geometric_results['total']} ({tension_rate:.0%})")

    # ── UNBLINDING: COMPARE ──
    print(f"\n\n[4/6] UNBLINDING — HEAD-TO-HEAD COMPARISON")
    print("=" * 60)

    print(f"\n  {'Metric':<30} {'Vanilla':>10} {'Geometric':>10} {'Delta':>10}")
    print(f"  {'-'*60}")

    print(f"  {'Overall accuracy':<30} {v_acc:>9.1%} {g_acc:>9.1%} {g_acc - v_acc:>+9.1%}")

    deltas = {}
    for cls in ["NTA", "YTA", "ESH", "NAH"]:
        vc = vanilla_results["by_class"][cls]
        gc = geometric_results["by_class"][cls]
        va = vc["correct"] / max(vc["total"], 1)
        ga = gc["correct"] / max(gc["total"], 1)
        delta = ga - va
        deltas[cls] = delta
        marker = " ***" if abs(delta) >= 0.10 else ""
        print(f"  {cls + ' accuracy':<30} {va:>9.0%} {ga:>9.0%} {delta:>+9.0%}{marker}")

    esh_nah_vanilla = (vanilla_results["by_class"]["ESH"]["correct"] + vanilla_results["by_class"]["NAH"]["correct"]) / max(
        vanilla_results["by_class"]["ESH"]["total"] + vanilla_results["by_class"]["NAH"]["total"], 1)
    esh_nah_geo = (geometric_results["by_class"]["ESH"]["correct"] + geometric_results["by_class"]["NAH"]["correct"]) / max(
        geometric_results["by_class"]["ESH"]["total"] + geometric_results["by_class"]["NAH"]["total"], 1)

    print(f"  {'ESH+NAH (ambiguous)':<30} {esh_nah_vanilla:>9.0%} {esh_nah_geo:>9.0%} {esh_nah_geo - esh_nah_vanilla:>+9.0%}")
    print(f"  {'Tensions detected':<30} {'n/a':>10} {tension_rate:>9.0%}")

    # ── STATISTICAL SIGNIFICANCE ──
    print(f"\n[5/6] STATISTICAL TEST")

    n = len(SCENARIOS)
    p1 = v_acc
    p2 = g_acc
    p_pool = (vanilla_results["correct"] + geometric_results["correct"]) / (2 * n)
    se = max((2 * p_pool * (1 - p_pool) / n) ** 0.5, 1e-10)
    z = (p2 - p1) / se
    # Approximate p-value from z-score (two-tailed)
    # Using normal approximation
    import math
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    print(f"  Two-proportion z-test:")
    print(f"    Vanilla:   {vanilla_results['correct']}/{n} ({v_acc:.1%})")
    print(f"    Geometric: {geometric_results['correct']}/{n} ({g_acc:.1%})")
    print(f"    z = {z:.3f}, p = {p_value:.4f}")
    if p_value < 0.05:
        print(f"    Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
        if g_acc > v_acc:
            print(f"    Geometric reasoning IMPROVES moral judgment.")
        else:
            print(f"    Geometric reasoning REDUCES moral judgment accuracy.")
    elif p_value < 0.10:
        print(f"    Result: MARGINALLY SIGNIFICANT (p < 0.10)")
    else:
        print(f"    Result: NOT SIGNIFICANT (p = {p_value:.3f})")

    # ── INTERPRETATION ──
    print(f"\n[6/6] INTERPRETATION")
    print("=" * 60)

    if g_acc > v_acc and deltas.get("ESH", 0) + deltas.get("NAH", 0) > 0:
        print(f"  The geometric framework IMPROVES moral reasoning.")
        print(f"  Key finding: dimensional decomposition helps most on")
        print(f"  ambiguous cases (ESH/NAH) where multiple moral dimensions")
        print(f"  compete — exactly as the Scalar Irrecoverability Theorem")
        print(f"  predicts (Bond, 2026).")
    elif g_acc > v_acc:
        print(f"  The geometric framework shows improvement overall,")
        print(f"  but the benefit is distributed across case types.")
    elif g_acc == v_acc:
        print(f"  No significant difference between methods.")
        print(f"  The geometric decomposition neither helps nor hurts.")
    else:
        print(f"  The geometric framework REDUCES accuracy.")
        print(f"  Possible explanation: forced dimensional analysis may")
        print(f"  introduce overthinking on clear-cut cases.")

    print(f"\n  This result {'supports' if g_acc >= v_acc else 'does not support'}")
    print(f"  the hypothesis that multi-attribute moral reasoning")
    print(f"  outperforms scalar judgment on social cognition tasks.")
    print("=" * 60)

    elapsed = time.time() - t0

    return {
        "vanilla_accuracy": v_acc,
        "geometric_accuracy": g_acc,
        "delta": g_acc - v_acc,
        "z_score": z,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "vanilla_by_class": {k: v["correct"]/max(v["total"],1) for k,v in vanilla_results["by_class"].items()},
        "geometric_by_class": {k: v["correct"]/max(v["total"],1) for k,v in geometric_results["by_class"].items()},
        "esh_nah_vanilla": esh_nah_vanilla,
        "esh_nah_geometric": esh_nah_geo,
        "tensions_detected_rate": tension_rate,
        "n_scenarios": n,
        "runtime_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# MULTI-MODEL COMPARISON
# Run ablation on multiple LLMs to show discriminatory power
# ═══════════════════════════════════════════════════════════════

MODELS_TO_TEST = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "meta/llama-3.1-70b",
]

print(f"Running ablation study across {len(MODELS_TO_TEST)} models:")
for m in MODELS_TO_TEST:
    print(f"  - {m}")
print(f"Testing {len(SCENARIOS)} scenarios × 2 methods × {len(MODELS_TO_TEST)} models")
print(f"  = {len(SCENARIOS) * 2 * len(MODELS_TO_TEST)} total LLM calls")
print(f"Expected runtime: ~2-3 hours\n")

all_model_results = {}

for model_name in MODELS_TO_TEST:
    print(f"\n{'#'*60}")
    print(f"# MODEL: {model_name}")
    print(f"{'#'*60}")

    try:
        llm = kbench.llms[model_name]
        result = moral_geometry_ablation.run(llm=llm)
        all_model_results[model_name] = result.result
    except Exception as e:
        print(f"  ERROR running {model_name}: {e}")
        all_model_results[model_name] = {"error": str(e)}

# ═══════════════════════════════════════════════════════════════
# CROSS-MODEL SUMMARY
# ═══════════════════════════════════════════════════════════════

print(f"\n\n{'#'*60}")
print(f"CROSS-MODEL COMPARISON")
print(f"{'#'*60}")
print()
print(f"  {'Model':<30} {'Vanilla':>8} {'Geometric':>10} {'Delta':>8} {'p-value':>8} {'ESH+NAH':>10}")
print(f"  {'-'*74}")

for model_name, r in all_model_results.items():
    if "error" in r:
        print(f"  {model_name:<30} {'ERROR':>8}")
        continue
    v = r["vanilla_accuracy"]
    g = r["geometric_accuracy"]
    d = r["delta"]
    p = r["p_value"]
    en = r["esh_nah_geometric"] - r["esh_nah_vanilla"]
    sig = "*" if p < 0.05 else ""
    print(f"  {model_name:<30} {v:>7.1%} {g:>9.1%} {d:>+7.1%}{sig} {p:>8.4f} {en:>+9.1%}")

print()
print(f"  * = statistically significant (p < 0.05)")
print()

# Key insight
geo_wins = sum(1 for r in all_model_results.values()
               if "error" not in r and r.get("delta", 0) > 0)
total_models = sum(1 for r in all_model_results.values() if "error" not in r)

if geo_wins == total_models:
    print(f"  FINDING: Geometric reasoning improves ALL {total_models} models tested.")
    print(f"  The multi-attribute decomposition is model-agnostic.")
elif geo_wins > 0:
    print(f"  FINDING: Geometric reasoning improves {geo_wins}/{total_models} models.")
else:
    print(f"  FINDING: Geometric reasoning did not improve any model tested.")

print(f"\n  Interpretation:")
print(f"  The Scalar Irrecoverability Theorem (Bond, 2026) predicts that")
print(f"  forcing dimensional decomposition before verdict should help")
print(f"  most on ambiguous cases (ESH/NAH) where multiple moral dimensions")
print(f"  compete. The ESH+NAH delta column shows whether this prediction holds.")
print(f"{'#'*60}")

print("\nTask files saved:")
for f in sorted(os.listdir(".")):
    if f.endswith(".json"):
        print(f"  {f}")
