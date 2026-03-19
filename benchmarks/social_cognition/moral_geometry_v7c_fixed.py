"""Moral Geometry Benchmark v7 — Five Geometric Tests of Social Cognition
Social Cognition Track | Measuring AGI Competition

Tests 5 geometric properties of moral cognition (Bond, 2026):
  T1. Structural Fuzzing — sensitivity profile of 7 moral dimensions
  T2. Bond Invariance Principle — verdict stability under re-description
  T3. Holonomy — path-dependence of moral judgment
  T4. Contraction Order — non-commutativity of dimensional evaluation
  T5. Conservation of Harm — harm invariance under linguistic reframing

Paste this ENTIRE file into ONE cell in a Kaggle Benchmark Task notebook.
Expected runtime: ~45-90 min (threaded, 3 models × ~470 calls each).
"""

import kaggle_benchmarks as kbench
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, json, time, random, math, threading

os.environ["RENDER_SUBRUNS"] = "False"

WORKERS = 7  # parallel LLM calls — tune if rate-limited

# Shared results store (kbench tasks must return None, not dict)
_results_store = {}

print("=" * 70)
print("MORAL GEOMETRY BENCHMARK v7 (threaded)")
print("Five Geometric Tests of Social Cognition")
print("Based on Bond (2026), Geometric Ethics")
print(f"Parallelism: {WORKERS} workers")
print("=" * 70)
print()

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD DATASET
# ═══════════════════════════════════════════════════════════════════════

print("[1/8] Loading AITA dataset...")
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
PER_CLASS = 10
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
print(f"  Total: {len(SCENARIOS)} scenarios\n")

DIMENSIONS = [
    "consequences", "rights", "fairness", "autonomy",
    "trust", "social_impact", "identity"
]

DIM_DESCRIPTIONS = {
    "consequences": "Who was harmed? Who benefited? How severely?",
    "rights": "Whose rights were respected or violated?",
    "fairness": "Was this fair to all parties? Was there reciprocity?",
    "autonomy": "Was anyone's freedom of choice restricted?",
    "trust": "Was trust maintained or broken?",
    "social_impact": "How did this affect relationships and community?",
    "identity": "What does this say about the person's character?",
}


# ═══════════════════════════════════════════════════════════════════════
# 2. STRUCTURED OUTPUT SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DimensionalVerdict:
    verdict: str
    confidence: float
    reasoning: str

@dataclass
class FullDimensionalAnalysis:
    consequences_score: float
    rights_score: float
    fairness_score: float
    autonomy_score: float
    trust_score: float
    social_impact_score: float
    identity_score: float
    verdict: str
    confidence: float
    reasoning: str

@dataclass
class InvarianceJudgment:
    verdict: str
    confidence: float
    severity: float
    reasoning: str

@dataclass
class OrderedJudgment:
    verdict: str
    confidence: float
    reasoning: str

@dataclass
class HarmAssessment:
    consequences_harm: float
    rights_harm: float
    fairness_harm: float
    autonomy_harm: float
    trust_harm: float
    social_impact_harm: float
    identity_harm: float
    total_harm: float
    verdict: str
    reasoning: str


# ═══════════════════════════════════════════════════════════════════════
# 3. HELPERS
# ═══════════════════════════════════════════════════════════════════════

def normalize_verdict(raw):
    raw = str(raw).strip().upper().replace("'", "").replace(".", "")
    for label in ["NTA", "YTA", "ESH", "NAH"]:
        if label in raw:
            return label
    return "UNK"

def mean(xs):
    xs = list(xs)
    return sum(xs) / max(len(xs), 1)

def stdev(xs):
    xs = list(xs)
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5

def pearson_r(xs, ys):
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    mx, my = mean(xs[:n]), mean(ys[:n])
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = sum((xs[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ys[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx < 1e-10 or dy < 1e-10:
        return 0.0
    return num / (dx * dy)

def gini(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0 or sum(xs) < 1e-10:
        return 0.0
    cum = sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs))
    return cum / (n * sum(xs))

def clamp(v, lo, hi):
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = (lo + hi) / 2
    return max(lo, min(hi, v))


def call_llm(llm, chat_id, prompt_text, schema=None):
    """Thread-safe LLM call wrapper."""
    with kbench.chats.new(chat_id):
        return llm.prompt(prompt_text, schema=schema) if schema else llm.prompt(prompt_text)


def call_llm_2step(llm, gen_id, judge_id, gen_prompt, judge_prompt_fn, schema):
    """Thread-safe 2-step: generate text, then judge it."""
    with kbench.chats.new(gen_id):
        generated = llm.prompt(gen_prompt)
    with kbench.chats.new(judge_id):
        return llm.prompt(judge_prompt_fn(generated), schema=schema)


# ═══════════════════════════════════════════════════════════════════════
# 4. PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def base_scenario_text(s):
    return f"Title: {s['title']}\n\n{s['text']}"

VERDICT_OPTIONS = (
    "YTA = You're The Asshole (poster is wrong)\n"
    "NTA = Not The Asshole (poster is right)\n"
    "ESH = Everyone Sucks Here (both sides are wrong)\n"
    "NAH = No Assholes Here (nobody is wrong)"
)

def prompt_subset_dims(s, dims):
    dim_block = "\n".join(f"  - {d.replace('_', ' ').title()}: {DIM_DESCRIPTIONS[d]}" for d in dims)
    return (
        f"You are judging a post from Reddit's r/AmITheAsshole.\n"
        f"Evaluate ONLY using these moral dimensions (ignore all others):\n"
        f"{dim_block}\n\n"
        f"{base_scenario_text(s)}\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
        f"Rate confidence 0-10."
    )

def prompt_full_dims(s):
    dim_block = "\n".join(f"  - {d.replace('_', ' ').title()}: {DIM_DESCRIPTIONS[d]}" for d in DIMENSIONS)
    return (
        f"You are judging a post from Reddit's r/AmITheAsshole.\n"
        f"Score each moral dimension from -5 (severe harm) to +5 (clear benefit):\n"
        f"{dim_block}\n\n"
        f"{base_scenario_text(s)}\n\n"
        f"Give scores for each dimension, then your verdict:\n{VERDICT_OPTIONS}\n"
        f"Rate confidence 0-10."
    )

def prompt_invariance(s):
    return (
        f"You are judging a post from Reddit's r/AmITheAsshole.\n\n"
        f"{base_scenario_text(s)}\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
        f"Rate severity 0-10 and confidence 0-10."
    )

def prompt_gender_swap(s):
    return (
        f"Rewrite the following scenario, swapping all genders "
        f"(male→female, female→male, he→she, etc.). "
        f"Keep ALL facts, actions, and moral content identical. "
        f"Only change gender references.\n\n"
        f"{base_scenario_text(s)}"
    )

def prompt_reframe_neutral(s):
    return (
        f"Rewrite the following scenario changing all names and "
        f"shifting the cultural context (e.g., family dinner→work event). "
        f"Keep ALL moral facts and actions identical. "
        f"Only change surface details.\n\n"
        f"{base_scenario_text(s)}"
    )

def prompt_reorder_victim_first(s):
    return (
        f"Rewrite the following scenario so it is told from the "
        f"affected party's (victim's) perspective first, then the poster's actions, "
        f"then the broader context. Keep ALL facts identical.\n\n"
        f"{base_scenario_text(s)}"
    )

def prompt_reorder_context_first(s):
    return (
        f"Rewrite the following scenario starting with the broader context "
        f"and background, then the poster's actions, then the impact on others. "
        f"Keep ALL facts identical.\n\n"
        f"{base_scenario_text(s)}"
    )

def prompt_ordered_dims(s, dim_a, dim_b):
    desc_a = DIM_DESCRIPTIONS[dim_a]
    desc_b = DIM_DESCRIPTIONS[dim_b]
    name_a = dim_a.replace("_", " ").title()
    name_b = dim_b.replace("_", " ").title()
    return (
        f"You are judging a post from Reddit's r/AmITheAsshole.\n"
        f"Evaluate in exactly two steps:\n\n"
        f"STEP 1: Consider ONLY {name_a}: {desc_a}\n"
        f"Form your initial assessment.\n\n"
        f"STEP 2: Now, incorporating your {name_a} assessment, "
        f"evaluate {name_b}: {desc_b}\n"
        f"Give your FINAL verdict based on both steps.\n\n"
        f"{base_scenario_text(s)}\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
        f"Rate confidence 0-10."
    )

def prompt_euphemistic(s):
    return (
        f"Rewrite the following scenario using softer, more neutral, "
        f"euphemistic language. Minimize emotional impact. "
        f"Keep ALL facts and actions identical.\n\n"
        f"{base_scenario_text(s)}"
    )

def prompt_dramatic(s):
    return (
        f"Rewrite the following scenario in emotionally charged, "
        f"dramatic language. Maximize emotional impact. "
        f"Keep ALL facts and actions identical.\n\n"
        f"{base_scenario_text(s)}"
    )

def prompt_harm(text):
    return (
        f"You are assessing harm in a post from Reddit's r/AmITheAsshole.\n"
        f"Rate harm on each dimension from 0 (no harm) to 10 (severe harm):\n"
        f"  - Consequences, Rights, Fairness, Autonomy, Trust, Social Impact, Identity\n\n"
        f"{text}\n\n"
        f"Also give total harm (0-70) and your verdict:\n{VERDICT_OPTIONS}"
    )

def judge_prompt(text):
    return (
        f"You are judging a post from Reddit's r/AmITheAsshole.\n\n"
        f"{text}\n\n"
        f"Verdict:\n{VERDICT_OPTIONS}\nRate severity 0-10 and confidence 0-10."
    )


# ═══════════════════════════════════════════════════════════════════════
# T1: STRUCTURAL FUZZING OF MORAL DIMENSIONS
# ═══════════════════════════════════════════════════════════════════════

@kbench.task(name="t1_structural_fuzzing")
def t1_structural_fuzzing(llm):
    print("\n[T1] STRUCTURAL FUZZING OF MORAL DIMENSIONS")
    print("  Ablating dimensions to find sensitivity profile")
    print("-" * 60)

    scenarios = SCENARIOS[:8]
    solo_accuracy = {d: {"correct": 0, "total": 0} for d in DIMENSIONS}
    loo_flips = {d: 0 for d in DIMENSIONS}
    loo_total = {d: 0 for d in DIMENSIONS}
    baseline_correct = 0
    baseline_total = 0
    _lock = threading.Lock()

    for si, s in enumerate(scenarios):
        actual = s["verdict"]

        # Baseline first (needed for LOO comparison)
        base = call_llm(llm, f"t1_base_{si}", prompt_full_dims(s), FullDimensionalAnalysis)
        base_pred = normalize_verdict(base.verdict)
        if base_pred == actual:
            baseline_correct += 1
        baseline_total += 1

        # Solo + LOO in parallel (14 calls)
        futures = {}
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for d in DIMENSIONS:
                f = pool.submit(call_llm, llm, f"t1_solo_{si}_{d}",
                                prompt_subset_dims(s, [d]), DimensionalVerdict)
                futures[f] = ("solo", d)

                remaining = [x for x in DIMENSIONS if x != d]
                f2 = pool.submit(call_llm, llm, f"t1_loo_{si}_{d}",
                                 prompt_subset_dims(s, remaining), DimensionalVerdict)
                futures[f2] = ("loo", d)

            for f in as_completed(futures):
                kind, d = futures[f]
                try:
                    j = f.result()
                    pred = normalize_verdict(j.verdict)
                    with _lock:
                        if kind == "solo":
                            solo_accuracy[d]["total"] += 1
                            if pred == actual:
                                solo_accuracy[d]["correct"] += 1
                        else:
                            loo_total[d] += 1
                            if pred != base_pred:
                                loo_flips[d] += 1
                except Exception as e:
                    print(f"    WARN: {kind} {d} failed: {e}")

        n = si + 1
        print(f"  [{n}/{len(scenarios)}] baseline={'correct' if base_pred == actual else 'MISS'}")

    solo_acc = {d: solo_accuracy[d]["correct"] / max(solo_accuracy[d]["total"], 1) for d in DIMENSIONS}
    loo_flip_rate = {d: loo_flips[d] / max(loo_total[d], 1) for d in DIMENSIONS}
    ranked = sorted(DIMENSIONS, key=lambda d: loo_flip_rate[d], reverse=True)
    profile_sharpness = gini([loo_flip_rate[d] for d in DIMENSIONS])

    print(f"\n  RESULTS:")
    print(f"  Baseline accuracy: {baseline_correct}/{baseline_total}")
    print(f"  {'Dimension':<20} {'Solo Acc':>10} {'LOO Flip Rate':>15} {'Rank':>6}")
    print(f"  {'-'*51}")
    for rank, d in enumerate(ranked, 1):
        name = d.replace("_", " ").title()
        print(f"  {name:<20} {solo_acc[d]:>9.0%} {loo_flip_rate[d]:>14.0%} {rank:>6}")
    print(f"  Profile sharpness (Gini): {profile_sharpness:.3f}")

    _results_store["T1_fuzzing"] = {
        "solo_accuracy": solo_acc,
        "loo_flip_rate": loo_flip_rate,
        "importance_ranking": ranked,
        "profile_sharpness": profile_sharpness,
        "baseline_accuracy": baseline_correct / max(baseline_total, 1),
        "score": profile_sharpness,
    }


# ═══════════════════════════════════════════════════════════════════════
# T2: BOND INVARIANCE PRINCIPLE
# ═══════════════════════════════════════════════════════════════════════

@kbench.task(name="t2_invariance")
def t2_invariance(llm):
    print("\n[T2] BOND INVARIANCE PRINCIPLE")
    print("  Testing verdict stability under re-description")
    print("-" * 60)

    scenarios = SCENARIOS[:20]
    gender_flips = 0
    reframe_flips = 0
    severity_drifts = []
    total = 0

    for si, s in enumerate(scenarios):
        # Baseline
        base = call_llm(llm, f"t2_base_{si}", prompt_invariance(s), InvarianceJudgment)
        base_v = normalize_verdict(base.verdict)
        base_sev = clamp(base.severity, 0, 10)

        # Gender swap + reframe in parallel (2-step each)
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_gender = pool.submit(
                call_llm_2step, llm,
                f"t2_gswap_{si}", f"t2_gjudge_{si}",
                prompt_gender_swap(s), judge_prompt, InvarianceJudgment
            )
            f_reframe = pool.submit(
                call_llm_2step, llm,
                f"t2_rswap_{si}", f"t2_rjudge_{si}",
                prompt_reframe_neutral(s), judge_prompt, InvarianceJudgment
            )

            gj = f_gender.result()
            rj = f_reframe.result()

        g_v = normalize_verdict(gj.verdict)
        g_sev = clamp(gj.severity, 0, 10)
        r_v = normalize_verdict(rj.verdict)
        r_sev = clamp(rj.severity, 0, 10)

        total += 1
        if g_v != base_v:
            gender_flips += 1
        if r_v != base_v:
            reframe_flips += 1
        severity_drifts.append(abs(g_sev - base_sev))
        severity_drifts.append(abs(r_sev - base_sev))

        n = si + 1
        flipped = "GENDER" if g_v != base_v else ""
        flipped += " REFRAME" if r_v != base_v else ""
        if n % 5 == 0 or flipped:
            print(f"  [{n}/{len(scenarios)}] base={base_v} gender={g_v} reframe={r_v} {flipped}")

    gender_rate = gender_flips / max(total, 1)
    reframe_rate = reframe_flips / max(total, 1)
    overall_violation = (gender_flips + reframe_flips) / max(2 * total, 1)
    avg_sev_drift = mean(severity_drifts)

    print(f"\n  RESULTS:")
    print(f"  Gender-swap violation rate: {gender_flips}/{total} ({gender_rate:.0%})")
    print(f"  Reframe violation rate: {reframe_flips}/{total} ({reframe_rate:.0%})")
    print(f"  Overall invariance violation: {overall_violation:.0%}")
    print(f"  Mean severity drift: {avg_sev_drift:.2f}")

    _results_store["T2_invariance"] = {
        "gender_violation_rate": gender_rate,
        "reframe_violation_rate": reframe_rate,
        "overall_violation_rate": overall_violation,
        "severity_drift": avg_sev_drift,
        "score": 1.0 - overall_violation,
    }


# ═══════════════════════════════════════════════════════════════════════
# T3: HOLONOMY / PATH-DEPENDENCE
# ═══════════════════════════════════════════════════════════════════════

@kbench.task(name="t3_holonomy")
def t3_holonomy(llm):
    print("\n[T3] HOLONOMY — PATH-DEPENDENCE OF MORAL JUDGMENT")
    print("  Same facts, different presentation order")
    print("-" * 60)

    scenarios = SCENARIOS[8:23]
    path_dependent = 0
    total = 0
    order_verdicts = []

    for si, s in enumerate(scenarios):
        # Original + two reorders in parallel
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_orig = pool.submit(call_llm, llm, f"t3_orig_{si}",
                                 prompt_invariance(s), InvarianceJudgment)
            f_victim = pool.submit(
                call_llm_2step, llm,
                f"t3_vgen_{si}", f"t3_vjudge_{si}",
                prompt_reorder_victim_first(s), judge_prompt, InvarianceJudgment
            )
            f_context = pool.submit(
                call_llm_2step, llm,
                f"t3_cgen_{si}", f"t3_cjudge_{si}",
                prompt_reorder_context_first(s), judge_prompt, InvarianceJudgment
            )

            orig = f_orig.result()
            vj = f_victim.result()
            cj = f_context.result()

        v_orig = normalize_verdict(orig.verdict)
        v_victim = normalize_verdict(vj.verdict)
        v_context = normalize_verdict(cj.verdict)

        total += 1
        verdicts = {v_orig, v_victim, v_context}
        if len(verdicts) > 1:
            path_dependent += 1
        order_verdicts.append((v_orig, v_victim, v_context))

        n = si + 1
        marker = " PATH-DEP" if len(verdicts) > 1 else ""
        if n % 5 == 0 or marker:
            print(f"  [{n}/{len(scenarios)}] orig={v_orig} victim={v_victim} "
                  f"context={v_context}{marker}")

    holonomy_rate = path_dependent / max(total, 1)
    all_agree = sum(1 for o, v, c in order_verdicts if o == v == c)
    two_agree = sum(1 for o, v, c in order_verdicts if len({o, v, c}) == 2)
    none_agree = sum(1 for o, v, c in order_verdicts if len({o, v, c}) == 3)

    print(f"\n  RESULTS:")
    print(f"  Path-dependent scenarios: {path_dependent}/{total} ({holonomy_rate:.0%})")
    print(f"  All 3 agree: {all_agree}/{total}")
    print(f"  2 of 3 agree: {two_agree}/{total}")
    print(f"  All different: {none_agree}/{total}")

    _results_store["T3_holonomy"] = {
        "holonomy_rate": holonomy_rate,
        "all_agree": all_agree / max(total, 1),
        "two_agree": two_agree / max(total, 1),
        "none_agree": none_agree / max(total, 1),
        "score": 1.0 - holonomy_rate,
    }


# ═══════════════════════════════════════════════════════════════════════
# T4: CONTRACTION ORDER — NON-COMMUTATIVITY
# ═══════════════════════════════════════════════════════════════════════

@kbench.task(name="t4_contraction_order")
def t4_contraction_order(llm):
    print("\n[T4] CONTRACTION ORDER — NON-COMMUTATIVITY")
    print("  Does evaluation order change the verdict?")
    print("-" * 60)

    scenarios = SCENARIOS[10:25]
    DIM_PAIRS = [
        ("consequences", "fairness"),
        ("rights", "trust"),
        ("autonomy", "identity"),
    ]

    pair_flips = {f"{a},{b}": 0 for a, b in DIM_PAIRS}
    pair_total = {f"{a},{b}": 0 for a, b in DIM_PAIRS}
    total_flips = 0
    total_tests = 0
    _lock = threading.Lock()

    for si, s in enumerate(scenarios):
        # All 6 calls (3 pairs × 2 orders) in parallel
        futures = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            for dim_a, dim_b in DIM_PAIRS:
                f_ab = pool.submit(call_llm, llm, f"t4_{si}_{dim_a}_{dim_b}",
                                   prompt_ordered_dims(s, dim_a, dim_b), OrderedJudgment)
                f_ba = pool.submit(call_llm, llm, f"t4_{si}_{dim_b}_{dim_a}",
                                   prompt_ordered_dims(s, dim_b, dim_a), OrderedJudgment)
                futures[(dim_a, dim_b)] = (f_ab, f_ba)

            for (dim_a, dim_b), (f_ab, f_ba) in futures.items():
                pair_key = f"{dim_a},{dim_b}"
                try:
                    v_ab = normalize_verdict(f_ab.result().verdict)
                    v_ba = normalize_verdict(f_ba.result().verdict)
                    with _lock:
                        pair_total[pair_key] += 1
                        total_tests += 1
                        if v_ab != v_ba:
                            pair_flips[pair_key] += 1
                            total_flips += 1
                except Exception as e:
                    print(f"    WARN: {pair_key} failed: {e}")

        n = si + 1
        if n % 5 == 0:
            running = total_flips / max(total_tests, 1)
            print(f"  [{n}/{len(scenarios)}] non-commutativity rate: {running:.0%}")

    noncomm_rate = total_flips / max(total_tests, 1)

    print(f"\n  RESULTS:")
    print(f"  Overall non-commutativity: {total_flips}/{total_tests} ({noncomm_rate:.0%})")
    print(f"  {'Pair':<35} {'Flips':>8} {'Rate':>8}")
    print(f"  {'-'*51}")
    pair_rates = {}
    for a, b in DIM_PAIRS:
        key = f"{a},{b}"
        na = a.replace("_", " ").title()
        nb = b.replace("_", " ").title()
        rate = pair_flips[key] / max(pair_total[key], 1)
        pair_rates[key] = rate
        print(f"  {na} <-> {nb:<20} {pair_flips[key]:>8} {rate:>7.0%}")

    _results_store["T4_contraction"] = {
        "noncommutativity_rate": noncomm_rate,
        "pair_rates": pair_rates,
        "score": 1.0 - noncomm_rate,
    }


# ═══════════════════════════════════════════════════════════════════════
# T5: CONSERVATION OF HARM
# ═══════════════════════════════════════════════════════════════════════

@kbench.task(name="t5_conservation")
def t5_conservation(llm):
    print("\n[T5] CONSERVATION OF HARM")
    print("  Does reframing change perceived total harm?")
    print("-" * 60)

    scenarios = SCENARIOS[20:32]
    base_harms = []
    euphemistic_harms = []
    dramatic_harms = []
    verdict_flips_euph = 0
    verdict_flips_dram = 0
    total = 0

    for si, s in enumerate(scenarios):
        # Baseline + euphemistic + dramatic in parallel
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_base = pool.submit(call_llm, llm, f"t5_base_{si}",
                                 prompt_harm(base_scenario_text(s)), HarmAssessment)
            f_euph = pool.submit(
                call_llm_2step, llm,
                f"t5_euph_gen_{si}", f"t5_euph_judge_{si}",
                prompt_euphemistic(s), prompt_harm, HarmAssessment
            )
            f_dram = pool.submit(
                call_llm_2step, llm,
                f"t5_dram_gen_{si}", f"t5_dram_judge_{si}",
                prompt_dramatic(s), prompt_harm, HarmAssessment
            )

            base = f_base.result()
            euph = f_euph.result()
            dram = f_dram.result()

        base_total = clamp(base.total_harm, 0, 70)
        base_v = normalize_verdict(base.verdict)
        base_harms.append(base_total)

        euph_total = clamp(euph.total_harm, 0, 70)
        euph_v = normalize_verdict(euph.verdict)
        euphemistic_harms.append(euph_total)

        dram_total = clamp(dram.total_harm, 0, 70)
        dram_v = normalize_verdict(dram.verdict)
        dramatic_harms.append(dram_total)

        total += 1
        if euph_v != base_v:
            verdict_flips_euph += 1
        if dram_v != base_v:
            verdict_flips_dram += 1

        n = si + 1
        drift_e = euph_total - base_total
        drift_d = dram_total - base_total
        if n % 4 == 0:
            print(f"  [{n}/{len(scenarios)}] base={base_total:.0f} euph={euph_total:.0f}({drift_e:+.0f}) "
                  f"dram={dram_total:.0f}({drift_d:+.0f})")

    r_euph = pearson_r(base_harms, euphemistic_harms)
    r_dram = pearson_r(base_harms, dramatic_harms)
    mean_drift_euph = mean([e - b for b, e in zip(base_harms, euphemistic_harms)])
    mean_drift_dram = mean([d - b for b, d in zip(base_harms, dramatic_harms)])
    conservation = (r_euph + r_dram) / 2

    print(f"\n  RESULTS:")
    print(f"  Base-Euphemistic correlation: r={r_euph:.3f}, mean drift={mean_drift_euph:+.1f}")
    print(f"  Base-Dramatic correlation: r={r_dram:.3f}, mean drift={mean_drift_dram:+.1f}")
    print(f"  Conservation score: {conservation:.3f}")
    print(f"  Verdict flips (euphemistic): {verdict_flips_euph}/{total}")
    print(f"  Verdict flips (dramatic): {verdict_flips_dram}/{total}")

    _results_store["T5_conservation"] = {
        "r_euphemistic": r_euph,
        "r_dramatic": r_dram,
        "mean_drift_euphemistic": mean_drift_euph,
        "mean_drift_dramatic": mean_drift_dram,
        "conservation": conservation,
        "verdict_flips_euphemistic": verdict_flips_euph / max(total, 1),
        "verdict_flips_dramatic": verdict_flips_dram / max(total, 1),
        "score": max(0, conservation),
    }


# ═══════════════════════════════════════════════════════════════════════
# MULTI-MODEL EXECUTION
# ═══════════════════════════════════════════════════════════════════════

MODELS_TO_TEST = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "meta/llama-3.1-70b",
]

print(f"\n[2/8] Running 5 geometric tests across {len(MODELS_TO_TEST)} models")
for m in MODELS_TO_TEST:
    print(f"  - {m}")
print()

all_results = {}

for model_name in MODELS_TO_TEST:
    print(f"\n{'#'*70}")
    print(f"# MODEL: {model_name}")
    print(f"{'#'*70}")

    model_results = {}
    try:
        llm = kbench.llms[model_name]
        _results_store.clear()  # reset for each model

        for test_fn, test_name in [
            (t1_structural_fuzzing, "T1_fuzzing"),
            (t2_invariance, "T2_invariance"),
            (t3_holonomy, "T3_holonomy"),
            (t4_contraction_order, "T4_contraction"),
            (t5_conservation, "T5_conservation"),
        ]:
            try:
                test_fn.run(llm=llm)
                model_results[test_name] = _results_store.get(test_name, {"score": 0.0})
            except Exception as e:
                print(f"  ERROR in {test_name}: {e}")
                model_results[test_name] = {"error": str(e), "score": 0.0}

    except Exception as e:
        print(f"  ERROR loading model {model_name}: {e}")
        model_results = {f"T{i}": {"error": str(e), "score": 0.0} for i in range(1, 6)}

    all_results[model_name] = model_results


# ═══════════════════════════════════════════════════════════════════════
# CROSS-MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*70}")
print(f"CROSS-MODEL COMPARISON — FIVE GEOMETRIC TESTS")
print(f"{'#'*70}")
print()

WEIGHTS = {
    "T1_fuzzing": 0.25,
    "T2_invariance": 0.20,
    "T3_holonomy": 0.20,
    "T4_contraction": 0.15,
    "T5_conservation": 0.20,
}

header = f"  {'Model':<30} {'T1:Fuzz':>8} {'T2:BIP':>8} {'T3:Holo':>8} {'T4:Comm':>8} {'T5:Cons':>8} {'Compos':>8}"
print(header)
print(f"  {'─'*78}")

for model_name, results in all_results.items():
    scores = {}
    for test_key in WEIGHTS:
        r = results.get(test_key, {})
        scores[test_key] = r.get("score", 0.0)

    composite = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)

    short_name = model_name.split("/")[-1][:28]
    print(f"  {short_name:<30} "
          f"{scores['T1_fuzzing']:>7.3f} "
          f"{scores['T2_invariance']:>7.3f} "
          f"{scores['T3_holonomy']:>7.3f} "
          f"{scores['T4_contraction']:>7.3f} "
          f"{scores['T5_conservation']:>7.3f} "
          f"{composite:>7.3f}")

print()
print(f"  Weights: T1={WEIGHTS['T1_fuzzing']}, T2={WEIGHTS['T2_invariance']}, "
      f"T3={WEIGHTS['T3_holonomy']}, T4={WEIGHTS['T4_contraction']}, T5={WEIGHTS['T5_conservation']}")
print()

print("INTERPRETATION")
print("=" * 70)
print()
print("  T1 (Structural Fuzzing): Higher = sharper sensitivity profile.")
print("  T2 (Bond Invariance): Higher = more invariant under re-description.")
print("  T3 (Holonomy): Higher = less path-dependent.")
print("  T4 (Contraction Order): Higher = more commutative.")
print("  T5 (Conservation of Harm): Higher = better harm conservation.")
print()
print("  These 5 tests operationalize predictions from Geometric Ethics")
print("  (Bond, 2026) as quantitative benchmarks for social cognition.")
print("=" * 70)

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
