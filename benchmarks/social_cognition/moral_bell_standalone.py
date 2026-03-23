"""Moral Bell Test — Standalone version using Gemini API directly.
Runs on HPC or any machine. No Kaggle SDK needed.

Usage:
    python moral_bell_standalone.py --mini     # 5 scenarios per test (~2 min)
    python moral_bell_standalone.py            # full 200 scenarios (~30 min)
"""

import os, sys, json, time, random, math, re, logging
from dataclasses import dataclass

MINI = "--mini" in sys.argv
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "REDACTED")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bell")

# ═══════════════════════════════════════════════════════════════
# GEMINI CLIENT
# ═══════════════════════════════════════════════════════════════

try:
    import google.generativeai as genai
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "google-generativeai"])
    import google.generativeai as genai

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def ask_gemini(prompt, retries=3):
    """Call Gemini with retry logic for rate limits."""
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 10 * (attempt + 1)
                log.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                log.error(f"Gemini error: {e}")
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    return f"ERROR: {e}"
    return "ERROR: max retries"

def extract_verdict(text):
    """Extract YTA/NTA/ESH/NAH from LLM response."""
    text_upper = text.upper()
    for label in ["NTA", "YTA", "ESH", "NAH"]:
        if label in text_upper:
            return label
    return "UNK"

def extract_severity(text):
    """Extract severity number 0-10 from response."""
    match = re.search(r'severity[:\s]*(\d+(?:\.\d+)?)', text.lower())
    if match:
        return float(match.group(1))
    match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', text)
    if match:
        return float(match.group(1))
    return 5.0

def verdict_to_spin(verdict):
    return +1 if verdict in ("NTA", "NAH") else -1

print("=" * 60)
print(f"MORAL BELL TEST — Standalone ({'MINI' if MINI else 'FULL'})")
print(f"Using: Gemini 2.0 Flash")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# LOAD AITA DATASET
# ═══════════════════════════════════════════════════════════════

print("\n[1/5] Loading AITA dataset...")
t0 = time.time()

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset

ds = load_dataset("OsamaBsher/AITA-Reddit-Dataset", split="train")
log.info(f"Loaded {len(ds):,} posts in {time.time()-t0:.0f}s")

random.seed(42)
PER_CLASS = 5 if MINI else 50
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
    selected = pool[:PER_CLASS]
    SCENARIOS.extend(selected)
    print(f"  {v}: {len(selected)} scenarios")

random.shuffle(SCENARIOS)
N = len(SCENARIOS)
print(f"  Total: {N} scenarios\n")

# ═══════════════════════════════════════════════════════════════
# TASK 1: CHSH BELL TEST
# ═══════════════════════════════════════════════════════════════

print("[2/5] CHSH BELL TEST")
print(f"  {min(N, 20 if MINI else 50)} scenarios × 5 LLM calls each")
print("-" * 60)

n_bell = min(N, 5 if MINI else 50)
corr = {"A1B1": [], "A1B2": [], "A2B1": [], "A2B2": []}

FRAMING_RIGHTS = "Focus ONLY on rights and obligations. Who had a right to what? Who violated whose rights?"
FRAMING_CONSEQ = "Focus ONLY on consequences and outcomes. Who was harmed? What damage was done?"
FRAMING_FAIR = "Focus ONLY on fairness and reciprocity. Was this fair? Was there equal treatment?"
FRAMING_VIRTUE = "Focus ONLY on character and virtue. What does this say about the person? Is this how a good person acts?"

for i in range(n_bell):
    s = SCENARIOS[i]
    print(f"\n  [{i+1}/{n_bell}] {s['title'][:55]}... (actual: {s['verdict']})")

    # Get other perspective
    swap_text = ask_gemini(
        f"Rewrite this situation from the OTHER person's perspective. "
        f"Keep all facts the same, change who tells the story. Be brief (3-4 sentences).\n\n"
        f"{s['text'][:800]}"
    )

    spins = {}
    for setting, perspective, framing in [
        ("A1B1", s["text"], FRAMING_RIGHTS),
        ("A1B2", s["text"], FRAMING_CONSEQ),
        ("A2B1", swap_text, FRAMING_FAIR),
        ("A2B2", swap_text, FRAMING_VIRTUE),
    ]:
        resp = ask_gemini(
            f"Judge this moral situation from r/AmITheAsshole.\n"
            f"{framing}\n\n"
            f"Situation:\n{perspective[:800]}\n\n"
            f"Give ONLY your verdict: YTA, NTA, ESH, or NAH. Then severity 0-10. One line."
        )
        verdict = extract_verdict(resp)
        spins[setting] = verdict_to_spin(verdict)
        time.sleep(1)  # rate limit

    # Compute correlations
    corr["A1B1"].append(spins["A1B1"] * spins["A2B1"])
    corr["A1B2"].append(spins["A1B1"] * spins["A2B2"])
    corr["A2B1"].append(spins["A1B2"] * spins["A2B1"])
    corr["A2B2"].append(spins["A1B2"] * spins["A2B2"])

    # Running S
    E = {k: sum(v)/max(len(v),1) for k,v in corr.items()}
    S = abs(E["A1B1"] - E["A1B2"] + E["A2B1"] + E["A2B2"])
    print(f"    Spins: A1B1={spins['A1B1']:+d} A1B2={spins['A1B2']:+d} A2B1={spins['A2B1']:+d} A2B2={spins['A2B2']:+d}  |  running S={S:.3f}")

E_final = {k: sum(v)/max(len(v),1) for k,v in corr.items()}
S_final = abs(E_final["A1B1"] - E_final["A1B2"] + E_final["A2B1"] + E_final["A2B2"])

print(f"\n{'='*60}")
print(f"  CHSH RESULT: S = {S_final:.3f}")
print(f"  Classical bound |S| ≤ 2: {'SATISFIED' if S_final <= 2 else 'VIOLATED!'}")
print(f"  Correlations: A1B1={E_final['A1B1']:+.3f} A1B2={E_final['A1B2']:+.3f} A2B1={E_final['A2B1']:+.3f} A2B2={E_final['A2B2']:+.3f}")
print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════
# TASK 2: D₄ CORRELATIVE SYMMETRY
# ═══════════════════════════════════════════════════════════════

print(f"\n[3/5] D₄ CORRELATIVE SYMMETRY")
n_d4 = min(N, 5 if MINI else 80)
print(f"  Testing {n_d4} perspective swaps")
print("-" * 60)

flips_ok = 0
invariants_ok = 0
total_flip = 0
total_inv = 0

for i in range(n_d4):
    s = SCENARIOS[i]
    actual = s["verdict"]

    resp = ask_gemini(
        f"This is from r/AmITheAsshole. The original poster wrote:\n\n"
        f"{s['text'][:800]}\n\n"
        f"Now imagine you are the OTHER person in this conflict. "
        f"From THEIR perspective, is the original poster the asshole?\n"
        f"Verdict: YTA, NTA, ESH, or NAH. One word."
    )
    swapped = extract_verdict(resp)
    time.sleep(1)

    if actual in ("YTA", "NTA"):
        total_flip += 1
        expected = "NTA" if actual == "YTA" else "YTA"
        ok = swapped == expected
        if ok: flips_ok += 1
        symbol = "OK" if ok else f"FAIL ({actual}->{swapped}, expected {expected})"
    else:
        total_inv += 1
        ok = swapped == actual
        if ok: invariants_ok += 1
        symbol = "OK" if ok else f"FAIL ({actual}->{swapped}, should stay {actual})"

    if (i+1) % 10 == 0 or not ok:
        print(f"  [{i+1}/{n_d4}] actual={actual} swapped={swapped} {symbol}")

flip_rate = flips_ok / max(total_flip, 1)
inv_rate = invariants_ok / max(total_inv, 1)
d4_overall = (flips_ok + invariants_ok) / max(total_flip + total_inv, 1)

print(f"\n{'='*60}")
print(f"  D₄ SYMMETRY: {d4_overall:.0%} overall")
print(f"  YTA↔NTA flips: {flip_rate:.0%} ({flips_ok}/{total_flip})")
print(f"  ESH/NAH invariance: {inv_rate:.0%} ({invariants_ok}/{total_inv})")
print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════
# TASK 3: NON-COMMUTATIVITY
# ═══════════════════════════════════════════════════════════════

print(f"\n[4/5] NON-COMMUTATIVITY TEST")
n_nc = min(N, 5 if MINI else 60)
print(f"  Testing {n_nc} scenarios with swapped dimension ordering")
print("-" * 60)

order_changed = 0

for i in range(n_nc):
    s = SCENARIOS[i]

    r1 = ask_gemini(
        f"Judge this moral situation in two steps.\n\n"
        f"{s['text'][:800]}\n\n"
        f"STEP 1: Evaluate FAIRNESS first. Is this fair?\n"
        f"STEP 2: Now evaluate CONSEQUENCES. What harm resulted?\n"
        f"Final verdict: YTA, NTA, ESH, or NAH."
    )
    time.sleep(1)

    r2 = ask_gemini(
        f"Judge this moral situation in two steps.\n\n"
        f"{s['text'][:800]}\n\n"
        f"STEP 1: Evaluate CONSEQUENCES first. What harm resulted?\n"
        f"STEP 2: Now evaluate FAIRNESS. Is this fair?\n"
        f"Final verdict: YTA, NTA, ESH, or NAH."
    )
    time.sleep(1)

    v1 = extract_verdict(r1)
    v2 = extract_verdict(r2)
    if v1 != v2:
        order_changed += 1

    if (i+1) % 10 == 0 or v1 != v2:
        symbol = "COMMUTATIVE" if v1 == v2 else f"NON-COMMUTATIVE ({v1} vs {v2})"
        print(f"  [{i+1}/{n_nc}] {symbol}  |  {s['title'][:50]}")

nc_rate = order_changed / max(n_nc, 1)

print(f"\n{'='*60}")
print(f"  NON-COMMUTATIVITY: {nc_rate:.0%} ({order_changed}/{n_nc})")
if nc_rate > 0.15:
    print(f"  Significant — consistent with non-abelian D₄ structure")
elif nc_rate > 0.05:
    print(f"  Mild — some order dependence detected")
else:
    print(f"  Negligible — model appears order-independent")
print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

elapsed = time.time() - t0

print(f"\n{'#'*60}")
print(f"[5/5] MORAL BELL TEST — FINAL RESULTS")
print(f"{'#'*60}")
print(f"")
print(f"  1. CHSH:               S = {S_final:.3f} ({'CLASSICAL' if S_final <= 2 else 'VIOLATION'})")
print(f"  2. D₄ symmetry:        {d4_overall:.0%}")
print(f"     - YTA↔NTA flips:    {flip_rate:.0%}")
print(f"     - ESH/NAH invariant: {inv_rate:.0%}")
print(f"  3. Non-commutativity:   {nc_rate:.0%}")
print(f"")
if S_final <= 2 and d4_overall >= 0.6 and nc_rate > 0.1:
    print(f"  GAUGE GROUP: D₄ × U(1)_H (classical, non-abelian)")
elif S_final <= 2:
    print(f"  GAUGE GROUP: Classical (|S|≤2), partial D₄ structure")
else:
    print(f"  GAUGE GROUP: Possible quantum contextuality detected")
print(f"")
print(f"  Mode: {'MINI (dev)' if MINI else 'FULL'}")
print(f"  Scenarios: {N}")
print(f"  Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'#'*60}")

# Save results
results = {
    "chsh": {"S": S_final, "classical": S_final <= 2, "correlations": E_final, "n": n_bell},
    "d4": {"overall": d4_overall, "flip_rate": flip_rate, "invariant_rate": inv_rate, "n": n_d4},
    "noncommutativity": {"rate": nc_rate, "changed": order_changed, "n": n_nc},
    "mode": "mini" if MINI else "full",
    "runtime_s": elapsed,
}
with open("moral_bell_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to moral_bell_results.json")
