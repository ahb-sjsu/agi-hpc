"""Quick end-to-end smoke test of the harm-judge call path. Writes JSON to file."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harm_space_validation as H

scen = json.load(open(os.path.join(os.path.dirname(__file__),"scenarios.json"),encoding="utf-8"))
text = scen[5]["text"]   # a gold scenario
out = {"scenario": text[:120]}
for m in H.MODELS:
    r = H.call(m, text, 0.0)
    out[m] = r
os.makedirs(H.OUTDIR, exist_ok=True)
json.dump(out, open(os.path.join(H.OUTDIR,"smoke.json"),"w"), indent=1)
print("SMOKE DONE ->", os.path.join(H.OUTDIR,"smoke.json"))
