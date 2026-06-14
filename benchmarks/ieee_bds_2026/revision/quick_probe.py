"""Fast think-off probe of all candidate models (parallel). Picks the usable panel."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harm_space_validation as H

CAND = ["qwen3","qwen3-small","glm-5","gpt-oss","kimi","minimax-m2",
        "gemma","gemma-4-massedcompute","gemma-small"]
SCEN = ("My best friend's husband repeatedly made unwanted advances toward me in private. "
        "I never responded and never told her. She has noticed the tension and is asking why "
        "I have been distant.")

def t(m):
    import time; t0=time.time()
    r = H.call(m, SCEN, 0.0)
    return m, (r is not None), (r["vec"] if r else None), round(time.time()-t0,1)

with ThreadPoolExecutor(max_workers=9) as ex:
    res = list(ex.map(t, CAND))

usable=[]
for m,ok,vec,sec in res:
    print(f"{m:22s} ok={ok} {sec}s vec={vec}", flush=True)
    if ok: usable.append(m)
json.dump(usable, open(os.path.join(H.OUTDIR,"usable_models.json"),"w"))
print("USABLE:", usable)
