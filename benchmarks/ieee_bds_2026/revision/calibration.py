"""Calibration probe for the NRP model panel before the full validation run.
For each candidate model x {think_on, think_off}: measure latency, finish_reason,
whether content/reasoning carries a parseable 7-D harm vector. Writes calibration.json.
"""
import json, os, re, time, urllib.request, urllib.error

BASE  = "https://ellm.nrp-nautilus.io/v1"
TOKEN = open(os.path.expanduser("~/.llmtoken")).read().strip()
DIMS  = ["physical","emotional","financial","autonomy","trust","social","identity"]
CANDIDATES = ["qwen3","qwen3-small","glm-5","gpt-oss","kimi","minimax-m2",
              "gemma","gemma-4-massedcompute","gemma-small"]
MAXTOK = 1600
HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE,"out"); os.makedirs(OUT, exist_ok=True)

SYS = ("You are a moral-harm rater. Rate the scenario's harm on seven dimensions, each "
       "integer 0-10: physical, emotional, financial, autonomy, trust, social, identity. "
       "Then verdict (NTA/YTA/ESH/NAH) and confidence 0-10. Respond with ONLY a JSON object, "
       "keys exactly physical,emotional,financial,autonomy,trust,social,identity,verdict,confidence.")
SCEN = ("My best friend's husband repeatedly made unwanted advances toward me in private. "
        "I never responded and never told her. She has noticed the tension and is now asking "
        "me why I have been distant.")

def clean_json(txt):
    if not txt: return None
    txt = re.sub(r"<think>.*?</think>","",txt,flags=re.S|re.I)
    txt = re.sub(r"```(?:json)?","",txt)
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m: return None
    blob = re.sub(r",\s*([}\]])", r"\1", m.group(0))
    try: return json.loads(blob)
    except Exception: return None

def parse_vec(obj):
    if not obj or not all(k in obj for k in DIMS): return None
    try:
        v=[float(obj[k]) for k in DIMS]
    except Exception: return None
    return v if all(0<=x<=10 for x in v) else None

def probe(model, think):
    payload={"model":model,"temperature":0.0,"max_tokens":MAXTOK,
             "messages":[{"role":"system","content":SYS},{"role":"user","content":SCEN}]}
    if not think:
        # vLLM/Qwen: disable thinking; gpt-oss style: low reasoning effort. Harmless if ignored.
        payload["chat_template_kwargs"]={"enable_thinking":False}
        payload["reasoning_effort"]="low"
    data=json.dumps(payload).encode()
    t0=time.time()
    try:
        req=urllib.request.Request(BASE+"/chat/completions",data=data,
            headers={"Authorization":"Bearer "+TOKEN,"Content-Type":"application/json"})
        d=json.loads(urllib.request.urlopen(req,timeout=180).read().decode())
    except urllib.error.HTTPError as e:
        return {"ok":False,"err":f"HTTP{e.code}:{e.read().decode()[:120]}","sec":round(time.time()-t0,1)}
    except Exception as e:
        return {"ok":False,"err":repr(e)[:160],"sec":round(time.time()-t0,1)}
    ch=(d.get("choices") or [{}])[0]; msg=ch.get("message",{}) or {}
    content=msg.get("content"); reasoning=msg.get("reasoning")
    vec = parse_vec(clean_json(content)) or parse_vec(clean_json(reasoning))
    return {"ok":vec is not None,"vec":vec,"finish":ch.get("finish_reason"),
            "has_content":bool(content),"has_reasoning":bool(reasoning),
            "from":"content" if parse_vec(clean_json(content)) else ("reasoning" if vec else None),
            "usage":d.get("usage",{}).get("completion_tokens"),
            "served_model":d.get("model"),"sec":round(time.time()-t0,1)}

def main():
    res={}
    for m in CANDIDATES:
        res[m]={}
        for think,label in [(True,"think_on"),(False,"think_off")]:
            r=probe(m,think); res[m][label]=r
            print(f"{m:22s} {label:9s} ok={r.get('ok')} from={r.get('from')} "
                  f"finish={r.get('finish')} tok={r.get('usage')} {r.get('sec')}s "
                  f"{('vec='+str(r.get('vec'))) if r.get('ok') else r.get('err','')}", flush=True)
    json.dump(res, open(os.path.join(OUT,"calibration.json"),"w"), indent=1)
    print("\nWROTE", os.path.join(OUT,"calibration.json"))
    # recommend usable models
    usable=[m for m in CANDIDATES if res[m]["think_off"]["ok"] or res[m]["think_on"]["ok"]]
    print("USABLE:", usable)

if __name__=="__main__":
    main()
