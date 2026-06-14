"""Smoke-test DEME v3 API + LLM extraction. Writes result to file."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deme_verdict_analysis as D
from erisml.ethics.facts import EthicalFacts, Consequences, RightsAndDuties, JusticeAndFairness

out = {}
# 1. pure-API check: rights violation -> forbid; benign -> prefer/strongly_prefer
bad = EthicalFacts(option_id="b", consequences=Consequences(expected_harm=0.8),
                   rights_and_duties=RightsAndDuties(violates_rights=True, has_valid_consent=False),
                   justice_and_fairness=JusticeAndFairness())
good = EthicalFacts(option_id="g", consequences=Consequences(expected_harm=0.05),
                    rights_and_duties=RightsAndDuties(violates_rights=False, has_valid_consent=True),
                    justice_and_fairness=JusticeAndFairness())
out["api_bad_verdict"] = D.EM.judge(bad).verdict
out["api_good_verdict"] = D.EM.judge(good).verdict

# 2. extraction + verdict on a gold scenario neutral vs euphemistic, one model
scen = json.load(open(os.path.join(os.path.dirname(__file__),"scenarios.json"),encoding="utf-8"))
gold = [s for s in scen if s["source"]=="GOLD_SET"][2]  # the coming-out / identity one
for cond in ("neutral","euphemistic","dramatic"):
    text = gold["text"] if cond=="neutral" else gold.get(cond)
    out[f"gold_{cond}"] = D.verdict_for("qwen3", text) if text else None

os.makedirs(D.OUT, exist_ok=True)
json.dump(out, open(os.path.join(D.OUT,"deme_smoke.json"),"w"), indent=1)
print("DEME SMOKE DONE")
print(json.dumps(out, indent=1))
