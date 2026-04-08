#!/usr/bin/env python3
"""Patch atlas-rag-server.py to add Kirk fallback when RH is unavailable.

When GPU 1 is busy (training, embedding), Kirk can't run on a separate
llama-server. This patch makes the single LLM on GPU 0 play both roles
using different system prompts.
"""
import sys

RAG_PATH = "/home/claude/atlas-rag-server.py"

with open(RAG_PATH, "r") as f:
    content = f.read()

# 1. Add helper functions after RH_SYSTEM definition
INSERT_AFTER = 'RH_SYSTEM = ('
insert_marker = content.find('def classify_query')
if insert_marker == -1:
    print("ERROR: could not find classify_query")
    sys.exit(1)

helpers = '''
def _rh_available():
    """Check if Right Hemisphere (Kirk) is reachable."""
    try:
        r = requests.get(f"{RH_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

# Cache RH availability (re-check every 60s)
_rh_status = {"available": False, "checked_at": 0}

def rh_is_up():
    import time as _t
    now = _t.time()
    if now - _rh_status["checked_at"] > 60:
        _rh_status["available"] = _rh_available()
        _rh_status["checked_at"] = now
    return _rh_status["available"]


def call_as_kirk(data):
    """Call LH model with Kirk persona when RH is unavailable."""
    msgs = []
    for m in data.get("messages", []):
        if m.get("role") == "system":
            msgs.append({"role": "system", "content": RH_SYSTEM})
        else:
            msgs.append(m)
    if not any(m.get("role") == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": RH_SYSTEM})
    return call_hemisphere(LH_URL, {**data, "messages": msgs})


'''

# Don't insert if already patched
if 'def rh_is_up' not in content:
    content = content[:insert_marker] + helpers + content[insert_marker:]
    print("Inserted helper functions")
else:
    print("Helpers already present")

# 2. Replace kirk_future calls with fallback versions
replacements = [
    # Round 1 kirk
    (
        '            kirk_future = ex.submit(call_hemisphere, RH_URL, {**data, "messages": kirk_msgs, "max_tokens": 512})',
        '            kirk_future = ex.submit(call_hemisphere if rh_is_up() else call_as_kirk,\n'
        '                                    RH_URL if rh_is_up() else {**data, "messages": kirk_msgs, "max_tokens": 512},\n'
        '                                    {**data, "messages": kirk_msgs, "max_tokens": 512}) if rh_is_up() else ex.submit(call_as_kirk, {**data, "messages": kirk_msgs, "max_tokens": 512})'
    ),
    # Round 2 kirk
    (
        '            kirk_ch_future = ex.submit(call_hemisphere, RH_URL, {**data, "messages": kirk_challenge_msgs, "max_tokens": 384})',
        '            kirk_ch_future = ex.submit(call_hemisphere, RH_URL, {**data, "messages": kirk_challenge_msgs, "max_tokens": 384}) if rh_is_up() else ex.submit(call_as_kirk, {**data, "messages": kirk_challenge_msgs, "max_tokens": 384})'
    ),
    # Captain's call
    (
        '        final = call_hemisphere(RH_URL, {**data, "messages": captain_msgs, "max_tokens": 1024})',
        '        final = call_hemisphere(RH_URL, {**data, "messages": captain_msgs, "max_tokens": 1024}) if rh_is_up() else call_as_kirk({**data, "messages": captain_msgs, "max_tokens": 1024})'
    ),
    # Single hemisphere fallback
    (
        '    target_url = LH_URL if hemisphere == "lh" else RH_URL',
        '    target_url = LH_URL if (hemisphere == "lh" or not rh_is_up()) else RH_URL'
    ),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new, 1)
        print(f"Replaced: {old[:60]}...")
    else:
        print(f"SKIP (not found): {old[:60]}...")

with open(RAG_PATH, "w") as f:
    f.write(content)

print("\nDone. RAG server patched with Kirk fallback.")
