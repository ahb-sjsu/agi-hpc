#!/usr/bin/env python3
"""Sanity probe for the Erebus ego pod.

Hits the pod's OpenAI-compatible /v1/chat/completions with a short
prompt and reports latency, token usage, and a few health checks.
Used once per probe deployment to validate the stack end-to-end
before promoting to a persistent Deployment.

Usage:
    # From a workstation with kubectl port-forward to the pod:
    kubectl -n ssu-atlas-ai port-forward svc/erebus-ego 8000:8000 &
    python3 scripts/probe_erebus_ego.py

    # Direct, from inside the cluster:
    python3 scripts/probe_erebus_ego.py \\
        --url http://erebus-ego.ssu-atlas-ai.svc.cluster.local:8000

Exit code 0 = healthy, non-zero = something wrong worth investigating.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def _get(url: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return 0, f"error: {type(e).__name__}: {e}"


def _chat(
    base_url: str,
    model: str,
    prompt: str,
    *,
    timeout: float,
    max_tokens: int,
) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8", errors="replace"))
            payload["_latency_s"] = time.time() - t0
            return payload
    except urllib.error.HTTPError as e:
        return {
            "_error": f"HTTP {e.code}",
            "_latency_s": time.time() - t0,
            "_body": e.read().decode("utf-8", errors="replace")[:500],
        }
    except Exception as e:  # noqa: BLE001
        return {
            "_error": f"{type(e).__name__}: {e}",
            "_latency_s": time.time() - t0,
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the erebus-ego Service (no trailing slash)",
    )
    p.add_argument(
        "--model",
        default="erebus-ego",
        help="served-model-name from the vLLM args",
    )
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args(argv)

    base = args.url.rstrip("/")

    print(f"probing erebus-ego at {base}")

    # Step 1 — /health must be 200
    health_code, health_body = _get(f"{base}/health")
    if health_code != 200:
        print(f"  /health FAIL: {health_code} {health_body[:200]}")
        return 2
    print("  /health ok")

    # Step 2 — /v1/models must list our model
    models_code, models_body = _get(f"{base}/v1/models")
    if models_code != 200:
        print(f"  /v1/models FAIL: {models_code} {models_body[:200]}")
        return 3
    try:
        served = [m.get("id") for m in json.loads(models_body).get("data", [])]
    except Exception:
        served = []
    if args.model not in served:
        print(f"  /v1/models WARN: '{args.model}' not in served list: {served}")
    else:
        print(f"  /v1/models lists {args.model}")

    # Step 3 — a round-trip chat completion
    prompt = "Reply with exactly the word ACK and nothing else."
    resp = _chat(
        base, args.model, prompt, timeout=args.timeout, max_tokens=args.max_tokens
    )
    if "_error" in resp:
        print(
            f"  /v1/chat/completions FAIL: {resp['_error']} "
            f"(latency {resp['_latency_s']:.1f}s)"
        )
        body = resp.get("_body")
        if body:
            print(f"    body: {body}")
        return 4
    content = ""
    try:
        content = resp["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    usage = resp.get("usage") or {}
    print(
        f"  /v1/chat/completions ok: latency={resp['_latency_s']:.1f}s "
        f"prompt_tok={usage.get('prompt_tokens')} "
        f"completion_tok={usage.get('completion_tokens')}"
    )
    print(f"    reply: {content[:80]!r}")

    # Basic sanity: did it actually produce something?
    if not content:
        print("  chat WARN: empty completion")
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
