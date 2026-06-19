# AGI-HPC Project — Divine Council NATS Service
# Copyright (c) 2026 Andrew H. Bond
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Divine Council NATS bridge.

Subscribes to ``agi.ego.deliberate`` and actually deliberates:

* **Fire-and-forget** (``fabric.publish("agi.ego.deliberate", ev)``):
  deliberates and publishes the outcome on ``agi.ego.deliberation.result``.
* **Request/reply** (``fabric.request("agi.ego.deliberate", ev, timeout)``):
  the fabric prefixes the subject to ``_rpc.agi.ego.deliberate``; we subscribe
  there at the raw-NATS level and reply on ``msg.reply`` with a response Event.

Deliberation runs on the local **Ego** (Gemma-4-E4B via llama-server on :8084)
by default; set ``payload["council"]=true`` to escalate to an **NRP frontier
model** (the Primer's vMOE resource) with graceful fallback to Ego.

Request payload keys (all optional):
  prompt | question : the thing to deliberate on (else the whole payload)
  system            : system prompt (else a sensible default)
  max_tokens        : int (default 512)
  council           : bool — escalate to NRP frontier model

Heartbeat published to ``agi.meta.monitor.ego`` every 30 s.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.request

from agi.common.event import Event
from agi.core.events.nats_fabric import NatsEventFabric, NatsFabricConfig

logger = logging.getLogger(__name__)

EGO_URL = os.environ.get("EGO_URL", "http://localhost:8084")
EGO_MODEL = os.environ.get("EGO_MODEL", "gemma-4-E4B-it-Q5_K_M.gguf")
NRP_BASE = os.environ.get("NRP_BASE", "https://ellm.nrp-nautilus.io/v1")
NRP_TOKEN_FILE = os.environ.get("NRP_TOKEN_FILE", "/home/claude/bible/.eris_token")
COUNCIL_MODEL = os.environ.get("COUNCIL_MODEL", "kimi")
HEARTBEAT_S = 30

DEFAULT_SYSTEM = (
    "You are the deliberative faculty of Erebus (the Ego, advised by the Divine "
    "Council). Reason carefully and answer the request concisely and usefully. "
    "If asked for a decision, state it plainly with a one-line rationale."
)

_FABRIC: NatsEventFabric | None = None
_NC = None  # raw nats client, for reply-to-inbox


# --------------------------------------------------------------------------- LLM calls
def _ego_chat(messages: list[dict], max_tokens: int = 512, temperature: float = 0.4,
              think: bool = False) -> str:
    # gemma-4-E4B is a THINKING model: with thinking on, a small max_tokens is
    # consumed entirely by reasoning_content and `content` comes back empty.
    # Default to thinking OFF for concise/structured deliberation; callers can
    # opt in via payload["think"] (and should raise max_tokens when they do).
    payload = {"model": EGO_MODEL, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if not think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        f"{EGO_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode("utf-8"))
    msg = d["choices"][0]["message"]
    # prefer final content; if a thinking run left it empty, surface reasoning
    return (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()


def _council_chat(messages: list[dict], max_tokens: int = 512, temperature: float = 0.4,
                  think: bool = False) -> str:
    # NRP frontier models (kimi/glm/qwen3) are THINKING models too: without
    # enable_thinking=false a small budget is spent on reasoning and `content`
    # comes back empty. Disable thinking by default for concise deliberation.
    from openai import OpenAI
    token = open(NRP_TOKEN_FILE).read().strip()
    client = OpenAI(base_url=NRP_BASE, api_key=token)
    extra = {} if think else {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    r = client.chat.completions.create(
        model=COUNCIL_MODEL, messages=messages,
        max_tokens=max_tokens, temperature=temperature, timeout=180, **extra)
    msg = r.choices[0].message
    return (msg.content or "").strip() or (getattr(msg, "reasoning_content", "") or "").strip()


def _deliberate(payload: dict) -> dict:
    """Synchronous deliberation (run via asyncio.to_thread). Never raises."""
    question = payload.get("prompt") or payload.get("question") or json.dumps(payload)
    system = payload.get("system") or DEFAULT_SYSTEM
    think = bool(payload.get("think", False))
    # give thinking runs room; concise runs a sane floor
    max_tokens = int(payload.get("max_tokens", 2048 if think else 512))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": str(question)}]
    want_council = bool(payload.get("council"))
    try:
        if want_council:
            return {"ok": True, "model": COUNCIL_MODEL,
                    "response": _council_chat(messages, max_tokens, think=think)}
        return {"ok": True, "model": f"ego:{EGO_MODEL}",
                "response": _ego_chat(messages, max_tokens, think=think)}
    except Exception as e:  # graceful: council->ego fallback, else error
        if want_council:
            try:
                return {"ok": True, "model": "ego(fallback)",
                        "response": _ego_chat(messages, max_tokens, think=think),
                        "note": f"council failed: {e}"}
            except Exception as e2:
                return {"ok": False, "model": None, "response": "",
                        "error": f"council: {e}; ego: {e2}"}
        return {"ok": False, "model": None, "response": "", "error": str(e)}


# --------------------------------------------------------------------------- handlers
async def _handle_deliberate(event: Event) -> None:
    """Fire-and-forget deliberation -> publish result on agi.ego.deliberation.result."""
    logger.info("[ego-nats] deliberate: %s", str(event.payload)[:120])
    result = await asyncio.to_thread(_deliberate, event.payload)
    out = Event.create("ego", "agi.ego.deliberation.result",
                       {**result, "request_id": event.id, "trace_id": event.trace_id})
    if _FABRIC is not None:
        try:
            await _FABRIC.publish("agi.ego.deliberation.result", out)
        except Exception:
            logger.exception("[ego-nats] failed to publish deliberation result")


async def _respond_rpc(msg) -> None:
    """Request/reply responder on _rpc.agi.ego.deliberate."""
    try:
        event = Event.from_bytes(msg.data)
    except Exception:
        logger.exception("[ego-nats] bad rpc request on %s", getattr(msg, "subject", "?"))
        return
    logger.info("[ego-nats] deliberate(rpc): %s", str(event.payload)[:120])
    result = await asyncio.to_thread(_deliberate, event.payload)
    reply = Event.create("ego", "agi.ego.deliberation", {**result, "request_id": event.id})
    if getattr(msg, "reply", None):
        try:
            await _NC.publish(msg.reply, reply.to_bytes())
        except Exception:
            logger.exception("[ego-nats] failed to reply on inbox %s", msg.reply)


async def _check_health() -> bool:
    try:
        r = await asyncio.to_thread(
            lambda: urllib.request.urlopen(f"{EGO_URL}/health", timeout=3).status)
        return r == 200
    except Exception:
        return False


async def _wire(fabric: NatsEventFabric) -> None:
    global _FABRIC, _NC
    _FABRIC = fabric
    _NC = fabric._nc
    await fabric.subscribe("agi.ego.deliberate", _handle_deliberate)
    await _NC.subscribe("_rpc.agi.ego.deliberate", cb=_respond_rpc)
    logger.info("[ego-nats] deliberation online (ego=%s, council=%s); rpc=_rpc.agi.ego.deliberate",
                EGO_MODEL, COUNCIL_MODEL)


async def run() -> None:
    fabric = NatsEventFabric(config=NatsFabricConfig(servers=["nats://localhost:4222"]))
    await fabric.connect()
    logger.info("[ego-nats] connected")
    await _wire(fabric)
    while True:
        healthy = await _check_health()
        hb = Event.create("ego", "agi.meta.monitor.ego",
                          {"service": "divine_council",
                           "status": "online" if healthy else "offline",
                           "ts": time.time()})
        await fabric.publish("agi.meta.monitor.ego", hb)
        await asyncio.sleep(HEARTBEAT_S)


async def _selftest() -> None:
    """Wire a responder and round-trip a request through the fabric (no heartbeat loop)."""
    fabric = NatsEventFabric(config=NatsFabricConfig(servers=["nats://localhost:4222"]))
    await fabric.connect()
    await _wire(fabric)
    await asyncio.sleep(0.3)
    ev = Event.create("selftest", "agi.ego.deliberate",
                      {"prompt": "In one sentence: is the sky blue, and why?",
                       "max_tokens": 150})
    print("[selftest] sending request to agi.ego.deliberate ...", flush=True)
    reply = await fabric.request("agi.ego.deliberate", ev, timeout=180.0)
    print("[selftest] REPLY:", json.dumps(reply.payload, indent=2)[:600], flush=True)
    try:
        await fabric.disconnect()
    except Exception:
        pass


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if "--selftest" in sys.argv:
        logger.info("[ego-nats] SELFTEST")
        asyncio.run(_selftest())
    else:
        logger.info("[ego-nats] starting Divine Council NATS bridge")
        asyncio.run(run())
