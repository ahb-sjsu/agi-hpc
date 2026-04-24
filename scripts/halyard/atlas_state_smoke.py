"""Halyard Table — halyard-state end-to-end smoke test.

Exercises the service as a browser client would: create a sheet,
patch it, observe the WebSocket fan-out, verify authz and schema
refusals.

**Binding policy.** The service binds to loopback on Atlas. Reach
it via one of:

- SSH port-forward::

    ssh -L 8090:127.0.0.1:8090 claude@100.68.134.21
    python scripts/halyard/atlas_state_smoke.py

- Once atlas-caddy is fronting the service::

    python scripts/halyard/atlas_state_smoke.py \\
        --base https://halyard-state.atlas-sjsu.duckdns.org

Tailscale is the admin backdoor, **not** a service channel; this
script does not default to the tailnet address.

Exits 0 on success. Every step prints a tagged line so failures
are pinpointable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import aiohttp

DEFAULT_BASE = "http://127.0.0.1:8090"  # assumes SSH port-forward
SESSION = "halyard-smoke-e2e"
PC = "cross"


SHEET = {
    "schema_version": "1.0",
    "session_id": SESSION,
    "pc_id": PC,
    "identity": {
        "name": "Halden Cross",
        "age": 41,
        "origin": "Luna, Apennine",
        "role": "security_officer",
        "chassis": "baseline_human",
        "credit_rating": 35,
    },
    "characteristics": {
        "str": 75, "con": 75, "siz": 65, "dex": 70,
        "app": 55, "int": 65, "pow": 55, "edu": 60,
    },
    "derived": {
        "hp_max": 14, "mp_max": 11, "san_starting": 55,
        "san_max": 65, "luck_max": 55, "move": 8,
        "build": 1, "damage_bonus": "+1D4", "dodge_base": 35,
    },
    "skills": {
        "firearms (handgun)": {"value": 70, "base": 20},
        "spot hidden": {"value": 60, "base": 25},
    },
    "bonds": [],
    "status": {
        "hp_current": 14, "mp_current": 11,
        "san_current": 55, "luck_current": 55,
    },
    "campaign": {"faction_loyalty": "clean"},
}


async def _main(base: str) -> int:
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as http:
        print(f"Base URL: {base}")
        BASE = base  # noqa: N806 — kept local for readability below
        # ── 1. Health ──
        async with http.get(f"{BASE}/healthz") as r:
            body = await r.json()
            assert r.status == 200 and body.get("ok") is True, body
            print(f"[1] healthz: {body}")

        # ── 2. List empty ──
        async with http.get(f"{BASE}/api/sheets/{SESSION}") as r:
            body = await r.json()
            print(f"[2] list empty session: {body}")

        # ── 3. Create sheet (deleting any stale one from a prior run) ──
        async with http.post(
            f"{BASE}/api/sheets/{SESSION}/{PC}", json=SHEET
        ) as r:
            if r.status == 409:
                print("[3] sheet already exists — using existing")
            else:
                body = await r.json()
                assert r.status == 201, f"create failed {r.status}: {body}"
                print(f"[3] create 201: name={body['identity']['name']}")

        # ── 4. Get the sheet ──
        async with http.get(f"{BASE}/api/sheets/{SESSION}/{PC}") as r:
            body = await r.json()
            assert r.status == 200 and body["pc_id"] == PC
            print(f"[4] get: hp_current={body['status']['hp_current']}")

        # ── 5. WebSocket subscribe and watch for an update ──
        print("[5] opening WS subscription...")
        ws_events: list[dict] = []

        async def _ws_listener():
            async with http.ws_connect(
                f"{BASE.replace('http', 'ws')}/ws/sheets/{SESSION}"
            ) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        ev = json.loads(msg.data)
                        ws_events.append(ev)
                        print(f"    ws event: {ev.get('kind')} "
                              f"pc={ev.get('pc_id','-')}")
                        if len(ws_events) >= 2:
                            break

        listener_task = asyncio.create_task(_ws_listener())
        await asyncio.sleep(0.4)  # let the hello land

        # ── 6. Apply a keeper patch — should fan out over WS ──
        envelope = {
            "author": "keeper",
            "patch": [
                {"op": "replace", "path": "/status/san_current", "value": 48}
            ],
            "reason": "smoke test — forced SAN 1D6 drop",
        }
        async with http.post(
            f"{BASE}/api/sheets/{SESSION}/{PC}/patch", json=envelope
        ) as r:
            body = await r.json()
            assert r.status == 200, f"patch failed {r.status}: {body}"
            print(f"[6] patch 200: san_current={body['status']['san_current']}")

        # ── 7. Wait for WS update ──
        try:
            await asyncio.wait_for(listener_task, timeout=3.0)
        except asyncio.TimeoutError:
            listener_task.cancel()
            print(f"[7] WS timeout — events so far: {ws_events}")
            return 1

        kinds = [e.get("kind") for e in ws_events]
        assert "session.hello" in kinds, kinds
        assert "sheet.update" in kinds, kinds
        update = next(e for e in ws_events if e.get("kind") == "sheet.update")
        assert update["sheet"]["status"]["san_current"] == 48
        print(f"[7] WS fan-out verified: events={kinds}")

        # ── 8. Player patch to keeper field (authz) ──
        bad_envelope = {
            "author": "player",
            "author_pc_id": PC,
            "patch": [{
                "op": "replace",
                "path": "/campaign/faction_loyalty",
                "value": "hollow_hand",
            }],
        }
        async with http.post(
            f"{BASE}/api/sheets/{SESSION}/{PC}/patch", json=bad_envelope
        ) as r:
            body = await r.json()
            assert r.status == 403, f"expected 403, got {r.status}: {body}"
            print(f"[8] authz 403: {body.get('code')}")

        # ── 9. Schema-violating patch ──
        bad_envelope = {
            "author": "keeper",
            "patch": [{
                "op": "replace", "path": "/status/san_current", "value": 999,
            }],
        }
        async with http.post(
            f"{BASE}/api/sheets/{SESSION}/{PC}/patch", json=bad_envelope
        ) as r:
            body = await r.json()
            assert r.status == 400, f"expected 400, got {r.status}: {body}"
            print(f"[9] schema-violation 400: {body.get('code')}")

    print()
    print("═" * 60)
    print("  ✓ halyard-state end-to-end smoke test PASSED")
    print(f"  REST: {base}/api/sheets/{SESSION}")
    print(f"  WS:   {base.replace('http', 'ws')}/ws/sheets/{SESSION}")
    print("═" * 60)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=(
            f"Base URL of halyard-state. Default {DEFAULT_BASE} — "
            "assumes SSH port-forward. Set explicitly when Caddy "
            "is fronting the service."
        ),
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args.base)))
