# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""NATS bridge for the halyard-state service.

Subscribes to character-sheet patch subjects and publishes updates
back out. See ``docs/HALYARD_TABLE.md`` §5.1 for the subject
contract.

The bridge is I/O-agnostic — :meth:`handle_patch_message` is the
unit-testable core, and takes already-decoded bytes. :meth:`run`
wraps it in a NATS subscribe loop (injectable for tests).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from agi.halyard import (
    SUBJECT_SHEET_PATCH_FMT,
    SUBJECT_SHEET_UPDATE_FMT,
)

from .schema.access import Author
from .schema.patch import (
    AuthorizationError,
    InvalidPatchError,
    PatchRequest,
)
from .store import NotFoundError, SchemaError, Store

log = logging.getLogger("halyard.state.bridge")


# Match a sheet subject to extract the pc_id.
_SHEET_SUBJECT_RE = re.compile(
    r"^agi\.rh\.halyard\.sheet\.(?P<pc_id>[a-z0-9-]+)\.(?:patch|update)$"
)

# Error subject format — mirrors the patch/update pair so clients
# can subscribe per-PC or by wildcard.
SUBJECT_SHEET_ERROR_FMT = "agi.rh.halyard.sheet.{pc_id}.error"


# ─────────────────────────────────────────────────────────────────
# NATS client protocol — matches nats-py's asyncio client surface
# closely enough that tests can inject fakes.
# ─────────────────────────────────────────────────────────────────


class NatsClient(Protocol):
    """Subset of nats.aio.client.Client used by the bridge."""

    async def publish(self, subject: str, payload: bytes) -> None: ...

    async def subscribe(
        self,
        subject: str,
        *,
        cb: Any = ...,
    ) -> Any: ...


# ─────────────────────────────────────────────────────────────────
# Bridge
# ─────────────────────────────────────────────────────────────────


@dataclass
class BridgeResult:
    """Outcome of handling one incoming patch message.

    Useful as a return value in tests — the bridge publishes back
    over NATS in production, but a BridgeResult is the richer
    shape for assertions.
    """

    ok: bool
    pc_id: str | None
    updated_sheet: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None


class StateBridge:
    """NATS bridge — runs inside the halyard-state service.

    Construction takes a :class:`Store` (required) and an optional
    NATS client (for ``run``; tests pass ``None`` and call
    :meth:`handle_patch_message` directly).
    """

    def __init__(
        self,
        *,
        store: Store,
        nats_client: NatsClient | None = None,
        ws_broadcast: Any = None,
    ) -> None:
        self._store = store
        self._nats = nats_client
        # ws_broadcast is a coroutine ``async def(session_id, payload)``
        # that the api layer plugs in to fan-out to subscribed
        # browsers. None is fine — tests and some deployments may
        # not need WS fan-out.
        self._ws_broadcast = ws_broadcast

    # ── core: testable without NATS ──

    async def handle_patch_message(
        self, subject: str, raw_payload: bytes
    ) -> BridgeResult:
        """Parse, authorize, apply, and publish one patch.

        Never raises — returns a :class:`BridgeResult` with
        structured error codes for any failure. Publishing (update
        or error) happens as a side effect when a NATS client is
        attached.
        """
        # 1. Parse subject → pc_id.
        m = _SHEET_SUBJECT_RE.match(subject)
        if not m:
            return self._err(None, "bad_subject", f"subject {subject!r}")
        pc_id_from_subject = m.group("pc_id")

        # 2. Parse JSON body.
        try:
            payload = json.loads(raw_payload.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return self._err(
                pc_id_from_subject, "bad_json", f"json decode: {e}"
            )

        if not isinstance(payload, dict):
            return self._err(
                pc_id_from_subject, "bad_envelope", "payload is not an object"
            )

        # 3. Build PatchRequest.
        try:
            req = _build_request(payload, pc_id_from_subject)
        except ValueError as e:
            return self._err(pc_id_from_subject, "bad_envelope", str(e))

        # 4. Apply via the store.
        try:
            updated = await self._store.patch(req)
        except AuthorizationError as e:
            return await self._publish_err(
                req.target_pc_id, "authz_denied", str(e)
            )
        except InvalidPatchError as e:
            return await self._publish_err(
                req.target_pc_id, "invalid_patch", str(e)
            )
        except SchemaError as e:
            return await self._publish_err(
                req.target_pc_id, "schema_violation", str(e)
            )
        except NotFoundError as e:
            return await self._publish_err(
                req.target_pc_id, "not_found", str(e)
            )

        # 5. Publish the update + fan-out to WS listeners.
        await self._publish_update(req.target_pc_id, req.session_id, updated)
        return BridgeResult(
            ok=True, pc_id=req.target_pc_id, updated_sheet=updated
        )

    # ── run loop: subscribe-and-dispatch ──

    async def run(self) -> None:  # pragma: no cover — thin wrapper
        """Subscribe to the patch subject and dispatch until
        cancelled. Production entry point."""
        if self._nats is None:
            raise RuntimeError(
                "StateBridge.run requires a NATS client — construct "
                "with nats_client=..."
            )
        # Wildcard subscribe: sheet.*.patch. Subject parsing in
        # handle_patch_message extracts the pc_id.
        wildcard = SUBJECT_SHEET_PATCH_FMT.format(pc_id="*")

        async def _on_msg(msg: Any) -> None:
            try:
                await self.handle_patch_message(msg.subject, msg.data)
            except Exception as e:  # noqa: BLE001
                log.warning("bridge dispatch failed: %s", e)

        await self._nats.subscribe(wildcard, cb=_on_msg)
        log.info("state bridge subscribed to %s", wildcard)

    # ── publish helpers ──

    async def _publish_update(
        self,
        pc_id: str,
        session_id: str,
        sheet: dict[str, Any],
    ) -> None:
        subject = SUBJECT_SHEET_UPDATE_FMT.format(pc_id=pc_id)
        payload = json.dumps(
            {
                "session_id": session_id,
                "pc_id": pc_id,
                "sheet": sheet,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        if self._nats is not None:
            try:
                await self._nats.publish(subject, payload)
            except Exception as e:  # noqa: BLE001
                log.warning("update publish failed (%s): %s", subject, e)
        if self._ws_broadcast is not None:
            try:
                await self._ws_broadcast(session_id, {
                    "kind": "sheet.update",
                    "pc_id": pc_id,
                    "sheet": sheet,
                })
            except Exception as e:  # noqa: BLE001
                log.warning("ws broadcast failed: %s", e)

    async def _publish_err(
        self, pc_id: str | None, code: str, message: str
    ) -> BridgeResult:
        """Publish a structured error subject and return the result."""
        if pc_id is not None and self._nats is not None:
            subject = SUBJECT_SHEET_ERROR_FMT.format(pc_id=pc_id)
            payload = json.dumps(
                {"code": code, "message": message},
                separators=(",", ":"),
            ).encode("utf-8")
            try:
                await self._nats.publish(subject, payload)
            except Exception as e:  # noqa: BLE001
                log.warning("err publish failed (%s): %s", subject, e)
        return BridgeResult(
            ok=False,
            pc_id=pc_id,
            error_code=code,
            error_message=message,
        )

    def _err(
        self, pc_id: str | None, code: str, message: str
    ) -> BridgeResult:
        """Synchronous error result (no publish)."""
        return BridgeResult(
            ok=False,
            pc_id=pc_id,
            error_code=code,
            error_message=message,
        )


# ─────────────────────────────────────────────────────────────────
# Envelope parsing
# ─────────────────────────────────────────────────────────────────


def _build_request(
    payload: dict[str, Any], pc_id_from_subject: str
) -> PatchRequest:
    """Decode an inbound patch envelope into a :class:`PatchRequest`.

    Raises :class:`ValueError` on malformed input.
    """
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("missing or empty 'session_id'")

    pc_id = payload.get("pc_id", pc_id_from_subject)
    if not isinstance(pc_id, str) or not pc_id:
        raise ValueError("missing or empty 'pc_id'")
    if pc_id != pc_id_from_subject:
        raise ValueError(
            f"envelope pc_id {pc_id!r} does not match subject "
            f"{pc_id_from_subject!r}"
        )

    author_raw = payload.get("author", "keeper")
    try:
        author = Author(author_raw)
    except ValueError as e:
        raise ValueError(f"unknown author {author_raw!r}") from e

    author_pc_id = payload.get("author_pc_id")
    if author is Author.PLAYER and not isinstance(author_pc_id, str):
        raise ValueError("player-origin patch requires 'author_pc_id'")

    patch = payload.get("patch")
    if not isinstance(patch, list):
        raise ValueError("missing or non-list 'patch'")

    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)

    return PatchRequest(
        session_id=session_id,
        target_pc_id=pc_id,
        author=author,
        author_pc_id=author_pc_id,
        patch=patch,
        reason=reason,
    )
