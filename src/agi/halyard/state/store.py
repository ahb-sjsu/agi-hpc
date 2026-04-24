# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""On-disk character-sheet storage with append-only patch log.

File layout::

  {SESSION_ARCHIVE_ROOT}/
  └── sheets/
      └── {session_id}/
          ├── {pc_id}.json             # canonical current state
          └── log.jsonl                # append-only patch log

Each sheet lives at ``.json``; the ``log.jsonl`` holds every
:class:`.schema.patch.PatchRequest` that was applied, in order,
with a ts timestamp and the resulting proof-hash-style state
digest. The log is the record for session replay.

Concurrency: one :class:`Store` instance per-process is the
expected deployment. Per-sheet writes are serialized inside this
class via a single :class:`asyncio.Lock`; multi-process writes
are not safe (and not needed — halyard-state is a singleton).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .schema import validate_sheet
from .schema.patch import PatchRequest, apply_patch

log = logging.getLogger("halyard.state.store")


# ─────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────


class NotFoundError(KeyError):
    """Sheet at ``(session_id, pc_id)`` does not exist."""


class AlreadyExistsError(ValueError):
    """Attempt to create a sheet that already exists."""


class SchemaError(ValueError):
    """Post-patch sheet does not validate against the schema."""


# ─────────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────────


def _sheet_digest(sheet: dict[str, Any]) -> str:
    """SHA-256 of the canonical-JSON encoding. Used as a weak
    tamper-detection fingerprint in the log; not meant for
    cryptographic signature."""
    canonical = json.dumps(sheet, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Store:
    """Per-process character-sheet store.

    Construction does no I/O. :meth:`load_session` and the put/get
    methods are async to keep the surface consistent with a future
    async-IO backend; today the on-disk calls are blocking and
    fast enough that we don't bother off-loading.
    """

    def __init__(
        self,
        *,
        archive_root: Path | None = None,
    ) -> None:
        default = os.environ.get("HALYARD_ARCHIVE_ROOT", "/archive/halyard")
        self._root = Path(archive_root) if archive_root else Path(default)
        self._sheets: dict[tuple[str, str], dict[str, Any]] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    # ── paths ──

    def _session_dir(self, session_id: str) -> Path:
        return self._root / "sheets" / session_id

    def _sheet_path(self, session_id: str, pc_id: str) -> Path:
        return self._session_dir(session_id) / f"{pc_id}.json"

    def _log_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "log.jsonl"

    # ── lock helper ──

    def _lock_for(self, session_id: str, pc_id: str) -> asyncio.Lock:
        key = (session_id, pc_id)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    # ── public API ──

    async def list_sheets(self, session_id: str) -> list[str]:
        """Return all pc_ids known for the session. Order undefined."""
        d = self._session_dir(session_id)
        if not d.is_dir():
            return []
        return sorted(
            p.stem for p in d.glob("*.json") if p.is_file()
        )

    async def get(self, session_id: str, pc_id: str) -> dict[str, Any]:
        """Return the current sheet for ``(session_id, pc_id)``.

        Raises :class:`NotFoundError` if missing.
        """
        key = (session_id, pc_id)
        if key in self._sheets:
            return json.loads(json.dumps(self._sheets[key]))  # defensive copy

        p = self._sheet_path(session_id, pc_id)
        if not p.exists():
            raise NotFoundError(f"sheet not found: {session_id}/{pc_id}")
        data = json.loads(p.read_text(encoding="utf-8"))
        self._sheets[key] = data
        return json.loads(json.dumps(data))

    async def create(
        self,
        session_id: str,
        pc_id: str,
        sheet: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a new sheet.

        Raises :class:`AlreadyExistsError` if already present,
        :class:`SchemaError` on schema violation.
        """
        key = (session_id, pc_id)
        async with self._lock_for(session_id, pc_id):
            if self._sheet_path(session_id, pc_id).exists():
                raise AlreadyExistsError(
                    f"sheet already exists: {session_id}/{pc_id}"
                )
            # Normalize the session_id / pc_id to match the envelope
            # so the caller can't sneak in inconsistent identities.
            sheet = dict(sheet)
            sheet["session_id"] = session_id
            sheet["pc_id"] = pc_id
            try:
                validate_sheet(sheet)
            except Exception as e:
                raise SchemaError(f"invalid sheet: {e}") from e
            self._session_dir(session_id).mkdir(parents=True, exist_ok=True)
            p = self._sheet_path(session_id, pc_id)
            p.write_text(
                json.dumps(sheet, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._sheets[key] = sheet
            # Append a "created" marker to the log for replayability.
            self._append_log(
                session_id,
                {
                    "ts": time.time(),
                    "pc_id": pc_id,
                    "author": "system",
                    "kind": "create",
                    "digest": _sheet_digest(sheet),
                    "reason": "initial sheet",
                },
            )
            return json.loads(json.dumps(sheet))

    async def patch(
        self,
        request: PatchRequest,
    ) -> dict[str, Any]:
        """Apply a patch, persist, log, and return the new sheet.

        Raises :class:`NotFoundError` if the target sheet doesn't
        exist, :class:`agi.halyard.state.schema.patch.PatchError`
        on apply-time issues, :class:`SchemaError` if the post-
        patch state fails validation.
        """
        key = (request.session_id, request.target_pc_id)
        async with self._lock_for(request.session_id, request.target_pc_id):
            current = await self.get(
                request.session_id, request.target_pc_id
            )
            # apply_patch can raise AuthorizationError or
            # InvalidPatchError — let those propagate.
            new_sheet = apply_patch(current, request)
            try:
                validate_sheet(new_sheet)
            except Exception as e:
                raise SchemaError(f"patch would invalidate sheet: {e}") from e
            # Persist.
            p = self._sheet_path(request.session_id, request.target_pc_id)
            p.write_text(
                json.dumps(new_sheet, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._sheets[key] = new_sheet
            # Log.
            self._append_log(
                request.session_id,
                {
                    "ts": time.time(),
                    "pc_id": request.target_pc_id,
                    "author": request.author.value,
                    "author_pc_id": request.author_pc_id,
                    "kind": "patch",
                    "patch": request.patch,
                    "reason": request.reason,
                    "digest": _sheet_digest(new_sheet),
                },
            )
            return json.loads(json.dumps(new_sheet))

    # ── append log ──

    def _append_log(self, session_id: str, record: dict[str, Any]) -> None:
        """Append one record to ``log.jsonl``.

        Never raises — log failures are warned, not propagated.
        Losing a log record is bad but losing a session is worse.
        """
        try:
            p = self._log_path(session_id)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception as e:  # noqa: BLE001
            log.warning(
                "log append failed for %s: %s", session_id, e
            )

    # ── testing helpers ──

    def _drop_cache(self) -> None:
        """Forget everything. Tests only."""
        self._sheets.clear()
        self._locks.clear()
