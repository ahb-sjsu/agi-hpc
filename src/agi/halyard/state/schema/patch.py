# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Minimal RFC 6902 JSON-Patch applier with field-level authz.

We implement the subset of JSON-Patch actually used by the Halyard
Table: ``add``, ``remove``, ``replace``. That covers every operation
the web client and Keeper console need; ``move``, ``copy``, and
``test`` are unimplemented (and rejected at apply-time).

Design notes:

- **Path parsing is strict.** Paths must be valid JSON-Pointers
  (RFC 6901) starting with ``/``; segment-level escapes (``~0``,
  ``~1``) are honored. Invalid paths raise :class:`PatchError`.
- **Authorization runs per-op.** Every op's path is authz-checked
  against :mod:`.access`. A deny on any op fails the whole patch
  (no partial application).
- **Copy-on-write.** The sheet passed in is not mutated;
  :func:`apply_patch` returns a new dict. This keeps the NATS
  bridge safe to re-run on transient failures.
- **Schema validation is the caller's job.** After applying a
  patch, call :func:`agi.halyard.state.schema.validate_sheet` on
  the result to catch patches that would produce an invalid
  sheet. The store does this; callers that bypass the store must
  do the same.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .access import Author, can_write

# ─────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────


class PatchError(ValueError):
    """Base class — any apply-time patch failure."""


class AuthorizationError(PatchError):
    """The author is not permitted to write to a path in the patch."""


class InvalidPatchError(PatchError):
    """The patch is malformed or targets an invalid path."""


# ─────────────────────────────────────────────────────────────────
# JSON-Pointer parsing
# ─────────────────────────────────────────────────────────────────


def _parse_pointer(pointer: str) -> list[str]:
    """Parse a JSON Pointer (RFC 6901) into a list of segments.

    An empty pointer (``""``) refers to the whole document and
    parses to an empty list. A pointer starting with anything other
    than ``/`` is invalid.

    JSON-Pointer escapes: ``~1`` → ``/``, ``~0`` → ``~``. Order
    matters: decode ``~1`` before ``~0`` so a literal ``~1`` in
    the source isn't mistaken for an escaped slash.
    """
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise InvalidPatchError(
            f"JSON Pointer must start with '/' (got: {pointer!r})"
        )
    out: list[str] = []
    for seg in pointer[1:].split("/"):
        # Unescape: ~1 → /, ~0 → ~ — order matters.
        seg = seg.replace("~1", "/").replace("~0", "~")
        out.append(seg)
    return out


def _get_parent(doc: dict[str, Any], segments: list[str]) -> tuple[Any, str]:
    """Return ``(parent, last_segment)`` for a non-empty path.

    Raises :class:`InvalidPatchError` if the path doesn't exist
    up to the parent (we don't auto-vivify).
    """
    if not segments:
        raise InvalidPatchError("cannot operate on the document root")
    parent: Any = doc
    for seg in segments[:-1]:
        if isinstance(parent, list):
            try:
                idx = int(seg)
            except ValueError as e:
                raise InvalidPatchError(
                    f"non-integer index {seg!r} in array path"
                ) from e
            try:
                parent = parent[idx]
            except IndexError as e:
                raise InvalidPatchError(
                    f"array index {idx} out of range"
                ) from e
        elif isinstance(parent, dict):
            if seg not in parent:
                raise InvalidPatchError(
                    f"missing path segment {seg!r}"
                )
            parent = parent[seg]
        else:
            raise InvalidPatchError(
                f"cannot descend into {type(parent).__name__} at {seg!r}"
            )
    return parent, segments[-1]


# ─────────────────────────────────────────────────────────────────
# Single-op apply
# ─────────────────────────────────────────────────────────────────


def _apply_op(doc: dict[str, Any], op: dict[str, Any]) -> None:
    """Apply one op in place to ``doc``. Raises on error.

    Supported ops: ``add``, ``remove``, ``replace``.
    """
    op_kind = op.get("op")
    path = op.get("path")
    if not isinstance(op_kind, str):
        raise InvalidPatchError("op missing 'op' field")
    if not isinstance(path, str):
        raise InvalidPatchError("op missing 'path' field")
    if op_kind not in {"add", "remove", "replace"}:
        raise InvalidPatchError(
            f"unsupported op {op_kind!r}; allowed: add, remove, replace"
        )

    segments = _parse_pointer(path)

    if op_kind == "remove":
        parent, key = _get_parent(doc, segments)
        if isinstance(parent, list):
            try:
                del parent[int(key)]
            except (ValueError, IndexError) as e:
                raise InvalidPatchError(
                    f"cannot remove at {path}: {e}"
                ) from e
        elif isinstance(parent, dict):
            if key not in parent:
                raise InvalidPatchError(
                    f"cannot remove missing key {key!r}"
                )
            del parent[key]
        else:
            raise InvalidPatchError(
                f"cannot remove from {type(parent).__name__}"
            )
        return

    if "value" not in op:
        raise InvalidPatchError(f"{op_kind!r} op missing 'value' field")
    value = op["value"]

    parent, key = _get_parent(doc, segments)
    if isinstance(parent, list):
        if key == "-":
            # RFC 6902: "-" appends.
            parent.append(value)
            return
        try:
            idx = int(key)
        except ValueError as e:
            raise InvalidPatchError(
                f"non-integer array index {key!r}"
            ) from e
        if op_kind == "add":
            if idx < 0 or idx > len(parent):
                raise InvalidPatchError(
                    f"add index {idx} out of range (size={len(parent)})"
                )
            parent.insert(idx, value)
        else:  # replace
            if idx < 0 or idx >= len(parent):
                raise InvalidPatchError(
                    f"replace index {idx} out of range (size={len(parent)})"
                )
            parent[idx] = value
        return

    if isinstance(parent, dict):
        if op_kind == "replace" and key not in parent:
            raise InvalidPatchError(
                f"replace on missing key {key!r}"
            )
        parent[key] = value
        return

    raise InvalidPatchError(
        f"cannot write into {type(parent).__name__}"
    )


# ─────────────────────────────────────────────────────────────────
# Public entry
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PatchRequest:
    """Normalized patch request.

    ``author`` is the write-origin; ``author_pc_id`` is set only if
    ``author == PLAYER`` and identifies which player is writing.
    ``target_pc_id`` is the sheet being modified; for a
    player-origin write, this must equal ``author_pc_id`` or the
    patch is refused.

    ``reason`` is a short free-text tag that goes into the
    append-only log — useful for post-session review and for
    Keeper audit of why a specific SAN drop happened.
    """

    session_id: str
    target_pc_id: str
    author: Author
    author_pc_id: str | None
    patch: list[dict[str, Any]]
    reason: str = ""


def apply_patch(
    sheet: dict[str, Any],
    request: PatchRequest,
) -> dict[str, Any]:
    """Apply a PatchRequest to a sheet, returning a new sheet.

    Raises :class:`AuthorizationError` on authz failure, and
    :class:`InvalidPatchError` on malformed ops or path errors.

    Does NOT validate the result against the schema — the caller
    does that after patch application.
    """
    if not isinstance(request.patch, list):
        raise InvalidPatchError("patch must be a list of ops")

    # Authorize every op first. Denying early keeps the sheet in a
    # consistent state if a multi-op patch is rejected partway.
    for op in request.patch:
        path = op.get("path")
        if not isinstance(path, str):
            raise InvalidPatchError("op missing 'path' field")
        ok, reason = can_write(
            path=path,
            author=request.author,
            author_pc_id=request.author_pc_id,
            target_pc_id=request.target_pc_id,
        )
        if not ok:
            raise AuthorizationError(reason)

    # Copy-on-write, then apply in order.
    new_sheet = copy.deepcopy(sheet)
    for op in request.patch:
        _apply_op(new_sheet, op)
    return new_sheet
