# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Verification check for wiki sensei notes.

A sensei note is "verified" iff its YAML frontmatter contains a
non-empty ``verified_by:`` field.  The Primer writes this field when
it publishes a note whose reference implementation passed against the
task's train + test fixtures.  Hand-written notes may or may not have
it; those that don't are drafts and must not be loaded by agents.

This is the loader-side enforcement of the verify-before-publish
invariant.  A pre-commit hook enforces the same rule at commit time.
"""

from __future__ import annotations

import re
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_VERIFIED_BY_RE = re.compile(r"^\s*verified_by\s*:\s*(.+?)\s*$", re.MULTILINE)


def is_verified(text: str) -> bool:
    """True if the note's frontmatter has a non-empty ``verified_by:`` field.

    Checks only the YAML frontmatter (between the opening ``---`` and
    the next ``---``). A ``verified_by:`` line elsewhere in the body
    doesn't count: it could be inside a code example or quoted prose.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return False
    frontmatter = m.group(1)
    v = _VERIFIED_BY_RE.search(frontmatter)
    if not v:
        return False
    value = v.group(1).strip()
    # Strip YAML quoting if present
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1].strip()
    return bool(value)


def read_if_verified(path: Path | str) -> str | None:
    """Return the note's text if verified, else None.

    A missing file returns None. A read error returns None (the caller
    usually wants to skip unreadable notes rather than crash).
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return None
    return text if is_verified(text) else None
