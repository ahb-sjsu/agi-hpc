# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Field-level access control for character-sheet patches.

Every JSON-path in the sheet belongs to one of three access tiers:

- **public**    — nobody can patch it after sheet creation (e.g.
                  identity, characteristics, derived ceilings).
- **player**    — the character's own player may patch (current
                  HP/SAN/Luck, improvement checks, condition
                  markers within the player's own character).
- **keeper**    — only the Keeper may patch (faction loyalty,
                  hidden hooks, Keeper-private fields).

A Keeper may patch anything. A player's patch is denied unless
its target path matches a ``player`` pattern AND the ``pc_id`` on
the envelope matches the player's own PC.

The mapping lives here as a single source of truth so the
validator, API authz middleware, and tests all consult the same
list.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Final


class AccessTier(Enum):
    PUBLIC = "public"
    PLAYER = "player"
    KEEPER = "keeper"


class Author(Enum):
    """Who is authoring a patch. The state service accepts this in
    the patch envelope's ``author`` field."""

    KEEPER = "keeper"
    PLAYER = "player"  # authorized against the envelope's pc_id
    SYSTEM = "system"  # rule-engine, LLM-driven effects, etc. — keeper-equivalent


# ─────────────────────────────────────────────────────────────────
# Path patterns
#
# JSON-Pointer style paths (RFC 6901), with a trailing `*` matching
# any single path-segment, and `**` matching zero or more segments.
# The tests exercise the pattern set; new fields added to the
# schema should be classified here explicitly.
# ─────────────────────────────────────────────────────────────────


# Public / immutable after creation.
_PUBLIC_PATHS: tuple[str, ...] = (
    "/schema_version",
    "/session_id",
    "/pc_id",
    "/identity/name",
    "/identity/age",
    "/identity/origin",
    "/identity/role",
    "/identity/chassis",
    "/identity/credit_rating",
    "/identity/pronouns",
    "/identity/voice",
    "/identity/visual",
    "/characteristics/*",
    "/derived/*",
    "/skills/*/base",  # skill starting-base immutable after creation
)

# Player-writable (their own sheet only). Covers routine in-session
# state changes: current HP/SAN/Luck/MP, skill improvement checks,
# bond status (player can mark their own bond "reaffirmed" during
# a scene), conditions that the player tracks themselves.
_PLAYER_PATHS: tuple[str, ...] = (
    "/status/hp_current",
    "/status/mp_current",
    "/status/san_current",
    "/status/luck_current",
    "/status/conditions/**",
    "/skills/*/value",
    "/skills/*/improvement_check",
    "/bonds/*/status",
    "/bonds/*/detail",
    "/equipment/**",
)

# Keeper-only. Faction loyalty, hidden hooks, Keeper-private fields,
# and structural changes the player is not permitted to make
# (adding/removing bonds and skills; the rem-log is Keeper-maintained
# because it reflects environment, not player choice).
_KEEPER_PATHS: tuple[str, ...] = (
    "/campaign/**",
    "/bonds/-",              # "-" per RFC 6902 = append
    "/bonds",                # whole-array replace
    "/skills/-",
    "/skills",
)


def _compile_pattern(glob_path: str) -> re.Pattern[str]:
    r"""Convert a path-glob to a regex.

    Glob syntax:

    - ``*`` — matches one path segment (no slashes).
    - ``/**`` — matches zero or more path segments (including the
      preceding slash as optional). Must appear as a ``/``-prefixed
      pair; a bare ``**`` not preceded by ``/`` is treated as two
      consecutive ``*``, which is not useful.
    - literal path chars match themselves.

    For example, ``/status/conditions/**`` matches
    ``/status/conditions`` (the whole array) as well as
    ``/status/conditions/0``, ``/status/conditions/0/note``, etc.

    Implementation: we look for the ``/**`` digraph and emit
    ``(?:/.*)?`` for the three-char run; otherwise ``*`` is a
    single-segment wildcard and literals are escaped.
    """
    parts: list[str] = []
    i = 0
    while i < len(glob_path):
        # /** → optional tail of any depth (including no tail).
        if glob_path[i:i + 3] == "/**":
            parts.append("(?:/.*)?")
            i += 3
            continue
        if glob_path[i] == "*":
            parts.append("[^/]+")
            i += 1
            continue
        parts.append(re.escape(glob_path[i]))
        i += 1
    return re.compile("^" + "".join(parts) + "$")


_PUBLIC_RES: Final[tuple[re.Pattern[str], ...]] = tuple(
    _compile_pattern(p) for p in _PUBLIC_PATHS
)
_PLAYER_RES: Final[tuple[re.Pattern[str], ...]] = tuple(
    _compile_pattern(p) for p in _PLAYER_PATHS
)
_KEEPER_RES: Final[tuple[re.Pattern[str], ...]] = tuple(
    _compile_pattern(p) for p in _KEEPER_PATHS
)


def tier_for_path(path: str) -> AccessTier:
    """Classify a JSON-Pointer path into its access tier.

    Decision order:

      1. Exact PUBLIC match → PUBLIC (nobody can write).
      2. KEEPER match → KEEPER (only Keeper).
      3. PLAYER match → PLAYER (own-PC player or Keeper).
      4. Unclassified → KEEPER (safe default — if it's a path
         we haven't classified, Keeper-only is the right refusal
         stance).

    The unclassified → KEEPER default is deliberate: if a future
    schema-change adds a field and we forget to classify it here,
    the fallout is "Keeper is the only one who can write to it,"
    which is safe. The opposite default (PUBLIC) would open a
    silent authorization hole.
    """
    if any(r.match(path) for r in _PUBLIC_RES):
        return AccessTier.PUBLIC
    if any(r.match(path) for r in _KEEPER_RES):
        return AccessTier.KEEPER
    if any(r.match(path) for r in _PLAYER_RES):
        return AccessTier.PLAYER
    return AccessTier.KEEPER


def can_write(
    *,
    path: str,
    author: Author,
    author_pc_id: str | None,
    target_pc_id: str,
) -> tuple[bool, str]:
    """Decide whether ``author`` may write to ``path`` on the sheet
    identified by ``target_pc_id``.

    Returns ``(allowed, reason)``. ``reason`` is a short English
    diagnostic suitable for inclusion in a 403 body.
    """
    tier = tier_for_path(path)

    if tier is AccessTier.PUBLIC:
        # Identity / characteristics / derived ceilings: nobody
        # writes these during play. A Keeper "override" to fix a
        # character-creation typo is a rare exception we'd do via
        # a dedicated admin endpoint, not via patch.
        return False, f"path {path} is immutable after sheet creation"

    if author is Author.KEEPER or author is Author.SYSTEM:
        return True, "keeper-authorized"

    if author is Author.PLAYER:
        if tier is AccessTier.KEEPER:
            return False, (
                f"path {path} is Keeper-restricted; player may not patch"
            )
        if author_pc_id is None:
            return False, "player patch requires author_pc_id"
        if author_pc_id != target_pc_id:
            return False, (
                f"player {author_pc_id} may not patch sheet of "
                f"{target_pc_id}"
            )
        return True, "player-authorized"

    return False, f"unknown author kind: {author}"
