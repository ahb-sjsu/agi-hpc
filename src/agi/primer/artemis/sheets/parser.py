# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""CSV → character-row parser for the Keeper's Google Sheet.

The Keeper owns a published-to-web CSV whose columns are the union of
everything any ARTEMIS surface might want: HUD stats, skills,
equipment, notes. Not every column flows into the player HUD — the
HUD wants a minimal, non-crowded view. :func:`row_for_hud` projects a
:class:`CharacterRow` down to the fields the HUD card renders.

The parser is pure and has no NATS / HTTP dependency, so it tests
from fixture strings.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

log = logging.getLogger("artemis.sheets.parser")

# Canonical column names the parser understands. Extra columns are
# preserved on the row so downstream consumers (portal, Primer context)
# can read them without a parser change.
CHARACTERISTIC_COLS = ("str", "con", "siz", "dex", "app", "int", "pow", "edu")
STAT_COLS = (
    "san",
    "san_max",
    "hp",
    "hp_max",
    "luck",
    "luck_max",
    "mp",
    "mp_max",
)
META_COLS = ("id", "name", "role", "occupation", "age", "status", "notes")
JSON_COLS = ("skills_json", "equipment_json")

# The subset of fields the player HUD renders. Keep minimal — the
# player table UI is cramped; everything else is portal/context-only.
HUD_FIELDS = (
    "id",
    "name",
    "role",
    "san",
    "san_max",
    "hp",
    "hp_max",
    "luck",
    "luck_max",
    "mp",
    "mp_max",
    "status",
)

VALID_STATUSES = {"ok", "shaken", "injured", "critical", "dead"}


@dataclass
class CharacterRow:
    """Parsed character row from the published sheet.

    ``id`` is required — the HUD keys by it. Everything else is
    optional; missing fields stay ``None`` / empty and the UI
    degrades gracefully.
    """

    id: str
    name: str = ""
    role: str = ""
    occupation: str = ""
    age: int | None = None
    # CoC characteristics
    str_: int | None = None
    con: int | None = None
    siz: int | None = None
    dex: int | None = None
    app: int | None = None
    int_: int | None = None
    pow_: int | None = None
    edu: int | None = None
    # Derived stats + caps
    san: int | None = None
    san_max: int | None = None
    hp: int | None = None
    hp_max: int | None = None
    luck: int | None = None
    luck_max: int | None = None
    mp: int | None = None
    mp_max: int | None = None
    # Status track
    status: str = "ok"
    notes: str = ""
    # Free-form structured data
    skills: list[dict[str, Any]] = field(default_factory=list)
    equipment: list[dict[str, Any]] = field(default_factory=list)
    # Any other columns the sheet carries that we didn't know about
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Stable dict representation — keys use HUD-friendly names.

        `str_` / `int_` / `pow_` are suffixed in the dataclass to avoid
        shadowing Python builtins; in the dict they become `str` / etc.
        """
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "occupation": self.occupation,
            "age": self.age,
            "str": self.str_,
            "con": self.con,
            "siz": self.siz,
            "dex": self.dex,
            "app": self.app,
            "int": self.int_,
            "pow": self.pow_,
            "edu": self.edu,
            "san": self.san,
            "san_max": self.san_max,
            "hp": self.hp,
            "hp_max": self.hp_max,
            "luck": self.luck,
            "luck_max": self.luck_max,
            "mp": self.mp,
            "mp_max": self.mp_max,
            "status": self.status,
            "notes": self.notes,
            "skills": self.skills,
            "equipment": self.equipment,
        }
        # Drop None-valued characteristics so they don't clutter the
        # payload when the Keeper hasn't filled them in.
        return {k: v for k, v in out.items() if v is not None and v != ""}


def _to_int(raw: str) -> int | None:
    s = (raw or "").strip()
    if not s:
        return None
    try:
        return int(float(s))  # tolerate "11.0" from Sheets auto-formatting
    except ValueError:
        return None


def _to_str(raw: str) -> str:
    return (raw or "").strip()


def _parse_json_cell(raw: str) -> list[dict[str, Any]]:
    """JSON-in-cell parser. Empty or malformed → empty list.

    Sheets users often copy/paste text with curly quotes; we only
    accept strictly valid JSON here to avoid silently-wrong parses.
    """
    s = (raw or "").strip()
    if not s:
        return []
    try:
        val = json.loads(s)
    except Exception as e:  # noqa: BLE001
        log.warning("bad JSON cell dropped: %s (%s)", s[:80], e)
        return []
    if isinstance(val, list):
        return [x for x in val if isinstance(x, dict)]
    return []


def _normalize_status(raw: str) -> str:
    s = (raw or "").strip().lower()
    return s if s in VALID_STATUSES else "ok"


def _alias(key: str) -> str:
    """Column-header alias map → canonical key."""
    k = key.strip().lower().replace(" ", "_")
    # The dataclass uses `str_` / `int_` / `pow_` to dodge builtins, but
    # Keepers writing column headers won't know or care — accept the
    # natural names.
    return {
        "str": "str_",
        "int": "int_",
        "pow": "pow_",
    }.get(k, k)


def _build_row(record: dict[str, str]) -> CharacterRow | None:
    """Turn one CSV DictReader record into a :class:`CharacterRow`.

    Returns ``None`` if the row has no ``id`` — those are treated as
    blank template rows the Keeper hasn't filled in yet.
    """
    normalized = {_alias(k): v for k, v in record.items() if k}
    cid = _to_str(normalized.get("id", ""))
    if not cid:
        return None

    row = CharacterRow(id=cid)
    row.name = _to_str(normalized.get("name", ""))
    row.role = _to_str(normalized.get("role", ""))
    row.occupation = _to_str(normalized.get("occupation", ""))
    row.age = _to_int(normalized.get("age", ""))
    row.status = _normalize_status(normalized.get("status", ""))
    row.notes = _to_str(normalized.get("notes", ""))

    for col in CHARACTERISTIC_COLS:
        key = _alias(col)
        setattr(row, key, _to_int(normalized.get(key, "")))
    for col in STAT_COLS:
        setattr(row, col, _to_int(normalized.get(col, "")))

    row.skills = _parse_json_cell(normalized.get("skills_json", ""))
    row.equipment = _parse_json_cell(normalized.get("equipment_json", ""))

    # Record any leftover columns so the Keeper can add ad-hoc fields
    # (e.g. "phobia", "current_weapon") without code changes.
    known = (
        set(META_COLS)
        | set(STAT_COLS)
        | {_alias(c) for c in CHARACTERISTIC_COLS}
        | set(JSON_COLS)
    )
    row.extra = {
        k: _to_str(v)
        for k, v in normalized.items()
        if k not in known and v and v.strip()
    }
    return row


def parse_csv(text: str) -> list[CharacterRow]:
    """Parse CSV text into a list of :class:`CharacterRow`.

    Blank rows and rows without an ``id`` are skipped silently —
    the Keeper routinely leaves blank rows as spacers in the sheet.
    """
    reader = csv.DictReader(io.StringIO(text))
    rows: list[CharacterRow] = []
    for rec in reader:
        row = _build_row(rec)
        if row is not None:
            rows.append(row)
    return rows


def row_for_hud(row: CharacterRow | dict[str, Any]) -> dict[str, Any]:
    """Project a row down to just the HUD fields.

    Accepts either a :class:`CharacterRow` or its dict form. Missing
    fields are carried through as-is so the JS side can decide how
    to render (usually: don't draw that bar).
    """
    d = row.to_dict() if isinstance(row, CharacterRow) else dict(row)
    return {k: d.get(k) for k in HUD_FIELDS if k in d or k == "id"}


def diff_rows(
    old: Iterable[CharacterRow | dict[str, Any]],
    new: Iterable[CharacterRow | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return rows in ``new`` whose HUD-projection differs from ``old``.

    Diff granularity is the HUD projection — a change in a non-HUD
    column (equipment tweak, skill value) doesn't trigger a HUD
    update. The portal subscribes to a different subject that carries
    full rows, so it gets those changes too.
    """
    old_map: dict[str, dict[str, Any]] = {}
    for r in old:
        d = r.to_dict() if isinstance(r, CharacterRow) else dict(r)
        old_map[d.get("id", "")] = row_for_hud(d)

    out: list[dict[str, Any]] = []
    for r in new:
        d = r.to_dict() if isinstance(r, CharacterRow) else dict(r)
        cid = d.get("id", "")
        if not cid:
            continue
        hud = row_for_hud(d)
        if old_map.get(cid) != hud:
            out.append(hud)
    return out
