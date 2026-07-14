#!/usr/bin/env python3
# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Project Erebus's human-authored charter (markdown) into charter.json.

The Director reads ``charter.json`` each cycle; humans edit ``charter.md``.
This linter keeps the two from drifting: it parses the objectives table and
the limits table (the machine-critical parts) plus the identity and
principles prose, and emits the JSON the ``agi.director.charter.Charter``
loader expects.

Usage:
    python3 scripts/charter_lint.py charter.md [charter.json]

Parsing is deliberately forgiving about surrounding prose — it keys off the
table headers (``metric``+``direction`` → objectives; ``limit``+``value`` →
limits), so the markdown can be reformatted freely as long as those two
tables survive.
"""

from __future__ import annotations

import json
import re
import sys


def _rows(md: str, must_have: set[str]) -> list[list[str]]:
    """Return the data rows of the first pipe-table whose header cells are a
    superset of ``must_have`` (lowercased). Separator rows are dropped."""
    header_cols: list[str] | None = None
    out: list[list[str]] = []
    for line in md.splitlines():
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            header_cols = None
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if set(re.sub(r"[-:| ]", "", "".join(cells))) == set():
            continue  # separator row like |---|---|
        low = [c.lower() for c in cells]
        if header_cols is None:
            if must_have.issubset(set(low)):
                header_cols = low
            continue
        out.append(cells)
    return out


def _num(s: str):
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def _section_body(md: str, header_re: str) -> str:
    """Text lines under a ``## <header_re>`` heading, up to the next ``##``.
    Blockquote (>) and blank lines dropped."""
    lines = md.splitlines()
    body: list[str] = []
    capturing = False
    for line in lines:
        if line.startswith("## "):
            capturing = bool(re.search(header_re, line, re.IGNORECASE))
            continue
        if capturing:
            s = line.strip()
            if s and not s.startswith(">"):
                body.append(s)
    return " ".join(body).strip()


def _principles(md: str) -> list[str]:
    """Numbered items under the Principles heading, one string each (bold
    markers stripped, wrapped continuation lines merged)."""
    lines = md.splitlines()
    capturing = False
    items: list[str] = []
    cur: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if capturing and cur:
                items.append(" ".join(cur))
                cur = []
            capturing = bool(re.search(r"principle", line, re.IGNORECASE))
            continue
        if not capturing:
            continue
        s = line.strip()
        if re.match(r"^\d+\.\s+", s):
            if cur:
                items.append(" ".join(cur))
            cur = [re.sub(r"^\d+\.\s+", "", s)]
        elif s and cur:
            cur.append(s)
        elif not s and cur:
            items.append(" ".join(cur))
            cur = []
    if cur:
        items.append(" ".join(cur))
    return [re.sub(r"\*\*", "", it).strip() for it in items]


def lint(md: str) -> dict:
    """Parse charter markdown into the charter.json dict."""
    objectives = []
    for r in _rows(md, {"metric", "direction"}):
        # columns: id, objective, weight, metric, target, direction
        if len(r) < 6 or r[0].lower() in ("id",):
            continue
        objectives.append({
            "id": r[0],
            "title": r[1],
            "weight": _num(r[2]) or 0.0,
            "metric": r[3],
            "target": _num(r[4]),
            "direction": (r[5].lower() or "up"),
        })
    limits = {}
    for r in _rows(md, {"limit", "value"}):
        if len(r) < 2 or r[0].lower() == "limit":
            continue
        limits[r[0]] = _num(r[1])
    return {
        "version": 1,
        "identity": _section_body(md, r"identity"),
        "principles": _principles(md),
        "objectives": objectives,
        "limits": limits,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    md = open(sys.argv[1], encoding="utf-8").read()
    doc = lint(md)
    out = json.dumps(doc, indent=2)
    if len(sys.argv) >= 3:
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(f"wrote {sys.argv[2]}: {len(doc['objectives'])} objectives, "
              f"{len(doc['limits'])} limits")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
