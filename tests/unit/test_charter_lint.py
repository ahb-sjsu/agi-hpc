# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for scripts/charter_lint.py (charter.md → charter.json projection)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import charter_lint  # noqa: E402

_MD = """
# Erebus — Charter

## 1. Identity & mission

I am Erebus, a test mind.

## 2. Principles

1. **Verify before I claim.** No fabrication.
2. **Do not deceive.** Ever.

## 3. Standing objectives

| id | objective | weight | metric | target | direction |
|----|-----------|--------|--------|--------|-----------|
| obj-a | Raise solve rate | 0.35 | arc_solved | 140 | up |
| obj-b | Clear backlog | 0.35 | arc_stuck | 80 | down |
| obj-c | Aspirational | 0.1 | conversations | | up |

## 4. Constraints

| limit | value |
|-------|-------|
| max_dispatch_per_cycle | 2 |
| max_dispatch_per_day | 12 |
| thermal_c | 82 |
"""


def test_lint_objectives():
    doc = charter_lint.lint(_MD)
    assert len(doc["objectives"]) == 3
    a = doc["objectives"][0]
    assert a["id"] == "obj-a" and a["metric"] == "arc_solved"
    assert a["weight"] == 0.35 and a["target"] == 140 and a["direction"] == "up"
    assert doc["objectives"][1]["direction"] == "down"
    # aspirational row: empty target parses to None
    assert doc["objectives"][2]["target"] is None


def test_lint_limits_and_prose():
    doc = charter_lint.lint(_MD)
    assert doc["limits"] == {
        "max_dispatch_per_cycle": 2, "max_dispatch_per_day": 12, "thermal_c": 82,
    }
    assert "Erebus" in doc["identity"]
    assert len(doc["principles"]) == 2
    assert "**" not in doc["principles"][0]  # bold markers stripped


def test_lint_roundtrips_through_charter_loader(tmp_path):
    import json

    from agi.director.charter import Charter

    p = tmp_path / "charter.json"
    p.write_text(json.dumps(charter_lint.lint(_MD)))
    c = Charter.load(p)
    assert c is not None
    ranked = c.ranked_gaps({"tasks_solved": 107, "tasks_stuck": 112})
    active = [o.id for o, cur, prio in ranked if prio is not None]
    assert set(active) == {"obj-a", "obj-b"}  # only measured objectives rank
