# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for the pregen roster + pregen handout renderer.

Tests are data-driven: every pregen in :data:`ROSTER` is validated
for CoC 7e sanity (characteristics in range, skills in range,
required prose fields present, no spoiler terms). The handout
renderer + CSV seed + generator-integration are tested against the
full roster.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agi.primer.artemis.artifacts import (
    ROSTER,
    Pregen,
    Skill,
    by_id,
    render_pregen_markdown,
    roster_to_csv,
    write_pregen_handouts,
)
from agi.primer.artemis.artifacts.generator import discover_handouts
from agi.primer.artemis.artifacts.pregens import (
    ARLO_VANCE,
    ASTA_NORDQUIST,
    ERIK_SULLIVAN,
    IMOGEN_ROTH,
    SAOIRSE_KELLEHER,
)

# ─────────────────────────────────────────────────────────────────
# Roster shape
# ─────────────────────────────────────────────────────────────────


def test_roster_matches_hud_placeholder_ids() -> None:
    # Single source of truth: these ids match the HUD PLACEHOLDER_CHARS
    # in table.js. If the HUD ever adds a role, this fails until the
    # pregen is written.
    expected = {"imogen", "sully", "asta", "arlo", "saoirse"}
    assert {p.id for p in ROSTER} == expected


def test_roster_is_ordered_canonical() -> None:
    # Order matters for the handout batch: the Expedition Lead goes
    # first, support crew after. This is an explicit design choice —
    # if we reorder, it should be deliberate.
    assert [p.id for p in ROSTER] == [
        "imogen",
        "sully",
        "asta",
        "arlo",
        "saoirse",
    ]


@pytest.mark.parametrize("pregen", ROSTER)
def test_pregen_characteristics_in_coc_range(pregen: Pregen) -> None:
    # CoC 7e characteristics are percentile — 15..90 is typical;
    # allow 10..95 as absolute bounds.
    for attr_name in (
        "str_",
        "con",
        "siz",
        "dex",
        "app",
        "int_",
        "pow_",
        "edu",
    ):
        val = getattr(pregen, attr_name)
        assert 10 <= val <= 95, f"{pregen.id}.{attr_name}={val} outside CoC 10..95"


@pytest.mark.parametrize("pregen", ROSTER)
def test_pregen_derived_stats_consistent(pregen: Pregen) -> None:
    # HP max ≈ (CON + SIZ)/10 in CoC 7e — we allow ±2 slack for
    # race/age modifiers but not wild divergence.
    expected_hp = (pregen.con + pregen.siz) // 10
    assert (
        abs(pregen.hp_max - expected_hp) <= 3
    ), f"{pregen.id} HP {pregen.hp_max} vs expected {expected_hp}"
    # Magic Points = POW / 5.
    assert abs(pregen.mp_max - pregen.pow_ // 5) <= 1
    # Starting SAN must be <= POW (canonical CoC 7e rule).
    assert pregen.san_start <= pregen.pow_


@pytest.mark.parametrize("pregen", ROSTER)
def test_pregen_skills_in_range(pregen: Pregen) -> None:
    assert (
        len(pregen.skills) >= 10
    ), f"{pregen.id} has only {len(pregen.skills)} skills — should be 10+"
    for skill in pregen.skills:
        assert isinstance(skill, Skill)
        assert (
            1 <= skill.value <= 95
        ), f"{pregen.id} skill {skill.name}={skill.value} out of range"


@pytest.mark.parametrize("pregen", ROSTER)
def test_pregen_prose_fields_nonempty(pregen: Pregen) -> None:
    assert len(pregen.background) >= 100, "background too short"
    assert len(pregen.hook) >= 50, "hook too short"
    assert pregen.portrait_note, "portrait note missing"
    assert pregen.gear, "no gear listed"


@pytest.mark.parametrize("pregen", ROSTER)
def test_pregen_does_not_leak_spoilers(pregen: Pregen) -> None:
    # Pre-session handouts must not name the antagonists or plot
    # locations that only the Keeper knows. If the campaign expands
    # the safe-to-publish surface, update this list.
    forbidden = [
        "mi-go",
        "Mi-go",
        "Mi‑go",
        "fungi",
        "Unborn",  # cult name
        "bailey",  # prior survey survivor (Keeper-only NPC)
        "vault",  # the vault reveal is a Session 1 thing
    ]
    full_text = (
        pregen.background + "\n" + pregen.hook + "\n" + pregen.portrait_note
    ).lower()
    hits = [w for w in forbidden if w.lower() in full_text]
    assert not hits, f"{pregen.id} handout leaks spoilers: {hits}"


def test_by_id_hits_and_misses() -> None:
    assert by_id("imogen") is IMOGEN_ROTH
    assert by_id("arlo") is ARLO_VANCE
    assert by_id("sully") is ERIK_SULLIVAN
    assert by_id("asta") is ASTA_NORDQUIST
    assert by_id("saoirse") is SAOIRSE_KELLEHER
    assert by_id("unknown") is None


# ─────────────────────────────────────────────────────────────────
# Skill helper
# ─────────────────────────────────────────────────────────────────


def test_skill_half_and_fifth_rounded_down() -> None:
    s = Skill(name="Dodge", value=77)
    assert s.half == 38
    assert s.fifth == 15
    s2 = Skill(name="Edge", value=10)
    assert s2.half == 5
    assert s2.fifth == 2


# ─────────────────────────────────────────────────────────────────
# Handout rendering
# ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pregen", ROSTER)
def test_render_pregen_markdown_has_expected_structure(pregen: Pregen) -> None:
    md = render_pregen_markdown(pregen)
    # Front matter.
    assert md.startswith("---\n")
    assert f"slug: pregen_{pregen.id}" in md
    assert "audience: all" in md
    assert "secrets: none" in md
    # Content sections.
    for heading in (
        "# " + pregen.name,
        "## Characteristics",
        "## Skills",
        "## Equipment",
        "## Background",
        "## Hook",
    ):
        assert heading in md, f"{pregen.id} handout missing: {heading!r}"
    # Every skill appears with its full value.
    for s in pregen.skills:
        assert s.name in md
        assert f"| {s.value} |" in md


def test_write_pregen_handouts_round_trip(tmp_path: Path) -> None:
    written = write_pregen_handouts(tmp_path)
    assert len(written) == len(ROSTER)
    for path, pregen in zip(written, ROSTER, strict=True):
        assert path.is_file()
        assert path.name == f"pregen_{pregen.id}.md"
        content = path.read_text(encoding="utf-8")
        assert pregen.name in content


# ─────────────────────────────────────────────────────────────────
# CSV seed for the Google Sheet
# ─────────────────────────────────────────────────────────────────


def test_roster_to_csv_has_header_and_rows() -> None:
    csv = roster_to_csv()
    lines = csv.strip().split("\n")
    assert len(lines) == len(ROSTER) + 1  # header + rows
    header = lines[0].split(",")
    for col in ("id", "name", "role", "san", "hp", "hp_max", "status"):
        assert col in header
    # Every pregen id must show up in the body.
    body = "\n".join(lines[1:])
    for p in ROSTER:
        assert p.id in body


def test_roster_to_csv_is_parseable_by_sheets_parser() -> None:
    # The whole point of the CSV seed is that it drops straight into
    # the sheets-poller — parse round-trip proves it.
    from agi.primer.artemis.sheets.parser import parse_csv

    csv = roster_to_csv()
    rows = parse_csv(csv)
    assert {r.id for r in rows} == {p.id for p in ROSTER}
    imogen_row = next(r for r in rows if r.id == "imogen")
    assert imogen_row.hp == IMOGEN_ROTH.hp_max
    assert imogen_row.san == IMOGEN_ROTH.san_start
    assert imogen_row.status == IMOGEN_ROTH.starting_status


# ─────────────────────────────────────────────────────────────────
# Generator integration — discovery picks up the pregens
# ─────────────────────────────────────────────────────────────────


def test_discover_includes_pregens_by_default() -> None:
    discovered = discover_handouts()
    slugs = {h.slug for h in discovered}
    for p in ROSTER:
        assert f"pregen_{p.id}" in slugs, f"pregen_{p.id} not found in discovery"


def test_discover_can_exclude_pregens(tmp_path: Path) -> None:
    # Empty source dir + include_pregens=False → empty list.
    out = discover_handouts(tmp_path, include_pregens=False)
    assert out == []
