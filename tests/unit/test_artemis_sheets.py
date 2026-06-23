# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for the ARTEMIS sheets package.

Parser and poller have zero real I/O — transport is injected — so
these tests run without network or NATS.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agi.primer.artemis.sheets.parser import (
    HUD_FIELDS,
    CharacterRow,
    diff_rows,
    parse_csv,
    row_for_hud,
)
from agi.primer.artemis.sheets.poller import (
    SUBJECT_DIFF_PREFIX,
    SUBJECT_SNAPSHOT_PREFIX,
    SheetBinding,
    SheetsPoller,
    _parse_bindings,
    build_poller_from_env,
)

# ─────────────────────────────────────────────────────────────────
# parser
# ─────────────────────────────────────────────────────────────────


MINIMAL_CSV = """\
id,name,role,san,san_max,hp,hp_max,luck,luck_max,mp,mp_max,status
imogen,IMOGEN ROTH,Expedition Lead,70,75,11,11,55,70,14,14,ok
arlo,ARLO VANCE,Surface Ops,55,55,15,15,45,70,11,11,shaken
"""


# CSV rows must fit on one line; lint this literal as-is.
FULL_CSV = (
    "id,name,role,occupation,age,str,con,siz,dex,app,int,pow,edu,"
    "san,san_max,hp,hp_max,luck,luck_max,mp,mp_max,"
    "status,notes,skills_json,equipment_json,phobia\n"
    "sully,ERIK SULLIVAN,Chief Engineer,Rocinante veteran,42,"
    "65,70,60,60,50,75,60,80,"
    "62,62,14,14,50,70,12,12,ok,,"
    '"[{""n"":""Mechanical Repair"",""v"":80}]",'
    '"[{""n"":""Vacuum Suit""}]",'
    "claustrophobia\n"
)


def test_parse_minimal_csv_happy_path() -> None:
    rows = parse_csv(MINIMAL_CSV)
    assert [r.id for r in rows] == ["imogen", "arlo"]
    r0 = rows[0].to_dict()
    assert r0["name"] == "IMOGEN ROTH"
    assert r0["role"] == "Expedition Lead"
    assert r0["san"] == 70 and r0["san_max"] == 75
    assert r0["status"] == "ok"


def test_parse_full_csv_carries_skills_and_extra() -> None:
    rows = parse_csv(FULL_CSV)
    assert len(rows) == 1
    r = rows[0]
    assert r.occupation == "Rocinante veteran"
    assert r.age == 42
    assert r.str_ == 65 and r.int_ == 75 and r.pow_ == 60
    assert r.skills == [{"n": "Mechanical Repair", "v": 80}]
    assert r.equipment == [{"n": "Vacuum Suit"}]
    # Unknown column preserved on `extra`.
    assert r.extra.get("phobia") == "claustrophobia"


def test_parse_skips_blank_id_rows() -> None:
    csv_text = "id,name\n,ghost\nimogen,IMOGEN\n"
    rows = parse_csv(csv_text)
    assert [r.id for r in rows] == ["imogen"]


def test_parse_handles_float_ints() -> None:
    csv_text = "id,name,hp,hp_max\nfoo,Foo,11.0,11.0\n"
    rows = parse_csv(csv_text)
    assert rows[0].hp == 11 and rows[0].hp_max == 11


def test_parse_invalid_status_falls_back_to_ok() -> None:
    csv_text = "id,status\nfoo,RADIOACTIVE\n"
    assert parse_csv(csv_text)[0].status == "ok"


def test_parse_bad_json_cell_is_dropped_silently() -> None:
    csv_text = "id,skills_json\nfoo,not-json-{\n"
    assert parse_csv(csv_text)[0].skills == []


def test_to_dict_drops_empty_optionals() -> None:
    r = CharacterRow(id="foo", name="Foo")
    d = r.to_dict()
    # Unfilled characteristics should not be present as `null`
    # in the payload — cleaner JSON on the wire.
    assert "str" not in d
    assert "san" not in d
    assert d["id"] == "foo"


def test_row_for_hud_projects_only_hud_fields() -> None:
    r = parse_csv(FULL_CSV)[0]
    hud = row_for_hud(r)
    assert set(hud).issubset(set(HUD_FIELDS))
    # Non-HUD stats like `occupation`, `skills` are NOT in the HUD
    # projection — otherwise the table would carry every sheet
    # column over DataChannel for every diff.
    assert "occupation" not in hud
    assert "skills" not in hud
    assert hud["id"] == "sully"


def test_diff_rows_reports_only_changed_hud_projections() -> None:
    old = parse_csv(MINIMAL_CSV)
    # Change only a non-HUD field (notes) on one row.
    new_csv = MINIMAL_CSV.replace("ok", "ok")  # identical
    unchanged = diff_rows(old, parse_csv(new_csv))
    assert unchanged == []

    # Change a HUD field (san dropped 70→64 for imogen).
    mutated = MINIMAL_CSV.replace(
        "imogen,IMOGEN ROTH,Expedition Lead,70,75",
        "imogen,IMOGEN ROTH,Expedition Lead,64,75",
    )
    changed = diff_rows(old, parse_csv(mutated))
    ids = [r["id"] for r in changed]
    assert ids == ["imogen"]
    assert changed[0]["san"] == 64


def test_diff_rows_accepts_dict_inputs() -> None:
    old = [{"id": "imogen", "san": 70, "san_max": 75, "name": "I"}]
    new = [{"id": "imogen", "san": 60, "san_max": 75, "name": "I"}]
    changed = diff_rows(old, new)
    assert changed and changed[0]["san"] == 60


def test_column_alias_case_insensitive_and_space_tolerant() -> None:
    csv_text = "ID,Name, HP , HP_Max\nfoo,Foo,3,10\n"
    r = parse_csv(csv_text)[0]
    assert r.id == "foo" and r.hp == 3 and r.hp_max == 10


# ─────────────────────────────────────────────────────────────────
# poller
# ─────────────────────────────────────────────────────────────────


class _FakePublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, subject: str, payload: bytes) -> None:
        self.calls.append((subject, json.loads(payload)))


def _fetcher_from_sequence(*texts: str):
    it = iter(texts)

    async def fetch(url: str) -> str:
        return next(it)

    return fetch


def test_poller_requires_positive_interval() -> None:
    with pytest.raises(ValueError):
        SheetsPoller(
            bindings=[SheetBinding("characters", "csv://")],
            fetcher=_fetcher_from_sequence(""),
            publisher=_FakePublisher(),
            poll_interval_s=0,
        )


def test_poller_first_tick_publishes_snapshot_and_hud() -> None:
    pub = _FakePublisher()
    poller = SheetsPoller(
        bindings=[SheetBinding("characters", "csv://ignored")],
        fetcher=_fetcher_from_sequence(MINIMAL_CSV),
        publisher=pub,
        poll_interval_s=30.0,
    )
    asyncio.run(poller.tick_once())
    subjects = [c[0] for c in pub.calls]
    assert f"{SUBJECT_SNAPSHOT_PREFIX}.characters" in subjects
    assert f"{SUBJECT_DIFF_PREFIX}.characters" in subjects
    # Snapshot carries all-rows with every column the sheet had.
    snap = next(p for s, p in pub.calls if "snapshot" in s)
    assert len(snap["rows"]) == 2
    assert snap["rows"][0]["id"] == "imogen"
    # HUD carries only HUD fields.
    hud = next(p for s, p in pub.calls if s == f"{SUBJECT_DIFF_PREFIX}.characters")
    assert set(hud["rows"][0].keys()).issubset(set(HUD_FIELDS))


def test_poller_subsequent_tick_silent_when_nothing_changed() -> None:
    pub = _FakePublisher()
    poller = SheetsPoller(
        bindings=[SheetBinding("characters", "csv://")],
        fetcher=_fetcher_from_sequence(MINIMAL_CSV, MINIMAL_CSV),
        publisher=pub,
        poll_interval_s=30.0,
    )
    asyncio.run(poller.tick_once())
    before = len(pub.calls)
    asyncio.run(poller.tick_once())
    # Second tick — snapshot already sent, nothing changed, so no new
    # messages go out. Keeps the NATS stream quiet.
    assert len(pub.calls) == before


def test_poller_second_tick_publishes_only_diff() -> None:
    mutated = MINIMAL_CSV.replace("70,75", "64,75")  # imogen SAN 70→64
    pub = _FakePublisher()
    poller = SheetsPoller(
        bindings=[SheetBinding("characters", "csv://")],
        fetcher=_fetcher_from_sequence(MINIMAL_CSV, mutated),
        publisher=pub,
        poll_interval_s=30.0,
    )
    asyncio.run(poller.tick_once())
    n0 = len(pub.calls)
    asyncio.run(poller.tick_once())
    new_calls = pub.calls[n0:]
    assert len(new_calls) == 1
    subj, payload = new_calls[0]
    assert subj == f"{SUBJECT_DIFF_PREFIX}.characters"
    assert [r["id"] for r in payload["rows"]] == ["imogen"]


def test_poller_fetcher_exception_is_swallowed() -> None:
    async def boom(url: str) -> str:
        raise ConnectionError("network down")

    pub = _FakePublisher()
    poller = SheetsPoller(
        bindings=[SheetBinding("characters", "csv://")],
        fetcher=boom,
        publisher=pub,
        poll_interval_s=30.0,
    )
    counts = asyncio.run(poller.tick_once())
    # Failure keeps the service alive for the next tick; no publishes.
    assert counts == {"characters": 0}
    assert pub.calls == []


def test_poller_two_sheets_tick_independently() -> None:
    pub = _FakePublisher()
    poller = SheetsPoller(
        bindings=[
            SheetBinding("characters", "csv://c"),
            SheetBinding("npcs", "csv://n"),
        ],
        fetcher=_fetcher_from_sequence(MINIMAL_CSV, "id,name\nmigo,Mi-Go\n"),
        publisher=pub,
        poll_interval_s=30.0,
    )
    asyncio.run(poller.tick_once())
    subjects = {c[0] for c in pub.calls}
    assert f"{SUBJECT_DIFF_PREFIX}.characters" in subjects
    assert f"{SUBJECT_DIFF_PREFIX}.npcs" in subjects


def test_parse_bindings_semicolon_format() -> None:
    bindings = _parse_bindings("characters=https://a;npcs=https://b")
    assert [(b.name, b.url) for b in bindings] == [
        ("characters", "https://a"),
        ("npcs", "https://b"),
    ]


def test_parse_bindings_tolerates_whitespace_and_drops_bad() -> None:
    bindings = _parse_bindings(" characters = https://a ; nope ; = bad ; npcs=b ")
    assert [(b.name, b.url) for b in bindings] == [
        ("characters", "https://a"),
        ("npcs", "b"),
    ]


def test_build_poller_from_env_empty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARTEMIS_SHEET_URLS", raising=False)
    with pytest.raises(RuntimeError):
        build_poller_from_env()


def test_build_poller_from_env_injected_transports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTEMIS_SHEET_URLS", "characters=https://example/csv")
    monkeypatch.setenv("ARTEMIS_SHEET_POLL_S", "5")

    async def fetch(url: str) -> str:
        return MINIMAL_CSV

    pub = _FakePublisher()
    poller = build_poller_from_env(fetcher=fetch, publisher=pub)
    assert poller.poll_interval_s == 5.0
    asyncio.run(poller.tick_once())
    assert any("snapshot.characters" in s for s, _ in pub.calls)
