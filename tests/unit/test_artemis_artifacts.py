# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for the handout generator.

pandoc itself is never invoked — every test injects a fake runner.
The Session 0 briefing shipped in this PR is also inspected to
confirm its front matter parses and promises no secrets.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agi.primer.artemis.artifacts import (
    HANDOUT_SOURCE_DIR,
    HandoutError,
    HandoutMeta,
    discover_handouts,
    render_handout,
)
from agi.primer.artemis.artifacts.generator import _parse_front_matter

# ─────────────────────────────────────────────────────────────────
# front matter
# ─────────────────────────────────────────────────────────────────


def _write_handout(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_parse_front_matter_happy_path(tmp_path: Path) -> None:
    body = (
        '---\nslug: demo\ntitle: "A Title"\n'
        "audience: all\nsecrets: none\n---\n\nbody\n"
    )
    p = _write_handout(tmp_path, "ok.md", body)
    meta = _parse_front_matter(p)
    assert meta.slug == "demo"
    assert meta.title == "A Title"
    assert meta.audience == "all"
    assert meta.secrets == "none"
    assert meta.path == p


def test_parse_front_matter_defaults(tmp_path: Path) -> None:
    p = _write_handout(
        tmp_path,
        "ok.md",
        "---\nslug: demo\ntitle: A Title\n---\nbody\n",
    )
    meta = _parse_front_matter(p)
    assert meta.audience == "all"
    assert meta.secrets == "none"


def test_parse_front_matter_strips_surrounding_quotes(tmp_path: Path) -> None:
    p = _write_handout(
        tmp_path,
        "ok.md",
        "---\nslug: 'demo'\ntitle: \"Quoted Title\"\n---\n",
    )
    meta = _parse_front_matter(p)
    assert meta.slug == "demo"
    assert meta.title == "Quoted Title"


def test_parse_front_matter_missing_block(tmp_path: Path) -> None:
    p = _write_handout(tmp_path, "no_fm.md", "# no front matter\n")
    with pytest.raises(HandoutError):
        _parse_front_matter(p)


def test_parse_front_matter_missing_slug(tmp_path: Path) -> None:
    p = _write_handout(
        tmp_path,
        "no_slug.md",
        '---\ntitle: "A Title"\n---\n',
    )
    with pytest.raises(HandoutError):
        _parse_front_matter(p)


def test_parse_front_matter_missing_title(tmp_path: Path) -> None:
    p = _write_handout(tmp_path, "no_title.md", "---\nslug: demo\n---\n")
    with pytest.raises(HandoutError):
        _parse_front_matter(p)


# ─────────────────────────────────────────────────────────────────
# discovery
# ─────────────────────────────────────────────────────────────────


def test_discover_ignores_readme_and_underscore(tmp_path: Path) -> None:
    _write_handout(
        tmp_path,
        "session0.md",
        '---\nslug: s0\ntitle: "A"\n---\n',
    )
    _write_handout(
        tmp_path,
        "README.md",
        "# not a handout\n",
    )
    _write_handout(
        tmp_path,
        "_draft.md",
        '---\nslug: draft\ntitle: "A"\n---\n',
    )
    discovered = discover_handouts(tmp_path, include_pregens=False)
    slugs = [h.slug for h in discovered]
    assert slugs == ["s0"]


def test_discover_skips_bad_front_matter_logs_and_continues(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write_handout(tmp_path, "good.md", "---\nslug: g\ntitle: G\n---\n")
    _write_handout(tmp_path, "bad.md", "no front matter here\n")
    with caplog.at_level("WARNING"):
        got = discover_handouts(tmp_path, include_pregens=False)
    assert [h.slug for h in got] == ["g"]
    assert any("bad.md" in msg for msg in caplog.messages)


def test_discover_returns_empty_when_dir_missing(tmp_path: Path) -> None:
    assert discover_handouts(tmp_path / "nope", include_pregens=False) == []


# ─────────────────────────────────────────────────────────────────
# rendering (fake pandoc runner)
# ─────────────────────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, returncode: int = 0, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stderr = stderr


def test_render_invokes_pandoc_with_expected_args(tmp_path: Path) -> None:
    src = _write_handout(
        tmp_path,
        "demo.md",
        '---\nslug: demo\ntitle: "Demo Title"\n---\n\nbody\n',
    )
    meta = _parse_front_matter(src)
    out_dir = tmp_path / "out"
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str]) -> _FakeResult:
        calls.append(cmd)
        # Simulate pandoc by just touching the output file.
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"%PDF-fake\n")
        return _FakeResult()

    pdf = render_handout(meta, out_dir, runner=fake_runner)
    assert pdf == out_dir / "demo.pdf"
    assert pdf.is_file()
    assert calls, "runner not called"
    cmd = calls[0]
    assert cmd[0] == "pandoc"
    assert str(src) in cmd
    assert "-o" in cmd and str(pdf) in cmd
    # Title must be passed to pandoc metadata.
    assert "--metadata" in cmd
    assert any("title=Demo Title" in part for part in cmd)


def test_render_raises_on_pandoc_nonzero(tmp_path: Path) -> None:
    src = _write_handout(
        tmp_path,
        "demo.md",
        '---\nslug: demo\ntitle: "D"\n---\nbody\n',
    )
    meta = _parse_front_matter(src)

    def fake_runner(cmd: list[str]) -> _FakeResult:
        return _FakeResult(returncode=3, stderr=b"xelatex not found")

    with pytest.raises(HandoutError, match="pandoc failed"):
        render_handout(meta, tmp_path / "out", runner=fake_runner)


def test_render_raises_when_source_file_missing(tmp_path: Path) -> None:
    meta = HandoutMeta(slug="demo", title="D", path=tmp_path / "gone.md")
    with pytest.raises(HandoutError):
        render_handout(meta, tmp_path / "out")


def test_render_raises_when_pandoc_claims_success_but_file_missing(
    tmp_path: Path,
) -> None:
    src = _write_handout(
        tmp_path,
        "demo.md",
        '---\nslug: demo\ntitle: "D"\n---\nbody\n',
    )
    meta = _parse_front_matter(src)

    def fake_runner(cmd: list[str]) -> _FakeResult:
        # returncode=0 but the file is never created
        return _FakeResult()

    with pytest.raises(HandoutError, match="missing"):
        render_handout(meta, tmp_path / "out", runner=fake_runner)


def test_render_missing_pandoc_is_raised_as_handout_error(tmp_path: Path) -> None:
    src = _write_handout(
        tmp_path,
        "demo.md",
        '---\nslug: demo\ntitle: "D"\n---\nbody\n',
    )
    meta = _parse_front_matter(src)
    # No runner, and pandoc='no-such-tool' guarantees shutil.which returns None.
    with pytest.raises(HandoutError, match="not found on PATH"):
        render_handout(meta, tmp_path / "out", pandoc="no-such-tool")


# ─────────────────────────────────────────────────────────────────
# Shipped Session 0 handout — content invariants
# ─────────────────────────────────────────────────────────────────


def test_session0_briefing_is_discoverable_and_no_secrets() -> None:
    handouts = discover_handouts(HANDOUT_SOURCE_DIR)
    slugs = {h.slug for h in handouts}
    assert (
        "session0_briefing" in slugs
    ), "Session 0 handout must be discoverable from the default source dir"
    s0 = next(h for h in handouts if h.slug == "session0_briefing")
    # Contract: pre-session 0 handout is shared with everyone and
    # carries no secrets — players should be able to read it freely.
    assert s0.audience == "all"
    assert s0.secrets == "none"


def test_session0_briefing_has_required_sections() -> None:
    # Catch accidental content regressions — the handout must still
    # cover the key pre-session topics.
    body = (HANDOUT_SOURCE_DIR / "session0_briefing.md").read_text(
        encoding="utf-8",
    )
    for required in [
        "Mao",  # employer
        "Halyard",  # ship
        "Nithon",  # mission target
        "ARTEMIS",  # the AI
        "Session Zero",  # what to expect
        "X-card",  # safety tools
        "Lines & Veils",
    ]:
        assert required in body, f"section missing: {required!r}"


def test_session0_briefing_does_not_leak_secrets() -> None:
    # Explicit guard against spoiler terms ending up in the
    # pre-session-0 handout. Update this list if setting-lore
    # becomes safe to publish; until then the handout stays clean.
    body = (HANDOUT_SOURCE_DIR / "session0_briefing.md").read_text(
        encoding="utf-8",
    )
    forbidden = [
        "mi-go",
        "Mi-go",
        "Mi‑go",  # the antagonists
        "fungi",  # their in-fiction alt-name
        "Unborn",  # cult name — spoilers
        "bailey",  # prior-survey NPC, spoilers
        "SAN loss",  # mechanical spoilers
        "Keeper:",  # Keeper-only marker
    ]
    hits = [term for term in forbidden if term in body]
    assert not hits, f"Session 0 handout contains spoiler terms: {hits}"
