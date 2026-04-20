"""Unit tests for agi.common.sensei_note.

The verify-before-publish invariant is only as strong as the check
that separates verified from draft notes. These tests lock in:

- non-empty verified_by in frontmatter → verified
- absent / empty / whitespace-only verified_by → NOT verified
- verified_by elsewhere in body (not frontmatter) → NOT verified
- missing frontmatter → NOT verified
- missing file → None from read_if_verified
"""

from __future__ import annotations

from pathlib import Path

from agi.common.sensei_note import is_verified, read_if_verified


def _frontmatter(verified_by: str | None, body: str = "# title\n") -> str:
    if verified_by is None:
        fm = "---\ntype: sensei_note\ntask: 1\n---\n"
    else:
        fm = f"---\ntype: sensei_note\ntask: 1\nverified_by: {verified_by}\n---\n"
    return fm + body


def test_verified_by_present_is_verified():
    assert is_verified(_frontmatter("reference_implementation (train 3/3, test 1/1)"))


def test_verified_by_absent_is_not_verified():
    assert not is_verified(_frontmatter(None))


def test_verified_by_empty_is_not_verified():
    assert not is_verified(_frontmatter(""))


def test_verified_by_whitespace_is_not_verified():
    assert not is_verified(_frontmatter("   "))


def test_verified_by_quoted_is_verified():
    assert is_verified(_frontmatter('"reference_implementation"'))
    assert is_verified(_frontmatter("'reference_implementation'"))


def test_verified_by_empty_quoted_is_not_verified():
    assert not is_verified(_frontmatter('""'))
    assert not is_verified(_frontmatter("''"))


def test_no_frontmatter_is_not_verified():
    assert not is_verified("# some note\n\nverified_by: yes\n")


def test_verified_by_in_body_only_does_not_count():
    text = (
        "---\ntype: sensei_note\n---\n\n"
        "# Title\n\nverified_by: this is in the body\n"
    )
    assert not is_verified(text)


def test_read_if_verified_returns_text_for_verified(tmp_path: Path):
    p = tmp_path / "sensei_task_001.md"
    p.write_text(_frontmatter("reference_implementation (train 3/3)"))
    assert read_if_verified(p) is not None


def test_read_if_verified_returns_none_for_draft(tmp_path: Path):
    p = tmp_path / "sensei_task_001.md"
    p.write_text(_frontmatter(None))
    assert read_if_verified(p) is None


def test_read_if_verified_returns_none_for_missing_file(tmp_path: Path):
    assert read_if_verified(tmp_path / "does_not_exist.md") is None
