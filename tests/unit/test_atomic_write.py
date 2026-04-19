"""Unit tests for agi.common.atomic_write.

The whole point of atomic_write_text is that the destination is never
observed in a half-written state. These tests verify:
- normal write produces the expected contents
- intermediate tempfile is cleaned up on failure
- no stray tempfiles are left behind after success
- a failure in the middle of writing leaves the old contents intact
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agi.common.atomic_write import atomic_write_text


def test_writes_expected_contents(tmp_path: Path):
    p = tmp_path / "state.json"
    atomic_write_text(p, '{"x": 1}')
    assert p.read_text() == '{"x": 1}'


def test_creates_parent_directory(tmp_path: Path):
    p = tmp_path / "nested" / "dir" / "state.json"
    atomic_write_text(p, "hello")
    assert p.read_text() == "hello"


def test_overwrites_existing_file(tmp_path: Path):
    p = tmp_path / "state.json"
    p.write_text("old")
    atomic_write_text(p, "new")
    assert p.read_text() == "new"


def test_no_tempfile_left_behind_on_success(tmp_path: Path):
    p = tmp_path / "state.json"
    atomic_write_text(p, "done")
    # Only the target file — no *.tmp siblings
    siblings = list(tmp_path.iterdir())
    assert siblings == [p]


def test_existing_file_preserved_on_failure(tmp_path: Path):
    p = tmp_path / "state.json"
    p.write_text("original")

    # Simulate os.replace failing after tempfile is written
    with patch("agi.common.atomic_write.os.replace", side_effect=OSError("boom")):
        with pytest.raises(OSError):
            atomic_write_text(p, "new")

    # Original contents untouched
    assert p.read_text() == "original"
    # Tempfile cleaned up
    siblings = [f for f in tmp_path.iterdir() if f.name != p.name]
    assert siblings == []


def test_write_failure_cleans_tempfile(tmp_path: Path):
    p = tmp_path / "state.json"

    # Make the write itself fail
    real_fdopen = os.fdopen

    def fake_fdopen(fd, *a, **kw):
        f = real_fdopen(fd, *a, **kw)

        class Wrapper:
            def __enter__(self_):
                return self_

            def __exit__(self_, *exc):
                f.close()

            def write(self_, _):
                raise OSError("disk full")

            def flush(self_):
                pass

            def fileno(self_):
                return f.fileno()

        return Wrapper()

    with patch("agi.common.atomic_write.os.fdopen", fake_fdopen):
        with pytest.raises(OSError):
            atomic_write_text(p, "x")

    # No file created, no tempfile left
    assert list(tmp_path.iterdir()) == []


def test_fsync_can_be_disabled(tmp_path: Path):
    p = tmp_path / "state.json"
    atomic_write_text(p, "fast-path", fsync=False)
    assert p.read_text() == "fast-path"
