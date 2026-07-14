# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for Erebus's staged self-tooling facility.

Covers the safety-critical behavior: static-safety scanning, sandboxed
(never in-process) execution, stage→ack→promote separation, the kill-
switch, and the audit trail. Candidates use no imports so the tests need
neither numpy nor onnxruntime, and all filesystem state is injected via
tmp_path.
"""

from __future__ import annotations

import json

import pytest

from agi.autonomous import erebus_compiler_tools as ect

# ── candidate sources ────────────────────────────────────────────

SAFE = (
    "def detect_flip(examples):\n"
    "    '''Flip the grid horizontally when input and output mirror.'''\n"
    "    return True\n"
    "def compile_flip():\n"
    "    return None\n"
)

UNSAFE_OS = "import os\ndef compile_x():\n    os.system('echo pwned')\n"
UNSAFE_EVAL = "def compile_x():\n    return eval('1+1')\n"
UNSAFE_SOCKET = "import socket\ndef compile_x():\n    return socket.socket()\n"
UNSAFE_DUNDER = "def compile_x():\n    return ().__class__.__bases__\n"
BROKEN = "def compile_x(\n"  # syntax error


def _paths(tmp_path):
    """Return the keyword paths that redirect all state into tmp_path."""
    return dict(
        staging_dir=tmp_path / "staging",
        task_dir=tmp_path / "tasks",
        pending_path=tmp_path / "pending.json",
        audit_path=tmp_path / "audit.jsonl",
        sentinel=tmp_path / ".compiler_disabled",
    )


# ── static safety ────────────────────────────────────────────────


def test_static_safety_allows_clean_module():
    ok, flags = ect.static_safety_check(SAFE)
    assert ok and flags == []


def test_static_safety_allows_numpy_onnx():
    ok, flags = ect.static_safety_check(
        "import numpy as np\nimport onnx\ndef compile_x():\n    return np.zeros(3)\n"
    )
    assert ok, flags


@pytest.mark.parametrize("code,needle", [
    (UNSAFE_OS, "os"),
    (UNSAFE_EVAL, "eval"),
    (UNSAFE_SOCKET, "socket"),
    (UNSAFE_DUNDER, "__bases__"),
])
def test_static_safety_rejects_dangerous(code, needle):
    ok, flags = ect.static_safety_check(code)
    assert not ok
    assert any(needle in f for f in flags)


# ── sandboxed import check (never in-process) ────────────────────


def test_import_check_finds_functions():
    ok, err, fns = ect.import_check_module(SAFE)
    assert ok and err == ""
    assert "detect_flip" in fns and "compile_flip" in fns


def test_import_check_catches_runtime_error():
    ok, err, fns = ect.import_check_module("raise RuntimeError('boom')\n")
    assert not ok and "boom" in err


# ── staging (autonomous, L2) ─────────────────────────────────────


def test_stage_success_writes_staging_and_queue(tmp_path):
    p = _paths(tmp_path)
    res = ect.stage_compiler_module(
        SAFE, [], "flip", provenance={"rationale": "handle mirror tasks"}, **p
    )
    assert res["staged"] is True
    pid = res["pending_id"]
    # staged file exists in staging dir, NOT anywhere callable
    assert (p["staging_dir"] / f"{pid}.py").exists()
    # queued as pending for the dashboard
    pend = ect.list_pending_promotions(p["pending_path"])
    assert len(pend) == 1 and pend[0]["id"] == pid and pend[0]["status"] == "pending"
    # audit recorded
    lines = p["audit_path"].read_text().strip().split("\n")
    assert any(json.loads(ln)["event"] == "staged" for ln in lines)


def test_stage_rejects_unsafe_before_execution(tmp_path):
    p = _paths(tmp_path)
    res = ect.stage_compiler_module(UNSAFE_OS, [], "evil", **p)
    assert res["staged"] is False
    assert res["reason"] == "failed static safety scan"
    assert ect.list_pending_promotions(p["pending_path"]) == []


def test_stage_rejects_syntax_error(tmp_path):
    p = _paths(tmp_path)
    res = ect.stage_compiler_module(BROKEN, [], "broken", **p)
    assert res["staged"] is False
    assert res["stages"][0]["stage"] == "syntax" and not res["stages"][0]["ok"]


def test_kill_switch_blocks_staging(tmp_path):
    p = _paths(tmp_path)
    p["sentinel"].write_text("stop")
    res = ect.stage_compiler_module(SAFE, [], "flip", **p)
    assert res["staged"] is False and "disabled" in res["reason"]


# ── promotion requires human ack (L3) ────────────────────────────


def test_promote_moves_to_live_dir(tmp_path):
    p = _paths(tmp_path)
    compiler_dir = tmp_path / "compiler"
    staged = ect.stage_compiler_module(SAFE, [], "flip", **p)
    pid = staged["pending_id"]

    out = ect.promote_pending(
        pid, approved_by="andrew",
        compiler_dir=compiler_dir, pending_path=p["pending_path"],
        audit_path=p["audit_path"], sentinel=p["sentinel"],
    )
    assert out["ok"] is True
    # now callable: lives in the compiler dir, gone from staging
    assert (compiler_dir / "flip.py").exists()
    assert not (p["staging_dir"] / f"{pid}.py").exists()
    # no longer pending
    assert ect.list_pending_promotions(p["pending_path"]) == []


def test_promote_unknown_id_errors(tmp_path):
    p = _paths(tmp_path)
    out = ect.promote_pending(
        "nope", approved_by="x",
        compiler_dir=tmp_path / "c", pending_path=p["pending_path"],
        audit_path=p["audit_path"], sentinel=p["sentinel"],
    )
    assert out["ok"] is False and "no pending" in out["error"]


def test_promote_twice_is_rejected(tmp_path):
    p = _paths(tmp_path)
    compiler_dir = tmp_path / "compiler"
    pid = ect.stage_compiler_module(SAFE, [], "flip", **p)["pending_id"]
    kw = dict(approved_by="a", compiler_dir=compiler_dir,
              pending_path=p["pending_path"], audit_path=p["audit_path"],
              sentinel=p["sentinel"])
    assert ect.promote_pending(pid, **kw)["ok"] is True
    assert ect.promote_pending(pid, **kw)["ok"] is False  # already promoted


def test_kill_switch_blocks_promotion(tmp_path):
    p = _paths(tmp_path)
    pid = ect.stage_compiler_module(SAFE, [], "flip", **p)["pending_id"]
    p["sentinel"].write_text("stop")
    out = ect.promote_pending(
        pid, approved_by="a", compiler_dir=tmp_path / "c",
        pending_path=p["pending_path"], audit_path=p["audit_path"],
        sentinel=p["sentinel"],
    )
    assert out["ok"] is False and "disabled" in out["error"]


# ── rejection ────────────────────────────────────────────────────


def test_reject_marks_and_deletes(tmp_path):
    p = _paths(tmp_path)
    res = ect.stage_compiler_module(SAFE, [], "flip", **p)
    pid = res["pending_id"]
    out = ect.reject_pending(
        pid, reason="not general enough", rejected_by="andrew",
        pending_path=p["pending_path"], audit_path=p["audit_path"],
    )
    assert out["ok"] is True
    assert not (p["staging_dir"] / f"{pid}.py").exists()
    assert ect.list_pending_promotions(p["pending_path"]) == []


# ── backward-compat entry point no longer auto-promotes ──────────


def test_write_compiler_module_only_stages(tmp_path, monkeypatch):
    # redirect module-level defaults so the legacy signature is safe too
    monkeypatch.setattr(ect, "STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(ect, "PENDING_PATH", tmp_path / "pending.json")
    monkeypatch.setattr(ect, "AUDIT_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr(ect, "DISABLED_SENTINEL", tmp_path / ".disabled")
    res = ect.write_compiler_module(SAFE, [], "flip", task_dir=tmp_path / "tasks")
    assert res["promoted"] is False
    assert res["staged_for_review"] is True
