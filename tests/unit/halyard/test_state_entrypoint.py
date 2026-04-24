# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for the ``python -m agi.halyard.state`` entrypoint.

We don't spin up the service (that requires asyncio + a free port);
we just guard the CLI / env handling so a bad default doesn't ship
silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_argparse_defaults_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env vars populate the argparse defaults when no CLI flag set."""
    from agi.halyard.state.__main__ import _parse_args

    monkeypatch.setenv("HALYARD_STATE_HOST", "127.0.0.1")
    monkeypatch.setenv("HALYARD_STATE_PORT", "9999")
    monkeypatch.setenv("HALYARD_ARCHIVE_ROOT", "/tmp/halyard-test")
    monkeypatch.setenv("NATS_URL", "nats://127.0.0.1:4222")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    # argparse reads sys.argv; reset it for this call.
    monkeypatch.setattr("sys.argv", ["halyard-state"])
    args = _parse_args()
    assert args.host == "127.0.0.1"
    assert args.port == 9999
    assert args.archive_root == "/tmp/halyard-test"
    assert args.nats_url == "nats://127.0.0.1:4222"
    assert args.log_level == "debug"


def test_argparse_cli_flags_override_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agi.halyard.state.__main__ import _parse_args

    monkeypatch.setenv("HALYARD_STATE_PORT", "1111")
    monkeypatch.setattr(
        "sys.argv",
        ["halyard-state", "--port", "2222", "--host", "10.0.0.1"],
    )
    args = _parse_args()
    assert args.port == 2222
    assert args.host == "10.0.0.1"


def test_argparse_no_nats_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NATS_URL unset → service runs REST+WS only. That's the
    intended dev/smoke-test default."""
    from agi.halyard.state.__main__ import _parse_args

    monkeypatch.delenv("NATS_URL", raising=False)
    monkeypatch.setattr("sys.argv", ["halyard-state"])
    args = _parse_args()
    assert args.nats_url == ""


def test_systemd_unit_present_and_parses() -> None:
    """The user-level systemd unit shipped alongside the service
    must exist and have the expected sections."""
    here = Path(__file__).resolve()
    unit = (
        here.parents[3]
        / "infra"
        / "local"
        / "halyard-state"
        / "systemd"
        / "halyard-state.service"
    )
    assert unit.is_file(), f"missing unit file: {unit}"
    text = unit.read_text(encoding="utf-8")
    assert "[Unit]" in text
    assert "[Service]" in text
    assert "[Install]" in text
    assert "ExecStart=" in text
    assert "python -m agi.halyard.state" in text
    # User-level unit targets default.target.
    assert "WantedBy=default.target" in text


def test_systemd_unit_has_sandbox_hardening() -> None:
    """The unit must apply the standard user-level sandbox flags.

    Not exhaustive (user-level systemd limits what we can
    enforce), but catches a regression where someone deletes the
    hardening section on the theory that the service is 'just
    python'."""
    here = Path(__file__).resolve()
    unit = (
        here.parents[3]
        / "infra"
        / "local"
        / "halyard-state"
        / "systemd"
        / "halyard-state.service"
    )
    text = unit.read_text(encoding="utf-8")
    required = [
        "NoNewPrivileges=yes",
        "PrivateTmp=yes",
        "ProtectSystem=strict",
        "ProtectHome=read-only",
        "ProtectKernelTunables=yes",
    ]
    for line in required:
        assert line in text, f"missing hardening: {line}"


def test_readme_present() -> None:
    """The operator runbook must ship with the unit."""
    here = Path(__file__).resolve()
    readme = (
        here.parents[3]
        / "infra"
        / "local"
        / "halyard-state"
        / "README.md"
    )
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    assert "Deploy on Atlas" in text
    assert "systemctl --user" in text
