# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Sprint-1 SFU configuration tests.

These are static tests of the LiveKit SFU deployment artifacts in
``infra/local/livekit-sfu/``. They do not bring the SFU up; they
validate that the shipped config is syntactically well-formed and
has the invariants the runbook documents.

The acceptance criteria in ``HALYARD_SPRINT_PLAN.md`` §Sprint-1
include an end-to-end two-browser test that cannot be automated in
this repo's CI (UDP media ports, real browser). These tests cover
everything up to that point.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# ─────────────────────────────────────────────────────────────────
# Fixtures — repo root + SFU dir
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Locate the agi-hpc repo root from this test file."""
    here = Path(__file__).resolve()
    # tests/unit/halyard/ → repo root is 3 parents up
    root = here.parents[3]
    assert (root / "pyproject.toml").exists() or (root / "setup.cfg").exists() \
        or (root / "src" / "agi").is_dir(), f"not a repo root: {root}"
    return root


@pytest.fixture(scope="module")
def sfu_dir(repo_root: Path) -> Path:
    """Path to the LiveKit SFU deployment directory."""
    d = repo_root / "infra" / "local" / "livekit-sfu"
    assert d.is_dir(), f"SFU dir missing: {d}"
    return d


# ─────────────────────────────────────────────────────────────────
# File presence
# ─────────────────────────────────────────────────────────────────


EXPECTED_FILES = [
    "docker-compose.yml",
    "livekit.yaml",
    "livekit-keys.yaml.example",
    ".env.example",
    ".gitignore",
    "caddy.snippet",
    "turn/turnserver.conf",
    "systemd/livekit-sfu.service",
    "README.md",
]


@pytest.mark.parametrize("rel_path", EXPECTED_FILES)
def test_sfu_file_present(sfu_dir: Path, rel_path: str) -> None:
    """Every file documented in the SFU README must exist on disk."""
    assert (sfu_dir / rel_path).is_file(), f"missing: {rel_path}"


# ─────────────────────────────────────────────────────────────────
# docker-compose.yml
# ─────────────────────────────────────────────────────────────────


def test_compose_parses(sfu_dir: Path) -> None:
    compose_text = (sfu_dir / "docker-compose.yml").read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_text)
    assert "services" in compose
    assert "livekit-server" in compose["services"]
    assert "coturn" in compose["services"]


def test_compose_uses_host_networking(sfu_dir: Path) -> None:
    """Host networking is required so LiveKit's 50000-60000/udp
    media ports are not subjected to Docker's per-container NAT.

    See HALYARD_TABLE.md §4.1 and the docker-compose.yml comment.
    """
    compose_text = (sfu_dir / "docker-compose.yml").read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_text)
    assert compose["services"]["livekit-server"]["network_mode"] == "host"
    assert compose["services"]["coturn"]["network_mode"] == "host"


def test_compose_pins_livekit_version(sfu_dir: Path) -> None:
    """Pinning the LiveKit image tag avoids ``latest`` surprises.

    Upgrades are a deliberate action with their own smoke test;
    a floating tag would mean every ``docker compose pull`` is a
    breaking-change roll of the dice.
    """
    compose_text = (sfu_dir / "docker-compose.yml").read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_text)
    image = compose["services"]["livekit-server"]["image"]
    assert image.startswith("livekit/livekit-server:"), image
    assert "latest" not in image, "must pin a specific tag"


def test_compose_mounts_config_readonly(sfu_dir: Path) -> None:
    """The SFU container must mount livekit.yaml and livekit-keys.yaml
    read-only. A writable mount of the keys file from inside the
    container would be a dangerous misconfiguration.
    """
    compose_text = (sfu_dir / "docker-compose.yml").read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_text)
    volumes = compose["services"]["livekit-server"]["volumes"]
    joined = " ".join(volumes)
    assert "livekit.yaml:/etc/livekit.yaml:ro" in joined
    assert "livekit-keys.yaml:/etc/livekit-keys.yaml:ro" in joined


# ─────────────────────────────────────────────────────────────────
# livekit.yaml
# ─────────────────────────────────────────────────────────────────


def test_livekit_yaml_parses(sfu_dir: Path) -> None:
    cfg = yaml.safe_load((sfu_dir / "livekit.yaml").read_text(encoding="utf-8"))
    assert isinstance(cfg, dict)


def test_livekit_ports_match_contract(sfu_dir: Path) -> None:
    """Port numbers are a cross-system contract (Caddy snippet, agent
    config, smoke test, runbook). If they change here, they change
    everywhere."""
    cfg = yaml.safe_load((sfu_dir / "livekit.yaml").read_text(encoding="utf-8"))
    assert cfg["port"] == 7880
    assert cfg["rtc"]["tcp_port"] == 7881
    assert cfg["rtc"]["port_range_start"] == 50000
    assert cfg["rtc"]["port_range_end"] == 60000


def test_livekit_turn_disabled_in_sfu(sfu_dir: Path) -> None:
    """Built-in TURN is OFF — we use the coturn sidecar instead.

    Two TURN servers listening on the same ports would be a fight
    we'd lose silently.
    """
    cfg = yaml.safe_load((sfu_dir / "livekit.yaml").read_text(encoding="utf-8"))
    assert cfg["turn"]["enabled"] is False


def test_livekit_logs_structured(sfu_dir: Path) -> None:
    """JSON logs keep the SFU's output aligned with atlas-primer's
    structured-log conventions so we can mux them into one
    dashboard."""
    cfg = yaml.safe_load((sfu_dir / "livekit.yaml").read_text(encoding="utf-8"))
    assert cfg["logging"]["json"] is True


def test_livekit_rooms_sane(sfu_dir: Path) -> None:
    cfg = yaml.safe_load((sfu_dir / "livekit.yaml").read_text(encoding="utf-8"))
    # Auto-create so Keeper doesn't need a pre-create step.
    assert cfg["room"]["auto_create"] is True
    # Empty timeout = 24h so a brief disconnect doesn't wipe the
    # room's server-side state.
    assert cfg["room"]["empty_timeout"] == 86400
    # Max participants comfortably above expected table size.
    assert cfg["room"]["max_participants"] >= 15


# ─────────────────────────────────────────────────────────────────
# turnserver.conf — plain-text format, grep for key directives
# ─────────────────────────────────────────────────────────────────


def test_turn_shared_secret_auth(sfu_dir: Path) -> None:
    """coturn must be using shared-secret auth, not static users.

    Static users are a rotation problem in a live-play setting;
    shared-secret TURN REST API auth generates time-limited creds
    automatically.
    """
    turn_conf = (sfu_dir / "turn" / "turnserver.conf").read_text(encoding="utf-8")
    assert "use-auth-secret=yes" in turn_conf
    assert "static-auth-secret" in turn_conf


def test_turn_relay_range_disjoint_from_livekit(sfu_dir: Path) -> None:
    """coturn's relay range (60001-61000) must not overlap with
    LiveKit's media range (50000-60000). Both run on host
    networking; overlap silently loses connections."""
    turn_conf = (sfu_dir / "turn" / "turnserver.conf").read_text(encoding="utf-8")
    assert "min-port=60001" in turn_conf
    assert "max-port=61000" in turn_conf


def test_turn_denies_private_ranges(sfu_dir: Path) -> None:
    """TURN must deny relays into Atlas's internal networks.

    Without this, a compromised browser could proxy back into
    atlas-primer or the Tailscale network. Defense in depth.
    """
    turn_conf = (sfu_dir / "turn" / "turnserver.conf").read_text(encoding="utf-8")
    required_denies = [
        "10.0.0.0-10.255.255.255",
        "172.16.0.0-172.31.255.255",
        "192.168.0.0-192.168.255.255",
        "127.0.0.0-127.255.255.255",
        "100.64.0.0-100.127.255.255",  # Tailscale CGNAT range
    ]
    for rng in required_denies:
        assert f"denied-peer-ip={rng}" in turn_conf, f"missing deny: {rng}"


# ─────────────────────────────────────────────────────────────────
# systemd unit
# ─────────────────────────────────────────────────────────────────


def test_systemd_unit_shape(sfu_dir: Path) -> None:
    """The livekit-sfu.service unit must declare the right sections
    and use docker compose for lifecycle."""
    unit = (sfu_dir / "systemd" / "livekit-sfu.service").read_text(encoding="utf-8")
    assert "[Unit]" in unit
    assert "[Service]" in unit
    assert "[Install]" in unit
    assert "Type=oneshot" in unit
    assert "RemainAfterExit=yes" in unit
    assert "docker compose" in unit
    # User-level unit targets default.target, not multi-user.target.
    assert "WantedBy=default.target" in unit


# ─────────────────────────────────────────────────────────────────
# Caddy snippet
# ─────────────────────────────────────────────────────────────────


def test_caddy_snippet_has_halyard_site(sfu_dir: Path) -> None:
    """The Caddy snippet must configure the halyard subdomain."""
    snippet = (sfu_dir / "caddy.snippet").read_text(encoding="utf-8")
    assert "halyard.atlas-sjsu.duckdns.org" in snippet
    assert "reverse_proxy" in snippet
    # WebSocket upgrade handled by LiveKit paths.
    assert "/rtc/*" in snippet


def test_caddy_snippet_has_hsts(sfu_dir: Path) -> None:
    """HSTS is a minimum security header for a player-facing host."""
    snippet = (sfu_dir / "caddy.snippet").read_text(encoding="utf-8")
    assert "Strict-Transport-Security" in snippet


# ─────────────────────────────────────────────────────────────────
# Smoke test script
# ─────────────────────────────────────────────────────────────────


def test_smoke_test_exists_and_shebangs(repo_root: Path) -> None:
    script = repo_root / "scripts" / "halyard" / "sfu_smoke.sh"
    assert script.is_file(), "smoke test script missing"
    first_line = script.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("#!"), "missing shebang"
    assert "bash" in first_line


def test_smoke_test_requires_api_credentials(repo_root: Path) -> None:
    """The smoke test must refuse to run without LIVEKIT_API_KEY and
    LIVEKIT_API_SECRET. Defensive exit keeps a misconfigured run
    from producing a misleading green."""
    script = (repo_root / "scripts" / "halyard" / "sfu_smoke.sh").read_text(
        encoding="utf-8"
    )
    assert 'LIVEKIT_API_KEY' in script
    assert 'LIVEKIT_API_SECRET' in script
    # Bash parameter expansion `${VAR:?...}` errors if unset.
    assert ':?' in script


def test_smoke_test_uses_existing_token_minter(repo_root: Path) -> None:
    """The smoke test should reuse the existing Python token-minting
    helper, not reimplement JWT signing in shell. Reduces drift
    between the test and the production agent path."""
    script = (repo_root / "scripts" / "halyard" / "sfu_smoke.sh").read_text(
        encoding="utf-8"
    )
    assert "mint_participant_token" in script
    assert "agi.primer.artemis.livekit_agent.token" in script


# ─────────────────────────────────────────────────────────────────
# .gitignore must keep secrets out
# ─────────────────────────────────────────────────────────────────


def test_gitignore_excludes_secrets(sfu_dir: Path) -> None:
    """A common footgun: accidentally committing the populated .env
    or livekit-keys.yaml. The .gitignore must explicitly list
    both."""
    gi = (sfu_dir / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gi
    assert "livekit-keys.yaml" in gi
    # Example files are checked in; real files are not.
    assert ".env.example" not in gi or ".env.example" not in gi.split("\n")
