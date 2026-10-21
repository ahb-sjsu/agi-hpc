# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for systemd service configuration and watchdog.

Validates that all service files exist, have proper restart policies,
correct ports, and the watchdog script covers all services.
"""

from __future__ import annotations

from pathlib import Path

SYSTEMD_DIR = Path("deploy/systemd")


class TestServiceFiles:
    """Tests for systemd service file existence."""

    EXPECTED_SERVICES = [
        "atlas-nats",
        "atlas-superego",
        "atlas-id",
        "atlas-ego",
        "atlas-rag-server",
        "atlas-telemetry",
        "atlas-caddy",
        "atlas-oauth2-proxy",
        "atlas-watchdog",
        "atlas-backup",
        "atlas-training",
    ]

    def test_all_services_exist(self) -> None:
        for svc in self.EXPECTED_SERVICES:
            assert (SYSTEMD_DIR / f"{svc}.service").exists(), f"{svc}.service missing"

    def test_target_exists(self) -> None:
        assert (SYSTEMD_DIR / "atlas.target").exists()

    def test_timers_exist(self) -> None:
        assert (SYSTEMD_DIR / "atlas-training.timer").exists()
        assert (SYSTEMD_DIR / "atlas-backup.timer").exists()


class TestRestartPolicies:
    """Tests that all long-running services have Restart=always."""

    LONG_RUNNING = [
        "atlas-nats",
        "atlas-superego",
        "atlas-id",
        "atlas-ego",
        "atlas-rag-server",
        "atlas-telemetry",
        "atlas-caddy",
        "atlas-oauth2-proxy",
        "atlas-watchdog",
    ]

    def test_restart_always(self) -> None:
        for svc in self.LONG_RUNNING:
            content = (SYSTEMD_DIR / f"{svc}.service").read_text()
            assert "Restart=always" in content, f"{svc} missing Restart=always"

    def test_oneshot_services_no_restart(self) -> None:
        for svc in ["atlas-backup", "atlas-training"]:
            content = (SYSTEMD_DIR / f"{svc}.service").read_text()
            assert "Type=oneshot" in content


class TestPortAssignments:
    """Tests that services use correct ports."""

    def test_superego_port_8080(self) -> None:
        content = (SYSTEMD_DIR / "atlas-superego.service").read_text()
        assert "8080" in content

    def test_id_port_8082(self) -> None:
        content = (SYSTEMD_DIR / "atlas-id.service").read_text()
        assert "8082" in content

    def test_ego_port_8084(self) -> None:
        content = (SYSTEMD_DIR / "atlas-ego.service").read_text()
        assert "8084" in content

    def test_nats_port_4222(self) -> None:
        content = (SYSTEMD_DIR / "atlas-nats.service").read_text()
        assert "4222" in content

    def test_ego_cpu_only(self) -> None:
        content = (SYSTEMD_DIR / "atlas-ego.service").read_text()
        assert "--n-gpu-layers 0" in content


class TestGPUAssignments:
    """Tests that GPU services use correct CUDA devices."""

    def test_superego_gpu0(self) -> None:
        content = (SYSTEMD_DIR / "atlas-superego.service").read_text()
        assert "CUDA_VISIBLE_DEVICES=0" in content

    def test_id_gpu1(self) -> None:
        content = (SYSTEMD_DIR / "atlas-id.service").read_text()
        assert "CUDA_VISIBLE_DEVICES=1" in content


class TestTargetDependencies:
    """Tests that atlas.target wants all services."""

    def test_wants_all_llm_services(self) -> None:
        content = (SYSTEMD_DIR / "atlas.target").read_text()
        assert "atlas-superego" in content
        assert "atlas-id" in content
        assert "atlas-ego" in content

    def test_wants_infrastructure(self) -> None:
        content = (SYSTEMD_DIR / "atlas.target").read_text()
        assert "atlas-nats" in content
        assert "atlas-rag-server" in content
        assert "atlas-caddy" in content


class TestWatchdogScript:
    """Tests for the watchdog health-check script."""

    def test_script_exists(self) -> None:
        assert Path("scripts/watchdog.sh").exists()

    def test_monitors_all_services(self) -> None:
        content = Path("scripts/watchdog.sh").read_text()
        assert "8082" in content  # Id (NRP LLM proxy; Superego 8080 is off)
        assert "8084" in content  # Ego
        assert "8081" in content  # RAG
        assert "8222" in content  # NATS

    def test_checks_thermals(self) -> None:
        content = Path("scripts/watchdog.sh").read_text()
        assert "nvidia-smi" in content
        assert "sensors" in content
        assert "THERMAL WARNING" in content


class TestInstallScript:
    """Tests for the install script."""

    def test_includes_all_services(self) -> None:
        content = (SYSTEMD_DIR / "install-services.sh").read_text()
        assert "atlas-nats" in content
        assert "atlas-llm-kirk" in content
        assert "atlas-ego" in content
        assert "atlas-watchdog" in content

    def test_includes_timers(self) -> None:
        content = (SYSTEMD_DIR / "install-services.sh").read_text()
        assert "atlas-training" in content
        assert "atlas-backup" in content

    def test_includes_targets(self) -> None:
        content = (SYSTEMD_DIR / "install-services.sh").read_text()
        assert "atlas" in content
        assert ".target" in content
