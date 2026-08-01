# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for CI/CD pipeline configuration.

Validates that the GitHub Actions workflow and deploy script
exist, have correct structure, and don't contain secrets.
"""

import os
from __future__ import annotations

from pathlib import Path


class TestWorkflowExists:
    """Tests for CI workflow file."""

    def test_ci_yaml_exists(self) -> None:
        assert Path(".github/workflows/ci.yaml").exists()

    def test_deploy_script_exists(self) -> None:
        assert Path("scripts/deploy_to_atlas.sh").exists()


class TestWorkflowStructure:
    """Tests for CI workflow content."""

    def _read_ci(self) -> str:
        return Path(".github/workflows/ci.yaml").read_text()

    def test_has_test_job(self) -> None:
        content = self._read_ci()
        assert "test:" in content

    def test_has_deploy_job(self) -> None:
        content = self._read_ci()
        assert "deploy:" in content

    def test_deploy_needs_test(self) -> None:
        content = self._read_ci()
        assert "needs: test" in content

    def test_deploy_only_on_main(self) -> None:
        content = self._read_ci()
        assert "refs/heads/main" in content

    def test_runs_ruff(self) -> None:
        content = self._read_ci()
        assert "ruff" in content

    def test_runs_black(self) -> None:
        content = self._read_ci()
        assert "black" in content

    def test_runs_pytest(self) -> None:
        content = self._read_ci()
        assert "pytest" in content


class TestNoSecretsInCode:
    """Tests that no secrets are hardcoded in CI/CD files."""

    def test_no_password_in_workflow(self) -> None:
        content = Path(".github/workflows/ci.yaml").read_text()
        secret = os.environ.get("ATLAS_PASS")
        if secret:
            assert secret not in content
        assert "password123" not in content

    def test_uses_github_secrets(self) -> None:
        content = Path(".github/workflows/ci.yaml").read_text()
        assert "secrets.ATLAS_HOST" in content
        assert "secrets.ATLAS_PASSWORD" in content

    def test_no_password_in_deploy_script(self) -> None:
        content = Path("scripts/deploy_to_atlas.sh").read_text()
        secret = os.environ.get("ATLAS_PASS")
        if secret:
            assert secret not in content

    def test_no_tailscale_ip_in_workflow(self) -> None:
        """Tailscale IP should be in secrets, not hardcoded."""
        content = Path(".github/workflows/ci.yaml").read_text()
        assert "100.68" not in content


class TestDeployScript:
    """Tests for the deploy script."""

    def test_has_shebang(self) -> None:
        content = Path("scripts/deploy_to_atlas.sh").read_text()
        assert content.startswith("#!/bin/bash")

    def test_copies_dashboard(self) -> None:
        content = Path("scripts/deploy_to_atlas.sh").read_text()
        assert "schematic.html" in content

    def test_restarts_rag_server(self) -> None:
        content = Path("scripts/deploy_to_atlas.sh").read_text()
        assert "atlas-rag-server" in content

    def test_verifies_services(self) -> None:
        content = Path("scripts/deploy_to_atlas.sh").read_text()
        assert "8081" in content  # RAG
        assert "8080" in content  # Superego
