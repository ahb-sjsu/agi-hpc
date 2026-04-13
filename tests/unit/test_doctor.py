"""Unit tests for agi.doctor checks."""

import os
import shutil
import socket
import subprocess
import sys
from collections import namedtuple
from unittest import mock

import pytest

from agi.doctor import (
    Check,
    check_python_version,
    check_venv_active,
    check_core_dependencies,
    check_git_installation,
    check_env_file,
    check_disk_space,
    check_required_ports,
    check_docker_running,
    perform_checks,
    _read_pyproject_dependencies,
)


# ── check_python_version ────────────────────────────────────────────

class TestCheckPythonVersion:
    def test_passes_on_310_plus(self):
        with mock.patch.object(sys, 'version_info', (3, 12, 0)):
            result = check_python_version()
        assert result.passed is True

    def test_fails_on_old_python(self):
        with mock.patch.object(sys, 'version_info', (3, 9, 1)):
            result = check_python_version()
        assert result.passed is False
        assert result.suggestion != ''


# ── check_venv_active ───────────────────────────────────────────────

class TestCheckVenvActive:
    def test_passes_when_venv_active(self):
        with mock.patch.object(sys, 'prefix', '/some/venv'), \
             mock.patch.object(sys, 'base_prefix', '/usr'):
            result = check_venv_active()
        assert result.passed is True

    def test_fails_when_no_venv(self):
        with mock.patch.object(sys, 'prefix', '/usr'), \
             mock.patch.object(sys, 'base_prefix', '/usr'):
            result = check_venv_active()
        assert result.passed is False
        assert 'activate' in result.suggestion


# ── check_core_dependencies ─────────────────────────────────────────

class TestCheckCoreDependencies:
    def test_passes_when_all_importable(self, monkeypatch):
        monkeypatch.setattr(
            'agi.doctor._read_pyproject_dependencies',
            lambda: ['flask', 'requests'],
        )
        result = check_core_dependencies()
        assert result.passed is True

    def test_fails_when_package_missing(self, monkeypatch):
        monkeypatch.setattr(
            'agi.doctor._read_pyproject_dependencies',
            lambda: ['nonexistent_pkg_xyz'],
        )
        result = check_core_dependencies()
        assert result.passed is False
        assert 'nonexistent_pkg_xyz' in result.message

    def test_fails_when_no_pyproject(self, monkeypatch):
        monkeypatch.setattr(
            'agi.doctor._read_pyproject_dependencies',
            lambda: [],
        )
        result = check_core_dependencies()
        assert result.passed is False
        assert 'pyproject.toml' in result.message


# ── check_git_installation ──────────────────────────────────────────

class TestCheckGitInstallation:
    def test_passes_when_git_configured(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            result = subprocess.CompletedProcess(cmd, 0)
            result.stdout = 'value\n'
            return result

        monkeypatch.setattr(subprocess, 'run', fake_run)
        result = check_git_installation()
        assert result.passed is True

    def test_fails_when_git_missing(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise FileNotFoundError

        monkeypatch.setattr(subprocess, 'run', fake_run)
        result = check_git_installation()
        assert result.passed is False
        assert 'not installed' in result.message

    def test_fails_when_config_missing(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd == ['git', '--version']:
                return subprocess.CompletedProcess(cmd, 0)
            # git config returns failure
            r = subprocess.CompletedProcess(cmd, 1)
            r.stdout = ''
            return r

        monkeypatch.setattr(subprocess, 'run', fake_run)
        result = check_git_installation()
        assert result.passed is False
        assert 'config missing' in result.message


# ── check_env_file ──────────────────────────────────────────────────

class TestCheckEnvFile:
    def test_passes_when_env_exists(self, tmp_path, monkeypatch):
        (tmp_path / '.env').touch()
        monkeypatch.setattr(os, 'getcwd', lambda: str(tmp_path))
        result = check_env_file()
        assert result.passed is True

    def test_fails_when_env_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os, 'getcwd', lambda: str(tmp_path))
        result = check_env_file()
        assert result.passed is False


# ── check_disk_space ────────────────────────────────────────────────

DiskUsage = namedtuple('DiskUsage', ['total', 'used', 'free'])


class TestCheckDiskSpace:
    def test_passes_with_enough_space(self, monkeypatch):
        monkeypatch.setattr(
            shutil, 'disk_usage',
            lambda path: DiskUsage(100e9, 50e9, 50e9),
        )
        result = check_disk_space()
        assert result.passed is True

    def test_fails_with_low_space(self, monkeypatch):
        monkeypatch.setattr(
            shutil, 'disk_usage',
            lambda path: DiskUsage(100e9, 98e9, 2e9),
        )
        result = check_disk_space()
        assert result.passed is False
        assert 'GB' in result.message


# ── check_required_ports ────────────────────────────────────────────

class TestCheckRequiredPorts:
    def test_passes_when_ports_free(self, monkeypatch):
        class FakeSocket:
            def __init__(self, *a, **kw): pass
            def connect_ex(self, addr): return 1  # refused = free
            def __enter__(self): return self
            def __exit__(self, *a): pass

        monkeypatch.setattr(socket, 'socket', FakeSocket)
        result = check_required_ports()
        assert result.passed is True

    def test_fails_when_port_busy(self, monkeypatch):
        class FakeSocket:
            def __init__(self, *a, **kw): pass
            def connect_ex(self, addr): return 0  # connected = busy
            def __enter__(self): return self
            def __exit__(self, *a): pass

        monkeypatch.setattr(socket, 'socket', FakeSocket)
        result = check_required_ports()
        assert result.passed is False
        assert '8000' in result.message


# ── check_docker_running ────────────────────────────────────────────

class TestCheckDockerRunning:
    def test_passes_when_docker_running(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, 'run',
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0),
        )
        result = check_docker_running()
        assert result.passed is True

    def test_fails_when_docker_not_installed(self, monkeypatch):
        def fake_run(cmd, **kw):
            raise FileNotFoundError

        monkeypatch.setattr(subprocess, 'run', fake_run)
        result = check_docker_running()
        assert result.passed is False
        assert 'not installed' in result.message

    def test_fails_when_daemon_stopped(self, monkeypatch):
        def fake_run(cmd, **kw):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(subprocess, 'run', fake_run)
        result = check_docker_running()
        assert result.passed is False
        assert 'not running' in result.message


# ── perform_checks ──────────────────────────────────────────────────

class TestPerformChecks:
    def test_returns_results_for_all_checks(self, capsys):
        def passing(): return Check(name='ok', passed=True, message='good')
        def failing(): return Check(name='bad', passed=False, message='fail', suggestion='fix it')

        results = perform_checks([passing, failing])
        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False

    def test_json_mode(self, capsys):
        import json
        def passing(): return Check(name='ok', passed=True, message='good')

        perform_checks([passing], json=True)
        output = capsys.readouterr().out
        data = json.loads(output)
        assert len(data) == 1
        assert data[0]['passed'] is True

    def test_fix_mode_retries(self, monkeypatch):
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Check(name='fixable', passed=False, message='broken', fix_command='echo fix')
            return Check(name='fixable', passed=True, message='fixed')

        monkeypatch.setattr(
            subprocess, 'run',
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0),
        )
        results = perform_checks([flaky], fix=True)
        assert results[0].passed is True
        assert call_count == 2
