"""Smoke tests to verify the environment is set up correctly."""

import sys


def test_python_version():
    assert sys.version_info >= (3, 10)


def test_core_imports():
    import flask
    import numpy
    import requests
    import yaml


def test_dev_imports():
    """Dev tools are installed and invokable.

    ``ruff`` is a CLI tool with no importable Python package, so we
    verify it via ``--version`` rather than ``import ruff`` (which
    raises ImportError even when ruff is installed correctly).
    """
    import shutil
    import subprocess

    # pytest is a Python package; plain import is correct
    import pytest  # noqa: F401

    # ruff must be on PATH and respond to --version
    ruff_path = shutil.which("ruff")
    assert ruff_path, "ruff not on PATH — install via `pip install -e '.[dev]'`"
    result = subprocess.run([ruff_path, "--version"], capture_output=True, text=True)
    assert result.returncode == 0, f"ruff --version failed: {result.stderr}"
