import argparse
import json as json_mod
import importlib.metadata
import re
from dataclasses import dataclass, asdict
from typing import Callable
import os
import shutil
import socket
import subprocess
import sys

REQUIRED_PORTS = (8081, 8082, 8085, 4222, 8222)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--verbose", action="store_true", help="prints per-check details"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="enables output mode for machine consumption (CI integrations)",
    ),

    parser.add_argument(
        "--fix",
        action="store_true",
        help="enable auto-fixing problems (when applicable)",
    )

    return parser.parse_args()


@dataclass
class Check:
    name: str
    passed: bool = False
    message: str = ""
    suggestion: str = ""
    fix_command: str | None = None
    advisory: bool = False


CheckFn = Callable[[], Check | None]


def check_python_version() -> Check:
    check = Check(name="Python version >= 3.10")
    if sys.version_info >= (3, 10, 0):
        check.passed = True
        check.message = "Python version is compatible"
    else:
        check.passed = False
        check.message = "Python version incompatible"
        check.suggestion = "Install Python 3.12 from python.org"
    return check


def check_venv_active() -> Check:
    active = sys.prefix != sys.base_prefix
    if active:
        return Check(
            name="Virtual environment active",
            passed=True,
            message=f"venv active at {sys.prefix}",
        )
    return Check(
        name="Virtual environment active",
        passed=False,
        message="No virtual environment detected",
        suggestion="Run: source .venv/bin/activate",
        fix_command="python3 -m venv .venv && source .venv/bin/activate",
    )


def _read_pyproject_dependencies() -> list[str]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    deps = [
        re.split(r"[<>=!~ ]", d, 1)[0]
        for d in data.get("project", {}).get("dependencies", [])
    ]
    return deps


def check_core_dependencies() -> Check:
    deps = _read_pyproject_dependencies()
    if not deps:
        return Check(
            name="Core dependencies installed",
            passed=False,
            message="Could not read dependencies from pyproject.toml",
            suggestion="Ensure pyproject.toml exists in project root",
        )
    missing = []
    for dep in deps:
        try:
            importlib.metadata.distribution(dep)
        except importlib.metadata.PackageNotFoundError:
            missing.append(dep)
    if not missing:
        return Check(
            name="Core dependencies installed",
            passed=True,
            message="All core packages are installed",
        )
    return Check(
        name="Core dependencies installed",
        passed=False,
        message=f'Missing packages: {", ".join(missing)}',
        suggestion="Run: pip install -e '.[dev]'",
        fix_command="pip install -e '.[dev]'",
    )


def check_git_installation() -> Check:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Check(
            name="Git installed and configured",
            passed=False,
            message="Git is not installed",
            suggestion="Install Git from https://git-scm.com/",
        )
    missing_configs = []
    for key in ["user.name", "user.email"]:
        result = subprocess.run(
            ["git", "config", "--global", key],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            missing_configs.append(key)
    if missing_configs:
        return Check(
            name="Git installed and configured",
            passed=False,
            message=f'Git config missing: {", ".join(missing_configs)}',
            suggestion="Run: git config --global user.name 'Your Name' && git config --global user.email 'you@example.com'",
        )
    return Check(
        name="Git installed and configured",
        passed=True,
        message="Git is installed and configured",
    )


def check_env_file() -> Check:
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(env_path):
        return Check(
            name=".env file exists",
            passed=True,
            message=".env file found",
        )
    return Check(
        name=".env file exists",
        passed=False,
        message=".env file not found in project root",
        suggestion="Copy .env.example to .env",
        fix_command="cp .env.example .env",
        advisory=True,
    )


def check_disk_space() -> Check:
    MIN_FREE_GB = 10
    usage = shutil.disk_usage(os.getcwd())
    free_gb = usage.free / (1024**3)
    if free_gb >= MIN_FREE_GB:
        return Check(
            name="Disk space > 10 GB free",
            passed=True,
            message=f"{free_gb:.1f} GB free",
        )
    return Check(
        name="Disk space > 10 GB free",
        passed=False,
        message=f"Only {free_gb:.1f} GB free",
        suggestion="Free up disk space",
    )


def check_required_ports(ports: tuple[int, ...] = REQUIRED_PORTS) -> Check:
    busy_ports = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("localhost", port))
            except OSError:
                busy_ports.append(port)
    ports_label = "/".join(str(p) for p in ports)
    if not busy_ports:
        return Check(
            name=f"Required ports available ({ports_label})",
            passed=True,
            message="All required ports are free",
        )
    ports_str = ", ".join(str(p) for p in busy_ports)
    return Check(
        name=f"Required ports available ({ports_label})",
        passed=False,
        message=f"Ports in use: {ports_str}",
        suggestion=f"Kill process on port: lsof -ti:{busy_ports[0]} | xargs kill",
    )


def _has_docker_config() -> bool:
    for name in ("Dockerfile", "docker-compose.yml", "docker-compose.yaml"):
        if os.path.isfile(os.path.join(os.getcwd(), name)):
            return True
    return False


def check_docker_running() -> Check | None:
    if not _has_docker_config():
        return None
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=True,
        )
        return Check(
            name="Docker running",
            passed=True,
            message="Docker daemon is running",
        )
    except FileNotFoundError:
        return Check(
            name="Docker running",
            passed=False,
            message="Docker is not installed",
            suggestion="Install Docker from https://docs.docker.com/get-docker/",
        )
    except subprocess.CalledProcessError:
        return Check(
            name="Docker running",
            passed=False,
            message="Docker is installed but the daemon is not running",
            suggestion="Start Docker Desktop",
        )


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def perform_checks(
    check_fns: list[CheckFn],
    verbose: bool = False,
    json: bool = False,
    fix: bool = False,
) -> list[Check]:
    def _try_fix(check: Check, retry: CheckFn) -> Check:
        if check.passed or not fix or not check.fix_command:
            return check
        print(f"  {YELLOW}⚙ Attempting fix:{RESET} {check.fix_command}")
        ret = subprocess.run(check.fix_command, shell=True, capture_output=True)
        if ret.returncode != 0:
            return check
        return retry() or check

    results: list[Check] = []
    for fn in check_fns:
        check = fn()
        if check is None:
            continue
        results.append(_try_fix(check, fn))

    if json:
        print(json_mod.dumps([asdict(c) for c in results], indent=2))
        return results

    for check in results:
        if check.passed:
            icon = f"{GREEN}✅{RESET}"
        elif check.advisory:
            icon = f"{YELLOW}⚠{RESET}"
        else:
            icon = f"{RED}❌{RESET}"
        print(f"{icon} {BOLD}{check.name}{RESET} — {check.message}")
        if verbose and check.passed:
            print(f"     {check.message}")
        if not check.passed and check.suggestion:
            print(f"     {YELLOW}↳ {check.suggestion}{RESET}")

    failures = sum(1 for c in results if not c.passed and not c.advisory)
    warnings = sum(1 for c in results if not c.passed and c.advisory)
    total = len(results)
    print()
    parts = []
    if failures:
        parts.append(f"{RED}{BOLD}{failures}/{total} checks failed.{RESET}")
    if warnings:
        parts.append(f"{YELLOW}{warnings} advisory warning(s).{RESET}")
    if not parts:
        print(f"{GREEN}{BOLD}All {total} checks passed.{RESET}")
    else:
        print("  ".join(parts))

    return results


ALL_CHECKS: list[CheckFn] = [
    check_python_version,
    check_venv_active,
    check_core_dependencies,
    check_git_installation,
    check_env_file,
    check_disk_space,
    check_required_ports,
    check_docker_running,
]


def main():
    args = parse_args()
    results = perform_checks(
        ALL_CHECKS,
        verbose=args.verbose,
        json=args.json,
        fix=args.fix,
    )
    failed = any(not c.passed and not c.advisory for c in results)
    sys.exit(1 if failed else 0)
