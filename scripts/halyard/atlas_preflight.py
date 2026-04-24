"""Halyard Table — Atlas preflight check.

Read-only inspection of Atlas: what's installed, what's running,
what ports are bound, what the Caddy config layout looks like.
Produces a report on stdout. Never modifies anything.

Uses paramiko per ``.claude/rules/atlas-operations.md``.

Usage::

    python scripts/halyard/atlas_preflight.py

Exits 0 on success (report printed). Exits non-zero if Atlas is
unreachable or auth fails.
"""

from __future__ import annotations

import sys

import paramiko

ATLAS_HOST = "100.68.134.21"  # Tailscale
ATLAS_USER = "claude"
ATLAS_PASS = "roZes9090!~"  # stored per atlas-operations.md


_CHECKS: list[tuple[str, str]] = [
    ("OS / kernel",                    "uname -a"),
    ("Date",                           "date -u"),
    ("Checkout HEAD",                  "cd /home/claude/agi-hpc && git rev-parse --short HEAD && git status --porcelain=v1 -b | head -5"),
    ("Python venv",                    "ls -la /home/claude/env/bin/python 2>&1 | head -3"),
    ("Docker",                         "docker --version && docker compose version 2>/dev/null | head -2"),
    ("Caddy",                          "systemctl status caddy.service --no-pager 2>&1 | head -5; echo '---'; ls /etc/caddy/ 2>&1 | head; echo '---'; ls /etc/caddy/sites/ 2>&1 | head"),
    ("NATS",                           "systemctl --user status atlas-nats 2>&1 | head -10 || systemctl status nats 2>&1 | head -10"),
    ("atlas-primer",                   "systemctl --user status atlas-primer 2>&1 | head -10"),
    ("User services list",             "systemctl --user list-units --type=service --all --no-pager 2>&1 | head -20"),
    ("User lingering",                 "loginctl show-user claude | grep -i linger 2>&1"),
    ("Bound ports (0.0.0.0)",          "ss -tuln 2>&1 | head -30"),
    ("DNS halyard subdomain",          "dig +short halyard.atlas-sjsu.duckdns.org a || echo 'dig not present'"),
    ("Disk free on /",                 "df -h / 2>&1 | tail -2"),
    ("Disk free on /archive",          "df -h /archive 2>&1 | tail -2 || echo '/archive not present'"),
    ("Halyard code present?",          "ls /home/claude/agi-hpc/src/agi/halyard/ 2>&1 | head"),
    ("LiveKit SFU dir on disk?",       "ls /home/claude/agi-hpc/infra/local/livekit-sfu/ 2>&1 | head"),
    ("Existing docker containers",     "docker ps --format '{{.Names}} {{.Status}}' 2>&1 | head -20"),
    ("Ports we plan to use",           "for p in 7880 7881 3478 8090 8091; do echo -n \"  :$p \"; ss -tln src \":$p\" 2>/dev/null | tail +2 | head -1 || echo ''; done"),
]


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            ATLAS_HOST,
            username=ATLAS_USER,
            password=ATLAS_PASS,
            timeout=10,
            allow_agent=False,
            look_for_keys=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"FAILED to connect: {e}", file=sys.stderr)
        return 1

    print("═" * 70)
    print(f"  Halyard Table — Atlas preflight ({ATLAS_HOST})")
    print("═" * 70)

    for label, cmd in _CHECKS:
        _stdin, stdout, stderr = client.exec_command(cmd, timeout=15)
        out = stdout.read().decode("utf-8", errors="replace").rstrip()
        err = stderr.read().decode("utf-8", errors="replace").rstrip()
        exit_code = stdout.channel.recv_exit_status()
        print(f"\n── {label} ──" + (f" (exit {exit_code})" if exit_code else ""))
        if out:
            for line in out.splitlines():
                print("   " + line)
        if err and not out:
            for line in err.splitlines():
                print("   (stderr) " + line)

    client.close()
    print()
    print("═" * 70)
    print("  Preflight complete. No modifications were made.")
    print("═" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
