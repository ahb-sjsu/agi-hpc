"""Focused preflight: what is the existing livekit-smoke container?

The first preflight discovered a container named ``livekit-smoke``
running for 41 hours on Atlas, with :7880 and :7881 bound.
Before I propose anything that collides with it, I need to know
what it is.
"""

from __future__ import annotations

import sys

import paramiko

ATLAS_HOST = "100.68.134.21"
ATLAS_USER = "claude"
ATLAS_PASS = "roZes9090!~"

_CHECKS: list[tuple[str, str]] = [
    ("livekit-smoke image / command",
        "docker inspect livekit-smoke --format '{{.Config.Image}} | {{.Config.Cmd}} | {{.NetworkSettings.IPAddress}}' 2>&1"),
    ("livekit-smoke ports",
        "docker port livekit-smoke 2>&1"),
    ("livekit-smoke env (filtered)",
        "docker inspect livekit-smoke --format '{{range .Config.Env}}{{.}}{{\"\\n\"}}{{end}}' | head -20"),
    ("livekit-smoke mounts",
        "docker inspect livekit-smoke --format '{{range .Mounts}}{{.Source}} -> {{.Destination}} ({{.Mode}}){{\"\\n\"}}{{end}}'"),
    ("livekit-smoke labels",
        "docker inspect livekit-smoke --format '{{range $k,$v := .Config.Labels}}{{$k}}={{$v}}{{\"\\n\"}}{{end}}' 2>&1 | head"),
    ("livekit-smoke started by",
        "docker inspect livekit-smoke --format '{{.Name}} started {{.State.StartedAt}}; created {{.Created}}'"),
    ("livekit-smoke recent logs",
        "docker logs --tail 20 livekit-smoke 2>&1"),
    ("compose-stack membership",
        "docker ps -a --filter 'label=com.docker.compose.project' --format '{{.Names}} {{.Labels}}' | head"),
    ("nats server anywhere?",
        "docker ps --format '{{.Names}} {{.Image}}' | grep -i nats 2>&1; echo '---'; systemctl list-units --all --no-pager 2>&1 | grep -i nats | head"),
    ("caddy config referenced sites",
        "grep -E '^\\s*[a-z0-9].*\\.[a-z0-9]+\\s*\\{' /etc/caddy/Caddyfile 2>&1 | head -20"),
    ("repo remote origin",
        "cd /home/claude/agi-hpc && git remote -v | head -3"),
    ("all branches on Atlas",
        "cd /home/claude/agi-hpc && git branch -a | head -20"),
    ("recent commits on current branch",
        "cd /home/claude/agi-hpc && git log --oneline -10"),
]


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        ATLAS_HOST, username=ATLAS_USER, password=ATLAS_PASS,
        timeout=10, allow_agent=False, look_for_keys=False,
    )
    print("─" * 70)
    print("  Halyard Table — focused livekit / context preflight")
    print("─" * 70)
    for label, cmd in _CHECKS:
        _stdin, stdout, stderr = client.exec_command(cmd, timeout=15)
        out = stdout.read().decode("utf-8", errors="replace").rstrip()
        err = stderr.read().decode("utf-8", errors="replace").rstrip()
        print(f"\n── {label} ──")
        for line in (out or err).splitlines():
            print("   " + line)
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
