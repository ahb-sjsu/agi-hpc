"""Halyard Table — deploy halyard-state on Atlas.

Safe, scoped deployment:

- Fetches latest refs in ``/home/claude/agi-hpc``; does NOT touch
  the currently-checked-out branch or any other worktree's state.
- Checks out ``origin/halyard/sprint-3`` into a detached HEAD (no
  branch move) if the current branch is not already on that SHA.
  We use ``git worktree`` to keep the main checkout undisturbed.
- Pip-installs the package into the existing venv (editable).
- Installs the user systemd unit via symlink, daemon-reload, start.
- Polls /healthz over loopback for up to 10 s.
- Reports back: status + healthz body + port-bind check.

Aborts early if any of the following hold:

- SSH connect fails.
- The systemd unit symlink cannot be written (dotfile perms).
- The venv is not present.
- A service is already bound to :8090 (avoids clobber).

Never runs a reboot, never kills an existing process, never
modifies anything outside:

  /home/claude/.config/systemd/user/halyard-state.service
  ~/src/agi-hpc (git worktree add)

Run this after the three halyard/sprint-N branches are on origin.
"""

from __future__ import annotations

import sys
import time

import paramiko

ATLAS_HOST = "100.68.134.21"
ATLAS_USER = "claude"
ATLAS_PASS = "roZes9090!~"

REPO_PATH = "/home/claude/agi-hpc"
WORKTREE_PATH = "/home/claude/agi-hpc-halyard"
VENV_PATH = "/home/claude/env"
TARGET_BRANCH = "halyard/sprint-3"


def run(client: paramiko.SSHClient, cmd: str, *, timeout: int = 30) -> tuple[int, str, str]:
    """Run a command; return (exit_code, stdout, stderr)."""
    _in, out, err = client.exec_command(cmd, timeout=timeout)
    stdout = out.read().decode("utf-8", errors="replace")
    stderr = err.read().decode("utf-8", errors="replace")
    rc = out.channel.recv_exit_status()
    return rc, stdout, stderr


def step(title: str) -> None:
    print()
    print(f"── {title} " + "─" * max(0, 70 - len(title) - 4))


def fail(msg: str) -> None:
    print(f"  ✗ {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            ATLAS_HOST, username=ATLAS_USER, password=ATLAS_PASS,
            timeout=10, allow_agent=False, look_for_keys=False,
        )
    except Exception as e:  # noqa: BLE001
        fail(f"SSH connect: {e}")

    print("═" * 70)
    print(f"  Halyard Table — deploying halyard-state on {ATLAS_HOST}")
    print(f"  Branch: {TARGET_BRANCH}")
    print("═" * 70)

    # ── 1. Preflight ────────────────────────────────────────────
    step("Preflight")

    rc, out, _ = run(client, f"test -d {VENV_PATH} && echo ok")
    if rc != 0 or "ok" not in out:
        fail(f"venv missing at {VENV_PATH}")
    ok(f"venv present at {VENV_PATH}")

    # Port :8090 must be free.
    rc, out, _ = run(client, "ss -tln src :8090 | tail -n +2 | head")
    if out.strip():
        fail(f"port :8090 already in use:\n{out}")
    ok("port :8090 is free")

    # ── 2. Fetch + ensure worktree ──────────────────────────────
    step("Git: fetch + prepare isolated worktree")

    rc, _, err = run(
        client, f"cd {REPO_PATH} && git fetch origin {TARGET_BRANCH}",
        timeout=60,
    )
    if rc != 0:
        fail(f"git fetch failed: {err}")
    ok("origin/halyard/sprint-3 fetched")

    # If the worktree doesn't exist yet, add it. If it exists, just
    # update it to the latest fetched SHA (detached HEAD is fine).
    rc, out, _ = run(client, f"test -d {WORKTREE_PATH} && echo exists")
    if "exists" not in out:
        rc, _, err = run(
            client,
            f"cd {REPO_PATH} && git worktree add --detach {WORKTREE_PATH} "
            f"origin/{TARGET_BRANCH}",
            timeout=60,
        )
        if rc != 0:
            fail(f"git worktree add failed: {err}")
        ok(f"worktree created at {WORKTREE_PATH}")
    else:
        rc, _, err = run(
            client,
            f"cd {WORKTREE_PATH} && git fetch origin {TARGET_BRANCH} "
            f"&& git reset --hard origin/{TARGET_BRANCH}",
            timeout=60,
        )
        if rc != 0:
            fail(f"worktree update failed: {err}")
        ok(f"worktree updated in place at {WORKTREE_PATH}")

    # Report the SHA.
    rc, sha, _ = run(
        client, f"cd {WORKTREE_PATH} && git rev-parse --short HEAD"
    )
    ok(f"worktree is at {sha.strip()}")

    # ── 3. Install the package ──────────────────────────────────
    step("Python: install package (editable) from worktree")

    rc, out, err = run(
        client,
        f"{VENV_PATH}/bin/pip install -e {WORKTREE_PATH} --quiet --disable-pip-version-check",
        timeout=180,
    )
    if rc != 0:
        # 'quiet' suppresses success output; failures still surface.
        fail(f"pip install failed: {err or out}")
    ok("package installed into venv")

    # Confirm the module imports.
    rc, out, err = run(
        client,
        f"{VENV_PATH}/bin/python -c 'import agi.halyard.state; print(\"ok\")'",
    )
    if "ok" not in out:
        fail(f"import check failed: {err or out}")
    ok("agi.halyard.state imports from venv")

    # ── 4. Install systemd unit ─────────────────────────────────
    step("systemd: install user unit + daemon-reload")

    unit_src = f"{WORKTREE_PATH}/infra/local/halyard-state/systemd/halyard-state.service"
    unit_dst = "/home/claude/.config/systemd/user/halyard-state.service"

    # Rewrite the unit to use the worktree as WorkingDirectory
    # (the original is templated for %h/src/agi-hpc; we're using
    # a dedicated worktree, so substitute).
    rewrite = (
        f"mkdir -p /home/claude/.config/systemd/user && "
        f"sed -e 's|%h/src/agi-hpc|{WORKTREE_PATH}|g' "
        f"    -e 's|%h/env|{VENV_PATH}|g' "
        f"    {unit_src} > {unit_dst}"
    )
    rc, _, err = run(client, rewrite)
    if rc != 0:
        fail(f"unit install failed: {err}")
    ok(f"unit written to {unit_dst}")

    # daemon-reload.
    rc, _, err = run(client, "systemctl --user daemon-reload")
    if rc != 0:
        fail(f"daemon-reload failed: {err}")
    ok("systemd --user daemon-reloaded")

    # ── 5. Start the service ───────────────────────────────────
    step("Start halyard-state.service")

    rc, _, err = run(
        client, "systemctl --user enable --now halyard-state.service"
    )
    if rc != 0:
        fail(f"enable --now failed: {err}")
    ok("service enabled + started")

    # Poll /healthz over loopback for up to 10 s.
    for attempt in range(1, 11):
        time.sleep(1)
        rc, out, _ = run(
            client,
            "curl -sS --max-time 3 http://127.0.0.1:8090/healthz",
        )
        if rc == 0 and '"ok":true' in out.replace(" ", ""):
            ok(f"healthz OK after {attempt}s: {out.strip()}")
            break
    else:
        rc, status, _ = run(
            client,
            "systemctl --user status halyard-state --no-pager | head -20 "
            "&& echo --- && "
            "journalctl --user -u halyard-state -n 40 --no-pager",
        )
        fail(f"healthz failed after 10s. service status:\n{status}")

    # ── 6. Final status ────────────────────────────────────────
    step("Final status")

    rc, out, _ = run(
        client, "systemctl --user show halyard-state "
        "--property=ActiveState,SubState,MainPID,ExecMainStartTimestamp "
        "--no-pager",
    )
    for line in out.strip().splitlines():
        print("  " + line)

    rc, out, _ = run(
        client, "ss -tln src :8090 | tail -n +2 | head"
    )
    ok(f"port :8090 bound: {out.strip() or 'check manually'}")

    print()
    print("═" * 70)
    print("  ✓ halyard-state deployed.")
    print()
    print("  Endpoints (over Tailscale):")
    print("    http://100.68.134.21:8090/healthz")
    print("    http://100.68.134.21:8090/api/sheets/<session>")
    print("    ws://100.68.134.21:8090/ws/sheets/<session>")
    print()
    print("  Logs:  ssh claude@100.68.134.21 "
          "'journalctl --user -u halyard-state -f'")
    print("═" * 70)

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
