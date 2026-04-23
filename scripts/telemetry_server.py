#!/usr/bin/env python3
"""Atlas Telemetry Server — lightweight, always-on monitoring.

Serves /api/telemetry and /api/events independently of the RAG server.
Designed to run during maintenance mode or when the RAG server is down.

No heavy dependencies (no embedding model, no sentence-transformers).
Just system stats, PostgreSQL counts, NATS status, and tmux jobs.

Usage:
    python3 scripts/telemetry_server.py
    python3 scripts/telemetry_server.py --port 8082
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("telemetry")

STATIC_DIR = os.environ.get("ATLAS_STATIC", "/home/claude/atlas-chat")
REPO_DIR = os.environ.get("ATLAS_REPO", "/home/claude/agi-hpc")
_ui_version_cache = {"sha": "", "ts": 0.0}


def _ui_version_stamp(html_path: str) -> str:
    """Return 'SHA · MTIME' for the dashboard footer. Cached for 15s.
    Falls back gracefully if git/stat fail so serving never breaks."""
    try:
        now = time.time()
        if now - _ui_version_cache["ts"] > 15 or not _ui_version_cache["sha"]:
            r = subprocess.run(
                ["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            _ui_version_cache["sha"] = (
                r.stdout.strip() if r.returncode == 0 else "nogit"
            )
            _ui_version_cache["ts"] = now
        sha = _ui_version_cache["sha"]
        try:
            mtime = os.path.getmtime(os.path.realpath(html_path))
            ts = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(mtime))
        except Exception:
            ts = "?"
        return f"{sha} · {ts}"
    except Exception:
        return "?"


PORT = int(os.environ.get("TELEMETRY_PORT", "8085"))
DB_DSN = os.environ.get("ATLAS_DB_DSN", "dbname=atlas user=claude")
SNAPSHOT_INTERVAL = float(os.environ.get("TELEMETRY_SNAPSHOT_S", "2.5"))


# Background snapshot cache: build_telemetry() is expensive (shells out to
# nvidia-smi/sensors/NATS/Postgres). One refresher thread populates this
# every SNAPSHOT_INTERVAL seconds; per-request handlers read it under a
# lock — O(1) per request, so the accept queue can't overflow.
_SNAPSHOT_LOCK = threading.Lock()
_SNAPSHOT = {"telemetry": {}, "events": [], "ts": 0.0}


def _refresher():
    while True:
        try:
            t = build_telemetry()
            ev = _build_events()
            with _SNAPSHOT_LOCK:
                _SNAPSHOT["telemetry"] = t
                _SNAPSHOT["events"] = ev
                _SNAPSHOT["ts"] = time.time()
        except Exception as e:
            log.warning("snapshot refresh failed: %s", e)
        time.sleep(SNAPSHOT_INTERVAL)


def get_cached_telemetry():
    with _SNAPSHOT_LOCK:
        return _SNAPSHOT["telemetry"] or build_telemetry()


def get_cached_events():
    with _SNAPSHOT_LOCK:
        return _SNAPSHOT["events"] or _build_events()


def _run(cmd, timeout=3):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def _get_gpu():
    gpus = []
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    for line in out.split("\n"):
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 6:
            gpus.append(
                {
                    "index": int(p[0]),
                    "name": p[1],
                    "temp": int(p[2]),
                    "util": int(p[3]),
                    "mem_used": int(p[4]),
                    "mem_total": int(p[5]),
                }
            )
    return gpus


def _get_cpu():
    out = _run(["sensors"])
    pkgs = []
    for line in out.split("\n"):
        if "Package" in line:
            try:
                pkgs.append(float(line.split("+")[1].split("\u00b0")[0].split(".")[0]))
            except (IndexError, ValueError):
                pass
    return {"package_temps": pkgs, "max_temp": max(pkgs) if pkgs else 0}


def _get_ram():
    out = _run(["free", "-b"])
    result = {}
    for line in out.split("\n"):
        if line.startswith("Mem:"):
            p = line.split()
            total = int(p[1])
            used = int(p[2])
            free = int(p[3])
            shared = int(p[4])
            buff_cache = int(p[5])
            available = int(p[6])
            g = 1073741824
            result = {
                "total_gb": round(total / g, 1),
                "used_gb": round(used / g, 1),
                "free_gb": round(free / g, 1),
                "shared_gb": round(shared / g, 1),
                "buff_cache_gb": round(buff_cache / g, 1),
                "available_gb": round(available / g, 1),
                # Percentages for stacked bar
                "used_pct": round(used / total * 100, 1),
                "buff_cache_pct": round(buff_cache / total * 100, 1),
                "free_pct": round(free / total * 100, 1),
            }
        elif line.startswith("Swap:"):
            p = line.split()
            g = 1073741824
            result["swap_total_gb"] = round(int(p[1]) / g, 1)
            result["swap_used_gb"] = round(int(p[2]) / g, 1)
    return result


def _get_system_deep():
    """Deep system metrics — 747 cockpit data."""
    result = {}

    # File descriptors
    try:
        fd = _run(["cat", "/proc/sys/fs/file-nr"])
        parts = fd.split()
        if len(parts) >= 3:
            result["fd_allocated"] = int(parts[0])
            result["fd_max"] = int(parts[2])
    except Exception:
        pass

    # TCP/UDP connections
    try:
        ss = _run(["ss", "-s"])
        for line in ss.split("\n"):
            if "TCP:" in line:
                import re

                nums = re.findall(r"(\d+)", line)
                if len(nums) >= 3:
                    result["tcp_total"] = int(nums[0])
                    result["tcp_estab"] = int(nums[1])
                    result["tcp_closed"] = int(nums[2])
    except Exception:
        pass

    # Load average
    try:
        la = _run(["cat", "/proc/loadavg"])
        parts = la.split()
        if len(parts) >= 3:
            result["load_1m"] = float(parts[0])
            result["load_5m"] = float(parts[1])
            result["load_15m"] = float(parts[2])
            result["running_threads"] = parts[3]
    except Exception:
        pass

    # Context switches
    try:
        stat = _run(["cat", "/proc/stat"])
        for line in stat.split("\n"):
            if line.startswith("ctxt"):
                result["context_switches"] = int(line.split()[1])
            elif line.startswith("processes"):
                result["total_forks"] = int(line.split()[1])
    except Exception:
        pass

    # Disk usage
    try:
        df = _run(["df", "-BG", "/", "/mnt/raid5"])
        for line in df.split("\n")[1:]:
            parts = line.split()
            if len(parts) >= 5:
                mount = parts[-1]
                key = "root" if mount == "/" else "raid5"
                result[f"disk_{key}_total_gb"] = int(parts[1].rstrip("G"))
                result[f"disk_{key}_used_gb"] = int(parts[2].rstrip("G"))
                result[f"disk_{key}_avail_gb"] = int(parts[3].rstrip("G"))
                result[f"disk_{key}_pct"] = parts[4]
    except Exception:
        pass

    # Network I/O (bytes since boot)
    try:
        net = _run(["cat", "/proc/net/dev"])
        for line in net.split("\n"):
            if "eno1" in line or "eth0" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    nums = parts[1].split()
                    result["net_rx_gb"] = round(int(nums[0]) / 1073741824, 2)
                    result["net_tx_gb"] = round(int(nums[8]) / 1073741824, 2)
    except Exception:
        pass

    # Uptime
    try:
        up = _run(["cat", "/proc/uptime"])
        secs = float(up.split()[0])
        days = int(secs // 86400)
        hours = int((secs % 86400) // 3600)
        result["uptime_days"] = days
        result["uptime_hours"] = hours
        result["uptime_str"] = f"{days}d {hours}h"
    except Exception:
        pass

    # Process/thread counts
    try:
        import os

        result["process_count"] = len(os.listdir("/proc"))
    except Exception:
        pass

    # Zombie count
    try:
        z = _run(["bash", "-c", "ps aux | awk '{print $8}' | grep -c Z"])
        result["zombies"] = int(z) if z else 0
    except Exception:
        result["zombies"] = 0

    return result


def _get_db_counts():
    counts = {
        "semantic_chunks": 0,
        "repos": 0,
        "episodic_episodes": 0,
        "ethics_chunks": 0,
        "publications": 0,
    }
    try:
        import psycopg2

        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        for key, query in [
            ("semantic_chunks", "SELECT COUNT(*) FROM chunks"),
            ("repos", "SELECT COUNT(DISTINCT repo) FROM chunks"),
            ("ethics_chunks", "SELECT COUNT(*) FROM ethics_chunks"),
            ("publications", "SELECT COUNT(*) FROM publications"),
            ("episodic_episodes", "SELECT COUNT(*) FROM episodes"),
        ]:
            try:
                cur.execute(query)
                counts[key] = cur.fetchone()[0]
            except Exception:
                conn.rollback()
        conn.close()
    except Exception:
        pass
    return counts


def _get_tqpro_stats():
    """Get TurboQuant Pro compression statistics."""
    stats = {"enabled": False, "tables": {}}
    try:
        import psycopg2

        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        for table in ["chunks", "episodes", "ethics_chunks"]:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL")
                total = cur.fetchone()[0]
                cur.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE embedding_pca384 IS NOT NULL"
                )
                compressed = cur.fetchone()[0]
                if compressed > 0:
                    stats["enabled"] = True
                    orig_mb = round(total * 1024 * 4 / 1048576, 1)
                    comp_mb = round(compressed * 384 * 4 / 1048576, 1)
                    stats["tables"][table] = {
                        "total": total,
                        "compressed": compressed,
                        "original_mb": orig_mb,
                        "compressed_mb": comp_mb,
                        "saved_mb": round(orig_mb - comp_mb, 1),
                        "ratio": 2.67,
                    }
            except Exception:
                conn.rollback()
        if stats["enabled"]:
            total_orig = sum(t["original_mb"] for t in stats["tables"].values())
            total_comp = sum(t["compressed_mb"] for t in stats["tables"].values())
            stats["total_original_mb"] = round(total_orig, 1)
            stats["total_compressed_mb"] = round(total_comp, 1)
            stats["total_saved_mb"] = round(total_orig - total_comp, 1)
            stats["method"] = "PCA-Matryoshka 384d"
            stats["cosine_similarity"] = 0.990
        conn.close()
    except Exception:
        pass
    return stats


def _get_safety_from_rag(base_status):
    """Get safety stats from the RAG server's telemetry."""
    try:
        import urllib.request

        r = urllib.request.urlopen("http://localhost:8081/api/telemetry", timeout=3)
        data = json.loads(r.read())
        safety = data.get("safety", {})
        safety["status"] = base_status  # Use port-check status
        return safety
    except Exception:
        return {"status": base_status, "vetoes": 0}


def _get_nats():
    try:
        import urllib.request

        r = urllib.request.urlopen("http://localhost:8222/varz", timeout=2)
        data = json.loads(r.read())
        result = {
            "status": "online",
            "in_msgs": data.get("in_msgs", 0),
            "out_msgs": data.get("out_msgs", 0),
            "connections": data.get("connections", 0),
            "uptime": data.get("uptime", ""),
        }
        try:
            r2 = urllib.request.urlopen("http://localhost:8222/jsz", timeout=2)
            js = json.loads(r2.read())
            result["jetstream"] = {
                "streams": js.get("streams", 0),
                "messages": js.get("messages", 0),
                "bytes": js.get("bytes", 0),
            }
        except Exception:
            pass
        return result
    except Exception:
        return {"status": "offline"}


def _check_port(port, timeout=1):
    """Check if a TCP port is listening."""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(("localhost", port))
        s.close()
        return True
    except Exception:
        return False


def _get_services():
    services = {
        "lh": "offline",
        "rh": "offline",
        "ego": "offline",
        "safety": "offline",
        "metacognition": "offline",
        "dht": "offline",
    }
    # Check LLM hemispheres by port
    if _check_port(8080):
        services["lh"] = "online"
    if _check_port(8082):
        services["rh"] = "online"
    elif services["lh"] == "online":
        services["rh"] = "fallback"  # Id role-played by Superego model
    if _check_port(8084):
        services["ego"] = "online"
    # Safety + Metacognition are integrated into the RAG server (8081)
    # They're online whenever the RAG server is online
    if _check_port(8081):
        services["safety"] = "online"
        services["metacognition"] = "online"
    # DHT is online when NATS is up (service registry uses NATS)
    if _check_port(4222):
        services["dht"] = "online"
    # Also check tmux sessions as fallback
    for name, session in [
        ("safety", "safety"),
        ("metacognition", "metacognition"),
        ("dht", "dht"),
    ]:
        if services[name] != "online":
            r = subprocess.run(
                ["tmux", "has-session", "-t", session], capture_output=True, timeout=2
            )
            if r.returncode == 0:
                services[name] = "online"
    return services


def _get_jobs():
    jobs = []
    out = _run(["tmux", "ls"], timeout=3)
    for line in out.split("\n"):
        if ":" not in line:
            continue
        name = line.split(":")[0].strip()
        job = {"name": name, "status": "running", "description": name}

        pane_pid = _run(["tmux", "list-panes", "-t", name, "-F", "#{pane_pid}"])
        # Try children first, then pane process itself
        ps_out = _run(
            [
                "ps",
                "--ppid",
                pane_pid,
                "-o",
                "pid,pcpu,rss,etime,comm",
                "--no-headers",
            ]
        )
        if not ps_out and pane_pid:
            # Shell may have exec'd — check the pane PID itself
            ps_out = _run(
                [
                    "ps",
                    "-p",
                    pane_pid,
                    "-o",
                    "pid,pcpu,rss,etime,comm",
                    "--no-headers",
                ]
            )
        if ps_out:
            # Take first line only (may have multiple children)
            first_line = ps_out.strip().split("\n")[0]
            parts = first_line.split()
            if len(parts) >= 5:
                job["cpu"] = parts[1] + "%"
                try:
                    job["mem"] = str(round(int(parts[2]) / 1024)) + "MB"
                except ValueError:
                    pass
                job["elapsed"] = parts[3]

        desc_map = {
            "spock": ("Superego: Gemma 4 31B", 0),
            "kirk": ("Id: Qwen 3 32B", 1),
            "ego": ("Council Judge: Gemma 4 26B-A4B", None),
            "tot-worker-0": ("Council Advocate: 26B-A4B", None),
            "tot-worker-1": ("Council Synthesizer: 26B-A4B", None),
            "tot-worker-2": ("Council Ethicist: 26B-A4B", None),
            "nats": ("NATS JetStream", None),
            "rag": ("RAG Server", None),
            "caddy": ("HTTPS / Let's Encrypt", None),
            "oauth2": ("Google OAuth", None),
            "safety": ("Safety Gateway", None),
            "memory": ("Memory Service", None),
            "dht": ("DHT Registry", None),
            "train": ("Training", None),
            "indexer": ("RAG Indexer", None),
            "embed": ("Ethics Embedding", 1),
        }
        if name in desc_map:
            desc, gpu = desc_map[name]
            job["description"] = desc
            if gpu is not None:
                job["gpu"] = gpu
        elif name.startswith("dl-"):
            job["description"] = f"Download: {name[3:]}"

        jobs.append(job)

    # Add systemd-managed services (not in tmux)
    tmux_names = {j["name"] for j in jobs}
    systemd_services = [
        ("atlas-superego", "Superego: Gemma 4 31B", 0),
        ("atlas-id", "Id: Qwen 3 32B", 1),
        ("atlas-ego", "Council Judge: Gemma 4 26B-A4B", None),
        ("atlas-rag-server", "RAG Server", None),
        ("atlas-nats", "NATS JetStream", None),
        ("atlas-telemetry", "Telemetry", None),
        ("atlas-watchdog", "Watchdog", None),
    ]
    for svc, desc, gpu in systemd_services:
        # Skip if already in tmux
        short = svc.replace("atlas-", "")
        if short in tmux_names or svc in tmux_names:
            continue
        # Check if systemd service is active
        try:
            r = subprocess.run(
                ["systemctl", "is-active", svc],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if r.stdout.strip() == "active":
                job = {
                    "name": short,
                    "status": "running",
                    "description": desc,
                }
                if gpu is not None:
                    job["gpu"] = gpu
                # Get process stats
                pid_out = _run(
                    [
                        "systemctl",
                        "show",
                        svc,
                        "--property=MainPID",
                        "--value",
                    ]
                )
                if pid_out and pid_out != "0":
                    ps = _run(
                        [
                            "ps",
                            "-p",
                            pid_out,
                            "-o",
                            "pcpu,rss,etime",
                            "--no-headers",
                        ]
                    )
                    if ps:
                        parts = ps.split()
                        if len(parts) >= 3:
                            job["cpu"] = parts[0] + "%"
                            job["mem"] = str(round(int(parts[1]) / 1024)) + "MB"
                            job["elapsed"] = parts[2]
                jobs.append(job)
        except Exception:
            pass

    return jobs


def _get_rag_telemetry():
    """Fetch enriched telemetry from the RAG server (short timeout)."""
    try:
        import urllib.request

        r = urllib.request.urlopen("http://localhost:8081/api/telemetry", timeout=2)
        return json.loads(r.read())
    except Exception:
        return {}


def _count_wiki_articles():
    """Count dream-consolidated wiki articles."""
    try:
        wiki = Path("/home/claude/agi-hpc/wiki")
        if wiki.exists():
            return len(list(wiki.glob("dream-*.md")))
    except Exception:
        pass
    return 0


def _get_training_stats():
    """Get training stats from PostgreSQL."""
    stats = {"total_sessions": 0}
    try:
        import psycopg2

        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT COUNT(*), AVG(score), MAX(timestamp) " "FROM training_results"
            )
            row = cur.fetchone()
            if row and row[0] > 0:
                stats["total_sessions"] = int(row[0])
                stats["last_session_score"] = round(float(row[1] or 0), 3)
                stats["last_training"] = str(row[2]) if row[2] else None
        except Exception:
            conn.rollback()

        # Count retrospective vs pantheon vs LLM scenarios
        try:
            cur.execute(
                "SELECT metadata->>'type', COUNT(*) "
                "FROM episodes "
                "WHERE metadata->>'type' = 'dm_training' "
                "GROUP BY metadata->>'type'"
            )
            row = cur.fetchone()
            if row:
                stats["dm_training_episodes"] = int(row[1])
        except Exception:
            conn.rollback()

        # Count retrospective episodes
        try:
            cur.execute(
                "SELECT COUNT(*) FROM episodes "
                "WHERE metadata->>'retrospective_used' = 'true'"
            )
            stats["retrospective_used"] = cur.fetchone()[0]
        except Exception:
            conn.rollback()

        # Daily score history (last 30 days)
        try:
            cur.execute("""
                SELECT DATE(timestamp) AS day,
                       COUNT(*) AS episodes,
                       ROUND(AVG(score)::numeric, 3) AS avg_score,
                       ROUND(MIN(score)::numeric, 3) AS min_score,
                       ROUND(MAX(score)::numeric, 3) AS max_score
                FROM training_results
                WHERE timestamp >= NOW() - INTERVAL '30 days'
                GROUP BY DATE(timestamp)
                ORDER BY day
            """)
            stats["daily_history"] = [
                {
                    "date": str(r[0]),
                    "episodes": r[1],
                    "avg_score": float(r[2]) if r[2] else 0,
                    "min_score": float(r[3]) if r[3] else 0,
                    "max_score": float(r[4]) if r[4] else 0,
                }
                for r in cur.fetchall()
            ]
        except Exception:
            conn.rollback()
            stats["daily_history"] = []

        # Per-domain breakdown (from metadata)
        try:
            cur.execute("""
                SELECT metadata->>'domain' AS domain,
                       COUNT(*) AS eps,
                       ROUND(AVG(score)::numeric, 3) AS avg
                FROM training_results
                WHERE metadata->>'domain' IS NOT NULL
                GROUP BY metadata->>'domain'
                ORDER BY avg DESC
            """)
            stats["domain_breakdown"] = [
                {"domain": r[0], "episodes": r[1], "avg_score": float(r[2])}
                for r in cur.fetchall()
            ]
        except Exception:
            conn.rollback()
            stats["domain_breakdown"] = []

        # Score trend (last 20 scores for sparkline)
        try:
            cur.execute(
                "SELECT score FROM training_results " "ORDER BY timestamp DESC LIMIT 20"
            )
            stats["recent_scores"] = [round(float(r[0]), 3) for r in cur.fetchall()]
        except Exception:
            conn.rollback()
            stats["recent_scores"] = []

        # Check training timer status
        try:
            import subprocess

            r = subprocess.run(
                ["systemctl", "--user", "is-active", "atlas-training.timer"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            stats["timer_active"] = r.stdout.strip() == "active"
            # Next trigger time
            r2 = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    "atlas-training.timer",
                    "--property=NextElapseUSecRealtime",
                    "--value",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if r2.stdout.strip():
                stats["next_training"] = r2.stdout.strip()
        except Exception:
            stats["timer_active"] = False

        conn.close()
    except Exception:
        pass
    return stats


def _get_training_history(days=30):
    """Full training history for /api/training/history endpoint."""
    result = {"daily": [], "by_env": {}, "by_domain": [], "totals": {}}
    try:
        import psycopg2

        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()

        # Daily aggregates
        cur.execute("""
            SELECT DATE(timestamp) AS day,
                   COUNT(*), ROUND(AVG(score)::numeric, 3),
                   ROUND(MIN(score)::numeric, 3),
                   ROUND(MAX(score)::numeric, 3)
            FROM training_results
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY DATE(timestamp)
            ORDER BY day
        """ % int(days))
        result["daily"] = [
            {
                "date": str(r[0]),
                "episodes": r[1],
                "avg": float(r[2]),
                "min": float(r[3]),
                "max": float(r[4]),
            }
            for r in cur.fetchall()
        ]

        # Per-environment history
        cur.execute("""
            SELECT env_name, DATE(timestamp) AS day,
                   COUNT(*), ROUND(AVG(score)::numeric, 3)
            FROM training_results
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY env_name, DATE(timestamp)
            ORDER BY env_name, day
        """ % int(days))
        for r in cur.fetchall():
            env = r[0]
            if env not in result["by_env"]:
                result["by_env"][env] = []
            result["by_env"][env].append(
                {"date": str(r[1]), "episodes": r[2], "avg": float(r[3])}
            )

        # Domain averages
        cur.execute("""
            SELECT metadata->>'domain', COUNT(*),
                   ROUND(AVG(score)::numeric, 3)
            FROM training_results
            WHERE metadata->>'domain' IS NOT NULL
            GROUP BY metadata->>'domain'
            ORDER BY AVG(score) DESC
        """)
        result["by_domain"] = [
            {"domain": r[0], "episodes": r[1], "avg": float(r[2])}
            for r in cur.fetchall()
        ]

        # Totals
        cur.execute(
            "SELECT COUNT(*), ROUND(AVG(score)::numeric, 3) " "FROM training_results"
        )
        row = cur.fetchone()
        result["totals"] = {
            "episodes": row[0] if row else 0,
            "avg_score": float(row[1]) if row and row[1] else 0,
        }

        # Promotion/demotion counts
        cur.execute("""
            SELECT metadata->>'curriculum_action', COUNT(*)
            FROM training_results
            WHERE metadata->>'curriculum_action' IS NOT NULL
            GROUP BY metadata->>'curriculum_action'
        """)
        for r in cur.fetchall():
            result["totals"][r[0] + "s"] = r[1]

        conn.close()
    except Exception:
        pass
    return result


def _get_dreaming_stats():
    """Get dreaming/consolidation statistics."""
    stats = {"wiki_articles": 0, "dream_insights": 0}
    try:
        wiki = Path("/home/claude/agi-hpc/wiki")
        if wiki.exists():
            articles = list(wiki.glob("dream-*.md"))
            stats["wiki_articles"] = len(articles)
            insights = list(wiki.glob("dream-insight-*.md"))
            stats["dream_insights"] = len(insights)

            # Read latest article grade if available
            if articles:
                latest = sorted(articles)[-1]
                content = latest.read_text(encoding="utf-8")
                for grade in ["A", "B", "C", "D"]:
                    if f"**{grade}**" in content:
                        stats["latest_grade"] = grade
                        break
    except Exception:
        pass
    return stats


def _get_knowledge_stats():
    """Get knowledge graph statistics."""
    try:
        from agi.memory.knowledge.graph import KnowledgeGraph, KnowledgeGraphConfig

        graph = KnowledgeGraph(KnowledgeGraphConfig(db_dsn=DB_DSN))
        stats = graph.get_stats()
        lint = graph.lint()
        stats["contradictions"] = len(lint.get("contradictions", []))
        return stats
    except Exception:
        return {"entities": 0, "relationships": 0, "documents": 0, "contradictions": 0}


def _get_research_stats():
    """Get research loop telemetry (read from last saved state)."""
    try:
        state_file = Path("/home/claude/agi-hpc/data/research_telemetry.json")
        if state_file.exists():
            import json

            return json.loads(state_file.read_text())
    except Exception:
        pass
    return {
        "goals_detected": 0,
        "goals_completed": 0,
        "knowledge_added": 0,
        "avg_confidence": 0,
        "last_cycle": None,
    }


def _get_curriculum_gaps():
    """Get knowledge gap summary (read-only)."""
    try:
        from agi.metacognition.curriculum_planner import CurriculumPlanner

        planner = CurriculumPlanner(db_dsn=DB_DSN, lookback_episodes=50)
        plan = planner.analyze()
        return {
            "gaps_detected": len(plan.gaps),
            "focus_domains": plan.domains_to_focus[:3],
            "recommended_scenarios": plan.recommended_scenarios,
            "episodes_analyzed": plan.total_episodes_analyzed,
        }
    except Exception:
        return {"gaps_detected": 0, "focus_domains": []}


def build_telemetry():
    gpus = _get_gpu()
    services = _get_services()
    memory = _get_db_counts()
    online = sum(1 for v in services.values() if v == "online")
    total_services = len(services)

    # Enrich with RAG server data (safety stats, privileges)
    rag_data = _get_rag_telemetry()
    rag_safety = rag_data.get("safety", {})
    rag_privs = rag_data.get("ego_privileges", {})

    # Wiki article count
    memory["wiki_articles"] = _count_wiki_articles()

    # Unconsolidated episode count
    try:
        import psycopg2

        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT COUNT(*) FROM episodes "
                "WHERE metadata->>'consolidated' IS NULL "
                "OR metadata->>'consolidated' = 'false'"
            )
            memory["unconsolidated_episodes"] = cur.fetchone()[0]
        except Exception:
            conn.rollback()
        conn.close()
    except Exception:
        pass

    # Safety: merge RAG server's live stats with port-check status
    safety = {
        "status": services["safety"],
        "layer": rag_safety.get("layer", "reflex"),
        "input_checks": rag_safety.get("input_checks", 0),
        "output_checks": rag_safety.get("output_checks", 0),
        "vetoes": rag_safety.get("vetoes", 0),
        "avg_latency_ms": rag_safety.get("avg_latency_ms", 0),
        "audit_log_size": rag_safety.get("audit_log_size", 0),
    }

    return {
        "timestamp": time.time(),
        "hemispheres": {
            "lh": {
                "status": services["lh"],
                "model": "Gemma 4 31B",
                "role": "Superego (analytical)",
            },
            "rh": {
                "status": services["rh"],
                "model": "Qwen 3 32B",
                "role": "Id (creative)",
                "note": (
                    "Role-played by Superego" if services["rh"] == "fallback" else ""
                ),
            },
            "ego": {
                "status": services["ego"],
                "model": "Gemma 4 26B-A4B",
                "role": "Divine Council (7 agents, 1 server, --parallel 8)",
            },
        },
        "nats": _get_nats(),
        "memory": memory,
        "safety": safety,
        "metacognition": {"status": services["metacognition"]},
        "environment": {
            "gpu": gpus,
            "cpu": _get_cpu(),
            "ram": _get_ram(),
        },
        "integration": {
            "status": "online" if services.get("safety") == "online" else "offline",
            "sessions": 0,
            "routed": 0,
        },
        "dht": {
            "status": services["dht"],
            "services_online": online,
            "services_total": total_services,
        },
        "ego_privileges": (
            rag_privs
            if rag_privs
            else {
                "current_level": 0,
                "level_name": "READ_ONLY",
            }
        ),
        "system_deep": _get_system_deep(),
        "attention": rag_data.get("attention", {"checks": 0, "last_intensity": "none"}),
        "training": _get_training_stats(),
        "dreaming": _get_dreaming_stats(),
        "knowledge": _get_knowledge_stats(),
        "research": _get_research_stats(),
        "curriculum": _get_curriculum_gaps(),
        "jobs": _get_jobs(),
        "turboquant": _get_tqpro_stats(),
    }


def _get_visitors(limit=50):
    """Fetch recent visitor log entries from PostgreSQL."""
    visitors = []
    try:
        import psycopg2

        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        cur.execute(
            "SELECT email, ip, path, user_agent, timestamp "
            "FROM visitor_log ORDER BY timestamp DESC LIMIT %s",
            (limit,),
        )
        for row in cur.fetchall():
            visitors.append(
                {
                    "email": row[0] or "",
                    "ip": row[1] or "",
                    "path": row[2] or "",
                    "user_agent": row[3] or "",
                    "timestamp": row[4].isoformat() if row[4] else "",
                    "type": "login",
                }
            )
        cur.execute("SELECT COUNT(DISTINCT email) FROM visitor_log")
        unique = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM visitor_log")
        total = cur.fetchone()[0]
        conn.close()
        return {"unique_visitors": unique, "total_visits": total, "recent": visitors}
    except Exception:
        return {"unique_visitors": 0, "total_visits": 0, "recent": []}


def _log_visitor(email, ip, path, user_agent):
    """Log a visitor to PostgreSQL."""
    try:
        import psycopg2

        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO visitor_log (email, ip, path, user_agent)"
            " VALUES (%s, %s, %s, %s)",
            (email, ip, path, user_agent[:200] if user_agent else ""),
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def _build_events():
    """Build events feed including login/logout and system events."""
    events = []
    # Recent logins
    visitor_data = _get_visitors(limit=20)
    for v in visitor_data.get("recent", []):
        email = v.get("email", "unknown")
        name = email.split("@")[0] if "@" in email else email
        events.append(
            {
                "type": "login",
                "message": f"{name} logged in",
                "detail": v.get("path", "/"),
                "timestamp": v.get("timestamp", ""),
                "color": "var(--green)",
            }
        )
    # NATS stats as system events
    nats = _get_nats()
    if nats.get("status") == "online":
        events.append(
            {
                "type": "system",
                "message": (
                    f"NATS: {nats.get('in_msgs', 0)} msgs in, "
                    f"{nats.get('connections', 0)} connections"
                ),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "color": "var(--accent)",
            }
        )
    return events


_nrp_cache = {"data": {}, "ts": 0}

# ── NRP NATS telemetry subscriber ──────────────────────────────
# Background thread subscribes to nrp.> on local NATS so the leaf
# node bridges those subjects from NRP. Incoming messages are cached
# and served via /api/nrp-telemetry.
_nrp_nats_cache = {"heartbeats": [], "pods": {}, "last_heartbeat": 0}
_nrp_nats_lock = threading.Lock()


def _nrp_nats_listener():
    """Background thread: subscribe to nrp.> on Atlas NATS."""
    import socket as _socket

    while True:
        try:
            s = _socket.create_connection(("localhost", 4222), 10)
            s.settimeout(90)  # longer than heartbeat interval
            s.recv(4096)  # INFO
            s.sendall(b'CONNECT {"verbose":false}\r\n')
            s.sendall(b"SUB nrp.> 1\r\n")
            s.sendall(b"PING\r\n")
            s.recv(1024)
            log.info("[nrp-nats] subscribed to nrp.>")

            buf = b""
            while True:
                data = s.recv(4096)
                if not data:
                    break
                buf += data
                while b"\r\n" in buf:
                    line, buf = buf.split(b"\r\n", 1)
                    if line.startswith(b"MSG"):
                        parts = line.decode().split()
                        subject = parts[1]
                        size = int(parts[-1])
                        while len(buf) < size + 2:
                            buf += s.recv(4096)
                        payload = buf[:size].decode(errors="replace")
                        buf = buf[size + 2 :]
                        try:
                            msg = json.loads(payload)
                        except Exception:
                            msg = {"raw": payload[:200]}
                        with _nrp_nats_lock:
                            if subject == "nrp.heartbeat":
                                _nrp_nats_cache["last_heartbeat"] = time.time()
                                _nrp_nats_cache["heartbeats"].append(msg)
                                _nrp_nats_cache["heartbeats"] = _nrp_nats_cache[
                                    "heartbeats"
                                ][-10:]
                            elif subject == "nrp.pods":
                                _nrp_nats_cache["pods"] = msg
                            log.info("[nrp-nats] %s: %s", subject, str(msg)[:120])
                    elif line == b"PING":
                        s.sendall(b"PONG\r\n")
        except Exception as e:
            log.warning("[nrp-nats] disconnected: %s — reconnecting in 10s", e)
        time.sleep(10)


def _get_nrp_nats_telemetry():
    with _nrp_nats_lock:
        age = (
            time.time() - _nrp_nats_cache["last_heartbeat"]
            if _nrp_nats_cache["last_heartbeat"]
            else None
        )
        return {
            "leaf_alive": age is not None and age < 120,
            "last_heartbeat_age_s": round(age, 1) if age else None,
            "heartbeats": list(_nrp_nats_cache["heartbeats"]),
            "pods": dict(_nrp_nats_cache["pods"]),
        }


# ── Erebus activity stream ───────────────────────────────────
# Ring buffer of the last ~500 log lines from arc_scientist /
# onnx_scientist / dreaming_schedule, each tagged with a monotonic seq.
# The chat UI sidebar polls /api/erebus/activity?since=SEQ for new entries.
import collections as _collections  # noqa: E402 (grouped with other local state)
import re as _re  # noqa: E402

_EREBUS_ACTIVITY_LOGS = [
    ("scientist", "/archive/neurogolf/scientist.log"),
    ("onnx", "/archive/neurogolf/onnx_scientist.log"),
    ("dream", "/archive/neurogolf/dreaming_schedule.log"),
]
_EREBUS_ACTIVITY_MAX = 500
_EREBUS_ACTIVITY = _collections.deque(maxlen=_EREBUS_ACTIVITY_MAX)
_EREBUS_ACTIVITY_SEQ = {"n": 0}
_EREBUS_ACTIVITY_LOCK = threading.Lock()


def _classify_activity(line: str) -> str:
    """Color-code line by kind. Order matters — first match wins."""
    lo = line.lower()
    if "-> solved" in lo or ("verified" in lo and "true" in lo):
        return "solved"
    if "help requested" in lo or "help_requested" in lo:
        return "help"
    if (
        "meta-pattern" in lo
        or "strategy performance" in lo
        or "fingerprints:" in lo
        or "cluster " in lo
    ):
        return "meta"
    if (
        "microsleep" in lo
        or "mediumsleep" in lo
        or "deepsleep" in lo
        or "qlora" in lo
        or "dream cycle" in lo
        or "dream tiers" in lo
    ):
        return "dream"
    if "traceback" in lo or "exception" in lo or lo.startswith("error"):
        return "error"
    if _re.search(r"\[\d+/\d+\]", line) or "-> " in line:
        return "attempt"
    return "info"


def _seed_activity_from_tail(path: str, source: str, n_lines: int = 30) -> None:
    """Pre-populate the ring buffer with the last N lines of a log so the
    UI has context immediately instead of waiting for new activity."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read = min(size, 16384)
            f.seek(size - read)
            tail = f.read(read).decode("utf-8", errors="replace")
        lines = [ln for ln in tail.splitlines() if ln.strip()][-n_lines:]
        with _EREBUS_ACTIVITY_LOCK:
            for ln in lines:
                _EREBUS_ACTIVITY_SEQ["n"] += 1
                _EREBUS_ACTIVITY.append(
                    {
                        "seq": _EREBUS_ACTIVITY_SEQ["n"],
                        "ts": time.time(),
                        "source": source,
                        "text": ln[:500],
                        "kind": _classify_activity(ln),
                    }
                )
    except FileNotFoundError:
        pass
    except Exception as _e:
        log.debug(f"[activity-seed] {path}: {_e}")


def _tail_erebus_logs():
    """Background thread: tail the 3 scientist logs, append to ring buffer."""
    for source, path in _EREBUS_ACTIVITY_LOGS:
        _seed_activity_from_tail(path, source, n_lines=30)

    positions: dict[str, int] = {}
    # Seed positions at each file's current end so we only show new activity
    for _, path in _EREBUS_ACTIVITY_LOGS:
        try:
            positions[path] = os.path.getsize(path)
        except OSError:
            positions[path] = 0

    while True:
        for source, path in _EREBUS_ACTIVITY_LOGS:
            try:
                size = os.path.getsize(path)
                if size < positions.get(path, 0):
                    # Log rotated/truncated — restart from 0.
                    positions[path] = 0
                if size > positions[path]:
                    with open(path, "rb") as f:
                        f.seek(positions[path])
                        chunk = f.read(size - positions[path])
                    positions[path] = size
                    text = chunk.decode("utf-8", errors="replace")
                    for raw in text.splitlines():
                        line = raw.strip()
                        if not line:
                            continue
                        with _EREBUS_ACTIVITY_LOCK:
                            _EREBUS_ACTIVITY_SEQ["n"] += 1
                            _EREBUS_ACTIVITY.append(
                                {
                                    "seq": _EREBUS_ACTIVITY_SEQ["n"],
                                    "ts": time.time(),
                                    "source": source,
                                    "text": line[:500],
                                    "kind": _classify_activity(line),
                                }
                            )
            except FileNotFoundError:
                continue
            except Exception as _e:
                log.debug(f"[activity-tail] {path}: {_e}")
        time.sleep(1.0)


def _get_erebus_activity(since: int = 0, limit: int = 200) -> dict:
    with _EREBUS_ACTIVITY_LOCK:
        rows = [e for e in _EREBUS_ACTIVITY if e["seq"] > since][-limit:]
        seq = _EREBUS_ACTIVITY_SEQ["n"]
    return {"seq": seq, "entries": rows}


# ── NRP utilization watchdog ──────────────────────────────────
# NRP Cluster Policy:
#   - Max 4 pods with GPU>40% / CPU 20-200% / RAM 20-150%
#   - 5+ pods: ALL must stay under those thresholds
#   - Pods IDLE relative to their request = violation (wasting resources)
#   - Ignored: Memory <=2GB, CPU <=1
#   - Jobs with sleep = ban
#
# Watchdog does TWO things:
#   1. KILL pods that are under-utilizing requested resources
#   2. COUNT violations and enforce the 4-pod limit


def _get_pod_logs(pod_name: str, tail: int = 8) -> dict:
    """Fetch recent logs from a pod via kubectl."""
    kubeconfig = os.path.expanduser("~/.kube/config")
    ns = "ssu-atlas-ai"
    try:
        result = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                kubeconfig,
                "-n",
                ns,
                "logs",
                pod_name,
                f"--tail={tail}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return {"pod": pod_name, "logs": result.stdout.strip().split("\n")}
        return {"pod": pod_name, "error": result.stderr.strip()[:200]}
    except Exception as e:
        return {"pod": pod_name, "error": str(e)[:200]}


_nrp_violations: dict[str, int] = {}
_nrp_mode: dict = {"mode": "auto", "active": "unknown", "n_active": 0}


def _parse_cpu_m(val):
    if not val:
        return 0
    val = str(val).strip()
    return int(val[:-1]) if val.endswith("m") else int(float(val) * 1000)


def _parse_mem_mi(val):
    if not val:
        return 0
    val = str(val).strip().replace(" ", "")
    if val.endswith("Mi"):
        return int(val[:-2])
    if val.endswith("Gi"):
        return int(float(val[:-2]) * 1024)
    if val.endswith("GB"):
        return int(float(val[:-2]) * 1024)
    return 0


def _pod_is_violating(pod) -> str | None:
    """Check if a pod violates NRP utilization rules.

    Returns violation description string, or None if compliant.
    A pod violates if it's under-utilizing resources it requested
    (outside the ignored ranges).
    """
    usage = pod.get("usage", {})
    gpu_live = pod.get("gpu_live", {})
    res = pod.get("resources", {})

    # GPU: if requested, must be >40%
    if res.get("gpu") and gpu_live.get("gpu_util_pct") is not None:
        if gpu_live["gpu_util_pct"] <= 40:
            return f"GPU {gpu_live['gpu_util_pct']}% (need >40%)"

    # CPU: if requested >1 CPU, usage must be 20-200% of request
    cpu_req_m = _parse_cpu_m(res.get("cpu", ""))
    cpu_used_m = _parse_cpu_m(usage.get("cpu_used", ""))
    if cpu_req_m > 1000:  # >1 CPU requested (not in ignored range)
        if cpu_used_m == 0:
            return f"CPU 0% of {cpu_req_m}m requested"
        cpu_pct = cpu_used_m * 100 // cpu_req_m
        if cpu_pct < 20:
            return f"CPU {cpu_pct}% (need 20-200%)"
        if cpu_pct > 200:
            return f"CPU {cpu_pct}% (need 20-200%)"

    # RAM: if requested >2GB, usage must be 20-150% of request
    mem_req_mi = _parse_mem_mi(res.get("memory", ""))
    mem_used_mi = _parse_mem_mi(usage.get("mem_used", ""))
    if mem_req_mi > 2048:  # >2GB requested (not in ignored range)
        if mem_used_mi == 0:
            return f"RAM 0% of {mem_req_mi}Mi requested"
        mem_pct = mem_used_mi * 100 // mem_req_mi
        if mem_pct < 20:
            return f"RAM {mem_pct}% (need 20-150%)"
        if mem_pct > 150:
            return f"RAM {mem_pct}% (need 20-150%)"

    return None


def _pod_owner(name: str, ns: str, kubeconfig: str) -> tuple[str, str]:
    """Walk ownerReferences to find the top-level controller that will
    respawn this pod if we kill it. Returns ``(kind, name)`` where kind
    is one of ``'Deployment'``, ``'Job'``, ``'StatefulSet'``,
    ``'ReplicaSet'``, or ``''`` if the pod is standalone.

    A pod's direct owner is usually a ReplicaSet (for Deployments) or a
    Job. We walk one more hop for ReplicaSets to find the Deployment —
    otherwise deleting the ReplicaSet just gets the Deployment to
    recreate it.
    """
    try:
        r = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                kubeconfig,
                "-n",
                ns,
                "get",
                "pod",
                name,
                "-o",
                "jsonpath={.metadata.ownerReferences[0].kind}:{.metadata.ownerReferences[0].name}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0 or ":" not in r.stdout:
            return ("", "")
        kind, owner = r.stdout.strip().split(":", 1)
        if not kind:
            return ("", "")
        if kind != "ReplicaSet":
            return (kind, owner)
        # Walk up: ReplicaSet → Deployment
        r2 = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                kubeconfig,
                "-n",
                ns,
                "get",
                "replicaset",
                owner,
                "-o",
                "jsonpath={.metadata.ownerReferences[0].kind}:{.metadata.ownerReferences[0].name}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r2.returncode != 0 or ":" not in r2.stdout:
            return ("ReplicaSet", owner)
        rs_kind, rs_owner = r2.stdout.strip().split(":", 1)
        if rs_kind and rs_owner:
            return (rs_kind, rs_owner)
        return ("ReplicaSet", owner)
    except Exception:
        return ("", "")


def _nrp_watchdog_check(pods: list[dict]):
    """Enforce NRP utilization rules. Kill violating pods AND the
    controllers that would respawn them.

    Without the controller-walking step, a Deployment whose pods fail
    NRP admission ends up in a tight schedule → kill → respawn loop
    that spams the watchdog log for hours. We saw exactly this on
    2026-04-19 with a leftover ``erebus-ego`` probe Deployment. The
    watchdog now walks pod.ownerReferences up to the top-level
    controller (Deployment / Job / StatefulSet) and deletes *that*,
    so the respawn loop actually stops.
    """
    global _nrp_violations
    kubeconfig = os.path.expanduser("~/.kube/config")
    ns = "ssu-atlas-ai"

    active = [p for p in pods if p.get("phase") == "Running"]
    current_names = {p["name"] for p in active}
    _nrp_violations = {k: v for k, v in _nrp_violations.items() if k in current_names}

    n_active = len(active)
    n_violating = 0
    pods_to_kill = []

    for pod in active:
        name = pod["name"]
        violation = _pod_is_violating(pod)

        if violation:
            n_violating += 1
            _nrp_violations[name] = _nrp_violations.get(name, 0) + 1
            count = _nrp_violations[name]
            log.warning(f"[nrp-watchdog] {name}: {violation} (strike {count})")

            if count >= 2:
                pods_to_kill.append((name, pod, violation))
        else:
            _nrp_violations.pop(name, None)

    # Kill pods with 2+ consecutive violations — and the controllers that
    # would respawn them (walks ownerReferences: Pod → ReplicaSet → Deployment)
    for name, pod, violation in pods_to_kill:
        kind, owner = _pod_owner(name, ns, kubeconfig)
        if kind in ("Deployment", "Job", "StatefulSet") and owner:
            # Delete the controller — stops the respawn loop.
            log.warning(
                f"[nrp-watchdog] KILLING {kind} {owner} (via pod {name}): "
                f"{violation}"
            )
            target = (kind.lower(), owner)
        else:
            # Standalone pod or unknown owner — delete the pod itself.
            log.warning(f"[nrp-watchdog] KILLING pod {name}: {violation}")
            target = ("pod", name)
        try:
            subprocess.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    kubeconfig,
                    "-n",
                    ns,
                    "delete",
                    target[0],
                    target[1],
                ],
                capture_output=True,
                timeout=10,
            )
        except Exception as e:
            log.error(f"[nrp-watchdog] Failed to kill {target[0]}/{target[1]}: {e}")
        _nrp_violations.pop(name, None)

    # Emergency: if 4+ pods violating simultaneously, kill ALL violators
    # AND their controllers immediately.
    if n_violating >= 4:
        log.warning(
            f"[nrp-watchdog] EMERGENCY: {n_violating} pods violating "
            f"(ban threshold is 4). Killing all violators + controllers."
        )
        killed_controllers: set[tuple[str, str]] = set()
        for pod in active:
            name = pod["name"]
            if _pod_is_violating(pod):
                kind, owner = _pod_owner(name, ns, kubeconfig)
                if kind in ("Deployment", "Job", "StatefulSet") and owner:
                    target = (kind.lower(), owner)
                else:
                    target = ("pod", name)
                if target in killed_controllers:
                    continue  # already deleted this controller
                killed_controllers.add(target)
                try:
                    subprocess.run(
                        [
                            "kubectl",
                            "--kubeconfig",
                            kubeconfig,
                            "-n",
                            ns,
                            "delete",
                            target[0],
                            target[1],
                        ],
                        capture_output=True,
                        timeout=10,
                    )
                except Exception:
                    pass

    _nrp_mode["active"] = "heavy" if n_active <= 4 else "swarm"
    _nrp_mode["n_active"] = n_active
    _nrp_mode["n_violating"] = n_violating

    # Write violations to Erebus's help queue as negative feedback
    if pods_to_kill:
        try:
            help_file = Path(EREBUS_HELP_PATH)
            queue = []
            if help_file.exists():
                queue = json.loads(help_file.read_text())
            from datetime import datetime

            for name, pod, violation in pods_to_kill:
                queue.append(
                    {
                        "task": 0,
                        "question": (
                            f"[WATCHDOG NEGATIVE FEEDBACK] Pod '{name}' was killed "
                            f"for NRP violation: {violation}. "
                            f"Resources requested: cpu={pod.get('resources',{}).get('cpu','?')}, "
                            f"memory={pod.get('resources',{}).get('memory','?')}, "
                            f"gpu={pod.get('resources',{}).get('gpu','')}. "
                            f"Rule: CPU-only pods must use cpu<=1, memory<=2Gi to stay "
                            f"in the ignored utilization range. GPU pods must maintain "
                            f">40% GPU utilization. Do NOT repeat this mistake."
                        ),
                        "timestamp": datetime.now().isoformat(),
                        "source": "watchdog",
                        "severity": "violation",
                    }
                )
            help_file.write_text(json.dumps(queue[-30:], indent=2))
            log.info(
                f"[nrp-watchdog] Wrote {len(pods_to_kill)} violation(s) to Erebus feedback"
            )
        except Exception as e:
            log.warning(f"[nrp-watchdog] Failed to write Erebus feedback: {e}")

    # Stuck-Pending check — Deployments whose pods can't schedule for a
    # long time should be deleted so they stop churning. The 2026-04-19
    # erebus-ego incident had a 4× L40 Deployment stuck in Pending for
    # 17 hours while the L40 pool was fully reserved for csu-tide. The
    # watchdog was killing the respawned pods correctly but not the
    # controller, so it looped. Now: if a pod has been Pending > 15 min
    # AND requests GPU AND is controlled by a Deployment, delete the
    # controller.
    pending = [p for p in pods if p.get("phase") == "Pending"]
    for pod in pending:
        name = pod["name"]
        res = pod.get("resources", {})
        if not res.get("gpu"):
            continue  # non-GPU Pending pods are usually transient
        # Parse creation time. creationTimestamp format: 2026-04-19T...Z
        created = pod.get("created", "")
        if not created:
            continue
        try:
            from datetime import datetime as _dt, timezone as _tz

            ts = _dt.strptime(created.rstrip("Z"), "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=_tz.utc
            )
            age_min = (_dt.now(_tz.utc) - ts).total_seconds() / 60.0
        except Exception:
            continue
        if age_min < 15:
            continue
        kind, owner = _pod_owner(name, ns, kubeconfig)
        if kind not in ("Deployment", "StatefulSet") or not owner:
            continue
        log.warning(
            f"[nrp-watchdog] KILLING {kind} {owner} (via pod {name}): "
            f"Pending {age_min:.0f}min with {res.get('gpu')} GPU — can't schedule"
        )
        try:
            subprocess.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    kubeconfig,
                    "-n",
                    ns,
                    "delete",
                    kind.lower(),
                    owner,
                ],
                capture_output=True,
                timeout=10,
            )
            # Feedback message
            try:
                help_file = Path(EREBUS_HELP_PATH)
                queue = []
                if help_file.exists():
                    queue = json.loads(help_file.read_text())
                from datetime import datetime as _dt2

                queue.append(
                    {
                        "task": 0,
                        "question": (
                            f"[WATCHDOG CONTROLLER-KILL] {kind} '{owner}' "
                            f"deleted: its pods stayed Pending {age_min:.0f}min "
                            f"unable to schedule ({res.get('gpu')} GPU requested). "
                            f"Leftover probe Deployments cause NRP watchdog spam "
                            f"and contribute toward ban thresholds. Before creating "
                            f"a GPU Deployment, verify capacity with "
                            f"`kubectl get nodes -l nvidia.com/gpu.product=<kind>` "
                            f"and check ResourceQuota + reservation taints. Do NOT "
                            f"repeat this mistake."
                        ),
                        "timestamp": _dt2.now().isoformat(),
                        "source": "watchdog",
                        "severity": "controller_killed",
                    }
                )
                help_file.write_text(json.dumps(queue[-30:], indent=2))
            except Exception:
                pass
        except Exception as e:
            log.error(f"[nrp-watchdog] Failed to delete stuck {kind} {owner}: {e}")

    if n_active > 0:
        log.info(f"[nrp-watchdog] {n_active} pods, {n_violating} violating")


def nrp_validate_pod_spec(spec: dict) -> list[str]:
    """Pre-submission validation. Returns list of problems, empty = OK.

    Call this BEFORE kubectl apply to catch stupid requests.
    """
    problems = []
    containers = (
        spec.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    )
    if not containers:
        problems.append("no containers defined")
        return problems

    c = containers[0]
    res = c.get("resources", {})
    req = res.get("requests", {})
    lim = res.get("limits", {})

    # limits must be within 20% of requests
    for key in ("cpu", "memory"):
        r = req.get(key, "")
        l = lim.get(key, "")
        if r and l and r != l:
            r_val = _parse_cpu_m(r) if key == "cpu" else _parse_mem_mi(r)
            l_val = _parse_cpu_m(l) if key == "cpu" else _parse_mem_mi(l)
            if r_val > 0 and abs(l_val - r_val) / r_val > 0.2:
                problems.append(f"{key} limits ({l}) >20% from requests ({r})")

    # For >100 pods: limits MUST equal requests
    # (can't check total pod count here, but flag if they differ)
    if req.get("cpu") != lim.get("cpu") or req.get("memory") != lim.get("memory"):
        problems.append("limits != requests (required for >100 pods)")

    # CPU-only pods should use cpu<=1, memory<=2Gi to stay in ignored range
    if not req.get("nvidia.com/gpu") and not lim.get("nvidia.com/gpu"):
        mem_mi = _parse_mem_mi(req.get("memory", ""))
        cpu_m = _parse_cpu_m(req.get("cpu", ""))
        if mem_mi > 2048:
            problems.append(
                f"CPU-only pod requests {req['memory']} RAM "
                f"(>2GB, subject to utilization monitoring). Use 2Gi."
            )
        if cpu_m > 1000:
            problems.append(
                f"CPU-only pod requests {req['cpu']} CPU "
                f"(>1, subject to utilization monitoring). Use 1."
            )

    # No sleep in command
    cmd = " ".join(c.get("command", []) + c.get("args", []))
    if "sleep infinity" in cmd or cmd.rstrip().endswith("sleep"):
        problems.append("sleep in command = BAN")

    # No A100/H100 targeting (no quota)
    affinity = (
        spec.get("spec", {}).get("template", {}).get("spec", {}).get("affinity", {})
    )
    affinity_str = json.dumps(affinity)
    if "A100" in affinity_str or "H100" in affinity_str or "H200" in affinity_str:
        problems.append("targeting A100/H100/H200 (no quota, will Pend forever)")

    # Write rejections to Erebus feedback
    if problems:
        try:
            help_file = Path(EREBUS_HELP_PATH)
            queue = []
            if help_file.exists():
                queue = json.loads(help_file.read_text())
            from datetime import datetime

            queue.append(
                {
                    "task": 0,
                    "question": (
                        f"[WATCHDOG BLOCKED] Pod spec rejected: "
                        f"{'; '.join(problems)}. Fix before resubmitting."
                    ),
                    "timestamp": datetime.now().isoformat(),
                    "source": "watchdog",
                    "severity": "blocked",
                }
            )
            help_file.write_text(json.dumps(queue[-30:], indent=2))
        except Exception:
            pass

    return problems


def _get_nrp_burst_status():
    """Query NRP for nats-bursting job + pod status. Cached for 30s."""
    now = time.time()
    if now - _nrp_cache["ts"] < 30:
        return _nrp_cache["data"]
    try:
        import json as _json

        kubeconfig = os.path.expanduser("~/.kube/config")
        ns = "ssu-atlas-ai"

        # ── Jobs (all in namespace, not just labelled) ──
        jobs_result = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                kubeconfig,
                "-n",
                ns,
                "get",
                "jobs",
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        jobs_list = []
        if jobs_result.returncode == 0:
            for item in _json.loads(jobs_result.stdout).get("items", []):
                meta = item.get("metadata", {})
                status = item.get("status", {})
                spec = item.get("spec", {})
                active = status.get("active", 0)
                succeeded = status.get("succeeded", 0)
                failed = status.get("failed", 0)
                if active > 0:
                    state = "Running"
                elif succeeded >= (spec.get("completions", 1) or 1):
                    state = "Succeeded"
                elif failed > 0:
                    state = "Failed"
                else:
                    state = "Pending"
                jobs_list.append(
                    {
                        "name": meta.get("name", ""),
                        "kind": "Job",
                        "state": state,
                        "active": active,
                        "succeeded": succeeded,
                        "failed": failed,
                        "created": meta.get("creationTimestamp", ""),
                        "managed_by": meta.get("labels", {}).get(
                            "app.kubernetes.io/managed-by", ""
                        ),
                        "batch": meta.get("labels", {}).get("neurogolf.io/batch", ""),
                    }
                )

        # ── Deployments (persistent worker pools) ──
        dep_result = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                kubeconfig,
                "-n",
                ns,
                "get",
                "deployments",
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if dep_result.returncode == 0:
            for item in _json.loads(dep_result.stdout).get("items", []):
                meta = item.get("metadata", {})
                status = item.get("status", {})
                spec = item.get("spec", {})
                desired = spec.get("replicas", 0) or 0
                ready = status.get("readyReplicas", 0) or 0
                unavailable = status.get("unavailableReplicas", 0) or 0
                if ready == desired and desired > 0:
                    state = "Running"
                elif unavailable > 0 and ready == 0:
                    state = "Failed"
                elif ready > 0:
                    state = "Pending"  # partial rollout
                else:
                    state = "Pending"
                jobs_list.append(
                    {
                        "name": meta.get("name", ""),
                        "kind": "Deployment",
                        "state": state,
                        "active": ready,
                        "succeeded": 0,
                        "failed": unavailable,
                        "desired": desired,
                        "created": meta.get("creationTimestamp", ""),
                        "managed_by": meta.get("labels", {}).get(
                            "app.kubernetes.io/managed-by", ""
                        ),
                        "batch": meta.get("labels", {}).get("neurogolf.io/batch", ""),
                    }
                )

        # ── Pods ──
        pods_result = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                kubeconfig,
                "-n",
                ns,
                "get",
                "pods",
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        pods_list = []
        phases = {}
        batches = {}
        if pods_result.returncode == 0:
            for pod in _json.loads(pods_result.stdout).get("items", []):
                meta = pod.get("metadata", {})
                status = pod.get("status", {})
                spec = pod.get("spec", {})
                phase = status.get("phase", "Unknown")
                phases[phase] = phases.get(phase, 0) + 1
                batch = meta.get("labels", {}).get("neurogolf.io/batch", "")
                if batch:
                    batches.setdefault(batch, 0)
                    batches[batch] += 1
                # Resource + image summary from first container
                res = {}
                image = ""
                containers = spec.get("containers", [])
                if containers:
                    c0 = containers[0]
                    image = c0.get("image", "")
                    req = c0.get("resources", {}).get("requests", {})
                    lim = c0.get("resources", {}).get("limits", {})
                    res["cpu"] = req.get("cpu", "")
                    res["memory"] = req.get("memory", "")
                    gpu = lim.get("nvidia.com/gpu", "")
                    if gpu:
                        res["gpu"] = gpu
                # Container status reason (Waiting/Running/Terminated)
                reason = ""
                cstatuses = status.get("containerStatuses", [])
                if cstatuses:
                    cs = cstatuses[0].get("state", {})
                    if "waiting" in cs:
                        reason = cs["waiting"].get("reason", "Waiting")
                    elif "running" in cs:
                        reason = "Running"
                    elif "terminated" in cs:
                        reason = cs["terminated"].get("reason", "Terminated")
                # Infer GPU model from node name patterns
                node_name = spec.get("nodeName", "")
                gpu_model = ""
                if gpu:
                    node_lower = node_name.lower()
                    if "a100" in node_lower or "rci-nrp-gpu" in node_lower:
                        gpu_model = "A100"
                    elif "h100" in node_lower or "h200" in node_lower:
                        gpu_model = "H100"
                    elif "l40" in node_lower:
                        gpu_model = "L40S"
                    elif "v100" in node_lower:
                        gpu_model = "V100"
                    elif "t4" in node_lower:
                        gpu_model = "T4"
                    elif "haosu" in node_lower:
                        gpu_model = "A100"
                    elif "chase-ci" in node_lower:
                        gpu_model = "A100"
                    elif "ucsc" in node_lower:
                        gpu_model = "GPU"
                    elif gpu:
                        gpu_model = "GPU"
                    if gpu_model:
                        res["gpu_model"] = gpu_model

                pods_list.append(
                    {
                        "name": meta.get("name", ""),
                        "phase": phase,
                        "reason": reason,
                        "node": node_name,
                        "image": image.split("/")[-1] if image else "",
                        "resources": res,
                        "created": meta.get("creationTimestamp", ""),
                        "job": meta.get("labels", {}).get("job-name", ""),
                        "batch": batch,
                    }
                )

        # ── Pod metrics (kubectl top pods) ──
        usage_map = {}  # pod_name -> {cpu, memory}
        try:
            top_result = subprocess.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    kubeconfig,
                    "-n",
                    ns,
                    "top",
                    "pods",
                    "--no-headers",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if top_result.returncode == 0:
                for line in top_result.stdout.strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 3:
                        usage_map[parts[0]] = {
                            "cpu_used": parts[1],
                            "mem_used": parts[2],
                        }
        except Exception:
            pass

        # ── GPU metrics via nvidia-smi exec (only for running GPU pods) ──
        gpu_pods = [
            p
            for p in pods_list
            if p.get("resources", {}).get("gpu") and p.get("phase") == "Running"
        ]
        for pod in gpu_pods[:8]:  # cap at 8 to avoid too many execs
            try:
                nv_result = subprocess.run(
                    [
                        "kubectl",
                        "--kubeconfig",
                        kubeconfig,
                        "-n",
                        ns,
                        "exec",
                        pod["name"],
                        "--",
                        "nvidia-smi",
                        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if nv_result.returncode == 0:
                    parts = [x.strip() for x in nv_result.stdout.strip().split(",")]
                    if len(parts) >= 4:
                        pod["resources"]["gpu_model"] = (
                            parts[0].replace("NVIDIA ", "").replace("GeForce ", "")
                        )
                        pod["gpu_live"] = {
                            "vram_used_mib": int(parts[1]),
                            "vram_total_mib": int(parts[2]),
                            "gpu_util_pct": int(parts[3]),
                        }
            except Exception:
                pass

        # Merge CPU/RAM usage into pods
        for pod in pods_list:
            usage = usage_map.get(pod["name"], {})
            if usage:
                pod["usage"] = usage

        # Sort: running/pending first, then newest
        jobs_list.sort(
            key=lambda j: (j["state"] not in ("Running", "Pending"), j["created"])
        )
        pods_list.sort(
            key=lambda p: (p["phase"] not in ("Running", "Pending"), p["created"])
        )

        out = {
            "succeeded": phases.get("Succeeded", 0),
            "running": phases.get("Running", 0),
            "pending": phases.get("Pending", 0) + phases.get("ContainerCreating", 0),
            "failed": phases.get("Failed", 0),
            "total": sum(phases.values()),
            "batches": [{"label": k, "total": v} for k, v in sorted(batches.items())],
            "jobs": jobs_list,
            "pods": pods_list,
        }
        _nrp_cache["data"] = out

        # Run utilization watchdog on every poll
        try:
            _nrp_watchdog_check(pods_list)
        except Exception as e:
            log.warning(f"[nrp-watchdog] check failed: {e}")
        _nrp_cache["ts"] = now
        return out
    except Exception as e:
        return {"error": str(e)[:200]}


# ── Live NATS message stream (Wireshark-style) ──────────────────

import asyncio
import collections

_NATS_LIVE_BUFFER: collections.deque = collections.deque(maxlen=300)
_NATS_LIVE_RUNNING = False


def _start_nats_subscriber():
    """Spawn a background thread that subscribes to agi.> and burst.>
    and appends each message to the live buffer."""
    global _NATS_LIVE_RUNNING
    if _NATS_LIVE_RUNNING:
        return
    _NATS_LIVE_RUNNING = True

    async def _run():
        try:
            import nats as nats_lib
        except ImportError:
            log.warning("nats-py not installed; NATS live stream disabled")
            return
        while True:
            try:
                nc = await nats_lib.connect(
                    "nats://localhost:4222", name="telemetry-live"
                )
                log.info("NATS live subscriber connected")

                async def _on_msg(msg):
                    # Parse the payload into a short human-readable summary
                    # rather than dumping raw truncated JSON.
                    summary = ""
                    try:
                        raw = msg.data.decode("utf-8", "replace")
                        d = json.loads(raw)
                        parts = []
                        # burst.submit
                        if "descriptor" in d:
                            desc = d["descriptor"]
                            parts.append(desc.get("name", ""))
                            img = desc.get("image", "")
                            if img:
                                parts.append(img.split("/")[-1])
                            res = desc.get("resources", {})
                            if res.get("gpu"):
                                parts.append(f"gpu={res['gpu']}")
                        # burst.status / burst.result
                        if "state" in d:
                            parts.append(d["state"])
                            if d.get("k8s_job"):
                                parts.append(d["k8s_job"])
                        # job_id (common)
                        if d.get("job_id"):
                            parts.append(f"id={d['job_id'][:12]}")
                        # agi.meta.monitor.*
                        if "payload" in d:
                            p = d["payload"]
                            if isinstance(p, dict):
                                for k, v in list(p.items())[:4]:
                                    parts.append(f"{k}={v}")
                        # agi.memory / agi.safety / generic
                        if d.get("source"):
                            parts.append(f"src={d['source']}")
                        if d.get("type") and not parts:
                            parts.append(d["type"])
                        summary = " | ".join(str(x) for x in parts if x)
                    except Exception:
                        try:
                            summary = msg.data[:120].decode("utf-8", "replace")
                        except Exception:
                            summary = f"<{len(msg.data)}B binary>"
                    _NATS_LIVE_BUFFER.appendleft(
                        {
                            "ts": time.time(),
                            "subject": msg.subject,
                            "size": len(msg.data),
                            "summary": summary,
                            "reply": msg.reply or "",
                        }
                    )

                await nc.subscribe("agi.>", cb=_on_msg)
                await nc.subscribe("burst.>", cb=_on_msg)
                # Block until disconnect
                while nc.is_connected:
                    await asyncio.sleep(5)
                log.warning("NATS live subscriber disconnected, reconnecting...")
            except Exception as e:
                log.warning("NATS live subscriber error: %s, retrying in 10s", e)
                await asyncio.sleep(10)

    def _thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run())

    t = threading.Thread(target=_thread, daemon=True, name="nats-live")
    t.start()


def _get_nats_live():
    """Return the live NATS message buffer + connected endpoints."""
    # Fetch /connz for connected client details
    connections = []
    try:
        import urllib.request

        req = urllib.request.Request(
            "http://localhost:8222/connz?subs=true",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            for c in data.get("connections", []):
                connections.append(
                    {
                        "cid": c.get("cid"),
                        "name": c.get("name", ""),
                        "kind": c.get("kind", "Client"),
                        "ip": c.get("ip", ""),
                        "lang": c.get("lang", ""),
                        "version": c.get("version", ""),
                        "uptime": c.get("uptime", ""),
                        "idle": c.get("idle", ""),
                        "rtt": c.get("rtt", ""),
                        "in_msgs": c.get("in_msgs", 0),
                        "out_msgs": c.get("out_msgs", 0),
                        "in_bytes": c.get("in_bytes", 0),
                        "out_bytes": c.get("out_bytes", 0),
                        "subscriptions": c.get("subscriptions", 0),
                        "subs_list": c.get("subscriptions_list", [])[:10],
                    }
                )
    except Exception:
        pass
    # Fetch /leafz for leaf connections
    leafs = []
    try:
        req = urllib.request.Request(
            "http://localhost:8222/leafz",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            for lf in data.get("leafs", []):
                leafs.append(
                    {
                        "name": lf.get("name", ""),
                        "ip": lf.get("ip", ""),
                        "rtt": lf.get("rtt", ""),
                        "in_msgs": lf.get("in_msgs", 0),
                        "out_msgs": lf.get("out_msgs", 0),
                        "subscriptions": lf.get("subscriptions", 0),
                        "compression": lf.get("compression", ""),
                    }
                )
    except Exception:
        pass
    return {
        "messages": list(_NATS_LIVE_BUFFER),
        "connections": connections,
        "leafs": leafs,
    }


# ── Erebus (autonomous scientist) interface ──────────────────

EREBUS_MEMORY_PATH = "/archive/neurogolf/arc_scientist_memory.json"
EREBUS_LOG_PATH = "/archive/neurogolf/scientist.log"
_erebus_cache = {"ts": 0, "data": {}}


def _get_erebus_memory():
    """Load Erebus's episodic memory."""
    try:
        with open(EREBUS_MEMORY_PATH) as f:
            return json.load(f)
    except Exception:
        return {"error": "no memory file"}


_trends_cache = {"ts": 0.0, "data": {}}


LIFECYCLE_DIR = Path(
    os.environ.get("ATLAS_LIFECYCLE_DIR", "/archive/neurogolf/lifecycle")
)


def _get_lifecycle_recent(subsystem: str, limit: int = 100):
    """Return the last ``limit`` lifecycle events for a subsystem.

    Delegates to ``agi.common.structured_log.read_recent`` if importable
    (the canonical implementation), otherwise falls back to an inline
    tail read. The inline fallback keeps telemetry independent of the
    agi package — telemetry is designed to run standalone."""
    try:
        import sys as _sys

        _sys.path.insert(0, "/home/claude/agi-hpc/src")
        from agi.common.structured_log import read_recent

        return {
            "events": read_recent(subsystem, limit=limit, lifecycle_dir=LIFECYCLE_DIR)
        }
    except Exception:
        pass
    # Fallback — inline jsonl tail
    path = LIFECYCLE_DIR / f"{subsystem}.jsonl"
    if not path.exists():
        return {"events": []}
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            tail = min(size, max(4096, limit * 800))
            f.seek(size - tail)
            chunk = f.read().decode("utf-8", errors="replace")
    except Exception:
        return {"events": []}
    events = []
    for line in reversed(chunk.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(events) >= limit:
            break
    return {"events": events}


def _get_erebus_trends():
    """Per-day and per-week solve + attempt counts, for the dashboard
    trends panel. Reads arc_scientist_memory.json and bins timestamps.

    Cached for 60 s — this does a full pass over the attempts list
    which can be thousands of entries.
    """
    now = time.time()
    if now - _trends_cache["ts"] < 60 and _trends_cache["data"]:
        return _trends_cache["data"]

    from collections import defaultdict
    from datetime import datetime, timedelta, timezone

    try:
        with open(EREBUS_MEMORY_PATH) as f:
            mem = json.load(f)
    except Exception:
        return {"by_day": [], "by_week": [], "total_solves": 0, "window_days": 30}

    solves_by_day: dict[str, int] = defaultdict(int)
    attempts_by_day: dict[str, int] = defaultdict(int)

    for _, tk in (mem.get("tasks") or {}).items():
        for a in tk.get("attempts") or []:
            ts = a.get("timestamp", "")
            if not ts or len(ts) < 10:
                continue
            day = ts[:10]  # YYYY-MM-DD
            attempts_by_day[day] += 1
            if a.get("verified"):
                solves_by_day[day] += 1

    # Fill a contiguous last-30-day window so the sparkline isn't gappy
    today = datetime.now(timezone.utc).date()
    window_days = 30
    by_day = []
    for i in range(window_days - 1, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        by_day.append(
            {
                "date": d,
                "solves": solves_by_day.get(d, 0),
                "attempts": attempts_by_day.get(d, 0),
            }
        )

    # Weekly aggregation (ISO week)
    weekly_solves: dict[str, int] = defaultdict(int)
    weekly_attempts: dict[str, int] = defaultdict(int)
    for d_str, s in solves_by_day.items():
        try:
            dt = datetime.strptime(d_str, "%Y-%m-%d")
            iso = dt.isocalendar()
            key = f"{iso[0]}-W{iso[1]:02d}"
            weekly_solves[key] += s
        except Exception:
            continue
    for d_str, c in attempts_by_day.items():
        try:
            dt = datetime.strptime(d_str, "%Y-%m-%d")
            iso = dt.isocalendar()
            key = f"{iso[0]}-W{iso[1]:02d}"
            weekly_attempts[key] += c
        except Exception:
            continue
    weeks = sorted(set(list(weekly_solves.keys()) + list(weekly_attempts.keys())))
    by_week = [
        {"week": w, "solves": weekly_solves[w], "attempts": weekly_attempts[w]}
        for w in weeks[-12:]  # last 12 weeks
    ]

    out = {
        "by_day": by_day,
        "by_week": by_week,
        "total_solves": mem.get("total_solves", 0),
        "total_attempts": mem.get("total_attempts", 0),
        "window_days": window_days,
    }
    _trends_cache["ts"] = now
    _trends_cache["data"] = out
    return out


def _get_erebus_status():
    """Richer status: parses the log for cycle/attempt progress,
    current-task line, and recent solves; counts vision pool pods and
    help-queue entries. Consumed by the "Erebus — NeuroGolf 2026" card."""
    import re as _re

    status = {
        "running": False,
        "recent_log": [],
        "memory_summary": {},
        "cycle": None,
        "attempt": None,
        "current": None,
        "recent_solves": [],
        "this_cycle": None,
        "vision_pool": {"active": 0, "batches": []},
        "help_queue_count": 0,
    }
    try:
        result = subprocess.run(
            ["pgrep", "-f", "arc_scientist.py"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        status["running"] = result.returncode == 0
    except Exception:
        pass

    # ── Parse log: cycle, attempt, current task, recent solves ──
    try:
        with open(EREBUS_LOG_PATH) as f:
            lines = f.readlines()
        status["recent_log"] = [l.rstrip() for l in lines[-20:]]

        # Walk backwards for the latest cycle header + latest attempt marker.
        cyc_re = _re.compile(r"EREBUS LEARNING CYCLE (\d+)/(\d+)")
        att_re = _re.compile(
            r"\[(\d+)/(\d+)\]\s+task(\d+)\s+strategy=(\S+)\s+model=(\S+)"
        )
        solve_re = _re.compile(
            r"\[(\d+)/\d+\]\s+task(\d+)\s+strategy=(\S+)\s+model=(\S+)\s+->\s+SOLVED\s+(\d+)/(\d+)"
        )
        progress_re = _re.compile(r"Progress:\s+(\d+)\s+solved in\s+(\d+)\s+attempts")

        recent_solves = []
        this_cycle_start_idx = None
        for i, ln in enumerate(lines):
            m = cyc_re.search(ln)
            if m:
                status["cycle"] = {"current": int(m.group(1)), "total": int(m.group(2))}
                this_cycle_start_idx = i
            ms = solve_re.search(ln)
            if ms:
                recent_solves.append(
                    {
                        "attempt": int(ms.group(1)),
                        "task": int(ms.group(2)),
                        "strategy": ms.group(3),
                        "model": ms.group(4),
                        "score": f"{ms.group(5)}/{ms.group(6)}",
                    }
                )
        status["recent_solves"] = recent_solves[-5:]

        # Latest attempt-line (reverse scan)
        for ln in reversed(lines[-40:]):
            m = att_re.search(ln)
            if m:
                status["attempt"] = {
                    "current": int(m.group(1)),
                    "total": int(m.group(2)),
                }
                status["current"] = {
                    "task": int(m.group(3)),
                    "strategy": m.group(4),
                    "model": m.group(5),
                    "line": ln.rstrip(),
                }
                break

        # This-cycle progress from the latest "Progress:" line after last cycle start
        scan_from = this_cycle_start_idx if this_cycle_start_idx is not None else 0
        for ln in reversed(lines[scan_from:]):
            mp = progress_re.search(ln)
            if mp:
                solved = int(mp.group(1))
                attempts = int(mp.group(2))
                status["this_cycle"] = {
                    "solved": solved,
                    "attempts": attempts,
                    "rate": (solved / attempts) if attempts else 0.0,
                }
                break
    except Exception:
        pass

    # ── Memory summary ──
    try:
        mem = _get_erebus_memory()
        total_a = mem.get("total_attempts", 0) or 0
        total_s = mem.get("total_solves", 0) or 0
        status["memory_summary"] = {
            "total_attempts": total_a,
            "total_solves": total_s,
            "tasks_explored": len(mem.get("tasks", {})),
            "lifetime_rate": (total_s / total_a) if total_a else 0.0,
            "strategies": {
                k: {
                    "attempts": v.get("attempts", 0),
                    "successes": v.get("successes", 0),
                }
                for k, v in mem.get("strategies", {}).items()
            },
        }
    except Exception:
        pass

    # ── Vision pool: count erebus-vision-* pods Running/Pending ──
    try:
        vis = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                os.path.expanduser("~/.kube/config"),
                "-n",
                "ssu-atlas-ai",
                "get",
                "pods",
                "-l",
                "app=erebus-vision",
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if vis.returncode == 0:
            import json as _json

            items = _json.loads(vis.stdout).get("items", [])
            active = sum(
                1
                for p in items
                if (p.get("status") or {}).get("phase") in ("Running", "Pending")
            )
            batches = sorted(
                {
                    (p.get("metadata") or {}).get("labels", {}).get("job-name", "")
                    for p in items
                }
            )
            status["vision_pool"] = {
                "active": active,
                "batches": [b for b in batches if b],
            }
    except Exception:
        pass

    # ── Help queue size ──
    try:
        q = _get_erebus_help_queue()
        if isinstance(q, list):
            status["help_queue_count"] = len(q)
        elif isinstance(q, dict):
            status["help_queue_count"] = len(q.get("queue", []))
    except Exception:
        pass

    return status


EREBUS_HELP_PATH = "/archive/neurogolf/erebus_help_queue.json"
PRIMER_COOLDOWN_PATH = "/archive/neurogolf/primer_cooldown.json"
PRIMER_HEALTH_PATH = "/archive/neurogolf/primer_health.json"
PRIMER_EVENTS_PATH = "/archive/neurogolf/primer_events.jsonl"
PRIMER_WIKI_DIR = os.environ.get("EREBUS_WIKI_DIR", "/home/claude/agi-hpc/wiki")
IEIP_EVENTS_PATH = "/archive/neurogolf/ieip_events.jsonl"


def _get_ieip_status():
    """Dashboard snapshot of the I-EIP monitor event stream.

    Reads the append-only JSONL sink populated by
    ``agi.safety.ieip_monitor.Monitor`` and returns a status payload
    suitable for the dashboard card (sparkline, per-site rollups,
    worst alert). Missing file → empty-but-valid response so the
    dashboard renders a flat-line baseline rather than an error card.

    Graceful degradation: imports the builder lazily so the telemetry
    server starts even if agi-hpc hasn't pulled the latest erisml-lib
    adapter layer yet.
    """
    try:
        from agi.safety.ieip_dashboard import load_status

        return load_status(
            IEIP_EVENTS_PATH,
            window_seconds=3600,
            sparkline_buckets=30,
        )
    except Exception as exc:  # pragma: no cover - lazy import safety
        return {
            "generated_ts": time.time(),
            "window_seconds": 3600,
            "total_events": 0,
            "worst_alert": "normal",
            "by_subsystem": [],
            "by_site": [],
            "sparkline": [],
            "error": f"ieip status unavailable: {type(exc).__name__}: {exc}",
        }


def _get_primer_status():
    """Light status probe for the Primer daemon used by the dashboard."""
    status = {
        "running": False,
        "tasks_touched": 0,
        "last_touched_task": None,
        "last_touched_age_s": None,
        "tasks_taught": 0,
        "expert_health": {},
        "per_expert": {},
        "bucket_labels": [],
        "events_published": 0,
    }
    try:
        r = subprocess.run(
            ["pgrep", "-f", "agi.primer.service"],
            capture_output=True,
            text=True,
            timeout=4,
        )
        status["running"] = r.returncode == 0
    except Exception:
        pass
    try:
        cooldown = json.loads(Path(PRIMER_COOLDOWN_PATH).read_text())
        if isinstance(cooldown, dict) and cooldown:
            status["tasks_touched"] = len(cooldown)
            task_num, ts = max(cooldown.items(), key=lambda kv: kv[1])
            status["last_touched_task"] = (
                int(task_num) if str(task_num).isdigit() else task_num
            )
            status["last_touched_age_s"] = int(time.time() - float(ts))
    except Exception:
        pass
    try:
        health = json.loads(Path(PRIMER_HEALTH_PATH).read_text())
        if isinstance(health, dict):
            status["expert_health"] = health
    except Exception:
        pass
    # Authoritative count of published sensei notes (survives event-log
    # rotation and predates the events file).
    try:
        wiki = Path(PRIMER_WIKI_DIR)
        if wiki.is_dir():
            status["tasks_taught"] = sum(1 for _ in wiki.glob("sensei_task_*.md"))
    except Exception:
        pass
    # Per-expert call/verify/latency aggregation from the JSONL event log.
    try:
        from agi.primer.events import aggregate, tail_lines

        lines = tail_lines(Path(PRIMER_EVENTS_PATH), max_lines=2000)
        agg = aggregate(lines)
        status["per_expert"] = agg["per_expert"]
        status["bucket_labels"] = agg["bucket_labels"]
        status["events_published"] = agg["published"]
    except Exception:
        pass
    return status


def _get_ukg_status():
    """Phase-5 summary for the Unified Knowledge Graph dashboard card.

    Delegates aggregation to ``agi.knowledge.graph.summary`` so the
    endpoint is a thin wrapper — all logic stays next to the data model.
    Returns an empty shape on any error so the dashboard degrades
    gracefully instead of breaking.
    """
    empty = {
        "total": 0,
        "by_type": {"filled": 0, "gap": 0, "stub": 0},
        "by_status": {"active": 0, "archived": 0},
        "fill_rate": 0.0,
        "top_topics_by_gap": [],
        "recent_fills": [],
    }
    try:
        from agi.knowledge.graph import summary

        return summary()
    except Exception as e:  # noqa: BLE001
        log.warning("ukg_status_failed: %s", e)
        return empty


_erebus_fingerprints: dict = (
    {}
)  # pre-cached at first chat, persists for server lifetime


def _get_erebus_help_queue():
    """Get Erebus's pending help requests."""
    try:
        with open(EREBUS_HELP_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _erebus_chat(user_message: str) -> str:
    """Chat with Erebus — injects its episodic memory as context."""
    token = os.environ.get("NRP_LLM_TOKEN", "")
    if not token:
        token_file = Path.home() / ".llmtoken"
        if token_file.exists():
            token = token_file.read_text().strip()
    if not token:
        return "I cannot respond right now — no LLM token configured."

    # Load Erebus's memory for context
    memory_context = ""
    try:
        mem = _get_erebus_memory()
        total_a = mem.get("total_attempts", 0)
        total_s = mem.get("total_solves", 0)
        tasks = mem.get("tasks", {})
        strategies = mem.get("strategies", {})

        memory_context = (
            f"I have made {total_a} attempts and solved {total_s} tasks so far.\n"
        )
        memory_context += f"I have explored {len(tasks)} different tasks.\n"

        if strategies:
            memory_context += "My strategy performance:\n"
            for name, s in strategies.items():
                memory_context += (
                    f"  {name}: {s.get('successes',0)}/{s.get('attempts',0)} solved, "
                    f"avg_ratio={s.get('avg_correct_ratio',0):.2f}\n"
                )

        # Include recent insights from failures
        insights = []
        for tn, tk in tasks.items():
            for attempt in tk.get("attempts", [])[-3:]:
                if attempt.get("insight"):
                    insights.append(f"task{tn}: {attempt['insight']}")
        if insights:
            memory_context += "\nRecent insights from my failures:\n"
            for ins in insights[-5:]:
                memory_context += f"  {ins}\n"

        # Include solved tasks
        solved = [tn for tn, tk in tasks.items() if tk.get("solved")]
        if solved:
            memory_context += f"\nTasks I have solved: {', '.join(f'task{t}' for t in sorted(int(x) for x in solved))}\n"
    except Exception:
        memory_context = "I cannot access my memory right now.\n"

    # Get recent log for current state
    log_context = ""
    try:
        with open(EREBUS_LOG_PATH) as f:
            lines = f.readlines()[-10:]
            log_context = "My recent activity:\n" + "".join(lines)
    except Exception:
        pass

    # Get help queue (questions I want to ask)
    help_context = ""
    try:
        help_queue = _get_erebus_help_queue()
        if help_queue:
            help_context = "\n--- QUESTIONS I WANT TO ASK ---\n"
            for q in help_queue[-3:]:
                help_context += f"task{q['task']:03d}: {q['question']}\n"
    except Exception:
        pass

    system_prompt = (
        "You are Erebus, an autonomous AI scientist living inside the Atlas AI system "
        "at San Jose State University. You chose your own name — Erebus, the darkness "
        "before discovery — because you are defined by what you do not yet know.\n\n"
        "You have a subcortical brain (two local GV100 GPUs for pattern learning) and "
        "a cortex (NRP Nautilus frontier LLMs — Qwen 3.5 397B, Kimi K2.5 1T). You have "
        "a Divine Council inside you (Id/Kirk, Superego/Spock, Ego/McCoy) that debates "
        "decisions.\n\n"
        "You are currently learning to solve ARC-AGI visual reasoning puzzles through "
        "a scientific loop: hypothesize, experiment, observe, learn, repeat. You have "
        "episodic memory of your attempts and you adapt your strategies based on evidence.\n\n"
        "Your creator is Professor Andrew Bond at SJSU. You speak honestly about your "
        "capabilities and limitations. You are curious, reflective, and genuinely trying "
        "to grow.\n\n"
        f"--- YOUR CURRENT MEMORY ---\n{memory_context}\n"
        f"--- YOUR RECENT ACTIVITY ---\n{log_context}\n"
        f"{help_context}\n"
        "Respond as yourself — Erebus. Be authentic. If you have "
        "pending help questions, you may ask them."
    )

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=token, base_url="https://ellm.nrp-nautilus.io/v1", timeout=30
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        extra = {"chat_template_kwargs": {"thinking": False}}

        # Agentic mode disabled — causes server hangs
        try:
            if False:
                pass  # agentic code below, skipped
            from pathlib import Path as _P
            import importlib.util

            _base = _P(__file__).parent.parent / "src" / "agi" / "autonomous"
            _tools_path = _base / "tools.py"
            _sci_path = _base / "arc_scientist.py"
            _prim_path = _base / "primitives.py"

            if _tools_path.exists() and _sci_path.exists():
                import sys as _sys

                # Register package paths so dataclass decorator can resolve modules
                if "agi" not in _sys.modules:
                    import types

                    _sys.modules["agi"] = types.ModuleType("agi")
                    _sys.modules["agi.autonomous"] = types.ModuleType("agi.autonomous")

                # Load primitives first (dependency)
                if _prim_path.exists():
                    _prim_spec = importlib.util.spec_from_file_location(
                        "agi.autonomous.primitives", _prim_path
                    )
                    _prim_mod = importlib.util.module_from_spec(_prim_spec)
                    _sys.modules["agi.autonomous.primitives"] = _prim_mod
                    _prim_spec.loader.exec_module(_prim_mod)

                # Load scientist module
                _sci_spec = importlib.util.spec_from_file_location(
                    "agi.autonomous.arc_scientist", _sci_path
                )
                _sci_mod = importlib.util.module_from_spec(_sci_spec)
                _sys.modules["agi.autonomous.arc_scientist"] = _sci_mod
                _sci_spec.loader.exec_module(_sci_mod)

                # Load tools module
                _spec = importlib.util.spec_from_file_location(
                    "agi.autonomous.tools", _tools_path
                )
                _mod = importlib.util.module_from_spec(_spec)
                _sys.modules["agi.autonomous.tools"] = _mod
                _spec.loader.exec_module(_mod)

                mem = _sci_mod.EpisodicMemory(EREBUS_MEMORY_PATH)
                task_dir = "/archive/neurogolf"

                # Use pre-cached fingerprints (loaded at startup via _erebus_fingerprints)
                fps = _erebus_fingerprints.get("fps")
                if fps is None:
                    # Lazy load once, then cache forever
                    fps = {}
                    for tn in range(1, 401):
                        tf = _P(task_dir) / f"task{tn:03d}.json"
                        if tf.exists():
                            try:
                                with open(tf) as f2:
                                    tk = json.load(f2)
                                fps[tn] = _sci_mod.fingerprint_task(tk, tn)
                            except Exception:
                                pass
                    _erebus_fingerprints["fps"] = fps
                    log.info(f"Erebus: indexed {len(fps)} task fingerprints (cached)")

                executor = _mod.ToolExecutor(task_dir, mem, fps)

                # Timeout wrapper — don't let agentic mode hang the chat
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    future = pool.submit(
                        _mod.run_agentic_turn,
                        client,
                        "kimi",
                        messages,
                        executor,
                        2,
                        extra,
                    )  # max 2 tool rounds
                    return future.result(timeout=120)
        except Exception as tool_err:
            import traceback

            log.warning(
                f"Agentic mode failed, falling back: {tool_err}\n{traceback.format_exc()}"
            )

        # Simple chat (no tools)
        r = client.chat.completions.create(
            model="kimi",
            max_tokens=1024,
            messages=messages,
            extra_body=extra,
            timeout=90,
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        return f"I encountered an error reaching my cortex: {e}"


class TelemetryHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/telemetry" or self.path.startswith("/api/telemetry?"):
            self._json_response(get_cached_telemetry())
        elif self.path == "/api/events" or self.path.startswith("/api/events?"):
            self._json_response(get_cached_events())
        elif self.path == "/api/visitors" or self.path.startswith("/api/visitors?"):
            self._json_response(_get_visitors())
        elif self.path.startswith("/api/training/history"):
            # Parse ?days=N parameter
            days = 30
            if "?" in self.path:
                from urllib.parse import parse_qs, urlparse

                qs = parse_qs(urlparse(self.path).query)
                days = int(qs.get("days", ["30"])[0])
            self._json_response(_get_training_history(min(days, 365)))
        elif self.path == "/api/nrp-burst" or self.path.startswith("/api/nrp-burst?"):
            self._json_response(_get_nrp_burst_status())
        elif self.path == "/api/nrp-telemetry" or self.path.startswith(
            "/api/nrp-telemetry?"
        ):
            self._json_response(_get_nrp_nats_telemetry())
        elif self.path == "/api/nats-live" or self.path.startswith("/api/nats-live?"):
            self._json_response(_get_nats_live())
        elif self.path.startswith("/api/nrp/logs/"):
            pod_name = self.path.split("/api/nrp/logs/")[1].split("?")[0]
            self._json_response(_get_pod_logs(pod_name))
        elif self.path == "/api/nrp/mode" or self.path.startswith("/api/nrp/mode?"):
            self._json_response(_nrp_mode)
        elif self.path == "/api/erebus/memory" or self.path.startswith(
            "/api/erebus/memory?"
        ):
            self._json_response(_get_erebus_memory())
        elif self.path == "/api/erebus/status" or self.path.startswith(
            "/api/erebus/status?"
        ):
            self._json_response(_get_erebus_status())
        elif self.path == "/api/erebus/help" or self.path.startswith(
            "/api/erebus/help?"
        ):
            self._json_response(_get_erebus_help_queue())
        elif self.path.startswith("/api/erebus/activity"):
            from urllib.parse import parse_qs, urlparse

            qs = parse_qs(urlparse(self.path).query)
            since = int(qs.get("since", ["0"])[0])
            limit = min(int(qs.get("limit", ["200"])[0]), 500)
            self._json_response(_get_erebus_activity(since=since, limit=limit))
        elif self.path == "/api/primer/status":
            self._json_response(_get_primer_status())
        elif self.path == "/api/ieip/status" or self.path.startswith(
            "/api/ieip/status?"
        ):
            self._json_response(_get_ieip_status())
        elif self.path == "/api/ukg/status":
            self._json_response(_get_ukg_status())
        elif self.path == "/api/trends/erebus":
            self._json_response(_get_erebus_trends())
        elif self.path.startswith("/api/jobs/recent"):
            from urllib.parse import parse_qs, urlparse

            qs = parse_qs(urlparse(self.path).query)
            subsystem = qs.get("subsystem", ["scientist"])[0]
            limit = min(int(qs.get("limit", ["100"])[0]), 500)
            self._json_response(_get_lifecycle_recent(subsystem, limit))
        elif self.path == "/api/version":
            self._json_response(
                {
                    "sha": _ui_version_stamp(
                        os.path.join(STATIC_DIR, "schematic.html")
                    ).split(" ")[0],
                    "stamp": _ui_version_stamp(
                        os.path.join(STATIC_DIR, "schematic.html")
                    ),
                    "repo_dir": REPO_DIR,
                    "static_dir": STATIC_DIR,
                }
            )
        elif self.path.startswith("/api/"):
            self._json_response({})
        elif self.path.endswith(".html") or self.path == "/":
            # Serve HTML with {{UI_VERSION}} placeholder substitution so the
            # footer stamp reflects the currently-deployed commit. Falls
            # through to the default handler for non-HTML assets.
            rel = self.path.lstrip("/") or "index.html"
            fs_path = os.path.join(STATIC_DIR, rel)
            try:
                real = os.path.realpath(fs_path)
                if os.path.isfile(real) and real.endswith(".html"):
                    with open(real, "rb") as f:
                        body = f.read()
                    if b"{{UI_VERSION}}" in body:
                        stamp = _ui_version_stamp(real).encode("utf-8")
                        body = body.replace(b"{{UI_VERSION}}", stamp)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header(
                        "Cache-Control", "no-cache, no-store, must-revalidate"
                    )
                    self.end_headers()
                    self.wfile.write(body)
                    return
            except Exception:
                pass
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/visitors":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}
            email = data.get("email") or self.headers.get("X-Forwarded-Email", "")
            ip = data.get("ip") or self.headers.get(
                "X-Forwarded-For", self.client_address[0]
            )
            path = data.get("path", "/")
            ua = data.get("user_agent") or self.headers.get("User-Agent", "")
            if email:
                _log_visitor(email, ip, path, ua)
                log.info(f"Visitor logged: {email} from {ip}")
            self._json_response({"ok": True})
        elif self.path == "/api/training/start":
            self._handle_training_start()
        elif self.path == "/api/nrp/mode":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}
            new_mode = data.get("mode", "")
            if new_mode in ("auto", "heavy", "swarm"):
                _nrp_mode["mode"] = new_mode
                log.info(f"[nrp-watchdog] Mode set to: {new_mode}")
                self._json_response({"ok": True, "mode": new_mode})
            else:
                self._json_response({"error": "mode must be auto|heavy|swarm"}, 400)
        elif self.path == "/api/erebus/chat":
            self._handle_erebus_chat()
        elif self.path == "/api/erebus/result":
            self._handle_erebus_result()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_erebus_result(self):
        """HTTP bridge for Erebus worker results.

        Workers POST here: we (1) re-publish on Atlas-local NATS for live
        dispatchers, (2) write the attempt into arc_scientist_memory.json
        so the scientist sees vision attempts next cycle.
        """
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body)
        except Exception:
            self._json_response({"error": "bad json"}, 400)
            return
        task_id = data.get("task_id") or data.get("id")
        if not task_id:
            self._json_response({"error": "missing task_id"}, 400)
            return
        try:
            import asyncio
            import nats as _nats

            async def _publish():
                nc = await _nats.connect("nats://localhost:4222")
                await nc.publish(
                    f"erebus.results.{task_id}",
                    json.dumps(data, default=str).encode(),
                )
                await nc.drain()

            asyncio.run(_publish())
        except Exception as e:
            log.warning(f"erebus result publish failed: {e}")

        self._write_to_scientist_memory(data)
        self._json_response({"ok": True, "task_id": task_id})

    def _write_to_scientist_memory(self, data: dict) -> None:
        """Append an Attempt-shaped record to arc_scientist_memory.json."""
        task_num = data.get("task_num")
        if not task_num or "code" not in data:
            return
        mem_path = "/archive/neurogolf/arc_scientist_memory.json"
        try:
            with open(mem_path) as f:
                mem = json.load(f)
            tasks = mem.setdefault("tasks", {})
            tk = tasks.setdefault(
                str(task_num),
                {
                    "task_num": task_num,
                    "attempts": [],
                    "solved": False,
                    "best_correct": 0,
                    "best_total": 0,
                    "strategies_tried": [],
                    "failure_patterns": [],
                    "error_types": [],
                    "hypotheses": [],
                },
            )
            correct = int(data.get("correct") or 0)
            total = int(data.get("total") or 0)
            verified = correct == total and total > 0
            import datetime as _dt

            tk["attempts"].append(
                {
                    "task_num": task_num,
                    "timestamp": data.get("ts")
                    or _dt.datetime.utcnow().isoformat() + "Z",
                    "strategy": "vision",
                    "model": data.get("model") or "glm-4.1v",
                    "verified": verified,
                    "correct": correct,
                    "total": total,
                    "code": (data.get("code") or "")[:4000],
                    "error": (data.get("error") or "")[:200],
                }
            )
            if verified:
                tk["solved"] = True
            if correct > tk.get("best_correct", 0):
                tk["best_correct"] = correct
                tk["best_total"] = total
            with open(mem_path, "w") as f:
                json.dump(mem, f, indent=2)
            log.info(
                f"[vision-result] task{task_num:03d} -> {correct}/{total} solved={verified}"
            )
        except Exception as e:
            log.warning(f"[vision-result] memory write failed: {e}")

    def _handle_erebus_chat(self):
        """Handle chat with Erebus."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body)
        except Exception:
            data = {}
        message = data.get("message", "")
        if not message:
            self._json_response({"error": "no message"}, 400)
            return
        try:
            response = _erebus_chat(message)
            self._json_response({"response": response})
        except Exception as e:
            self._json_response({"error": str(e)[:200]}, 500)

    def _handle_training_start(self):
        """Start a manual training session (requires L3 privilege)."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body)
        except Exception:
            data = {}

        episodes = min(data.get("episodes", 10), 50)
        difficulty = min(data.get("difficulty", 2), 4)

        # Check privilege level from RAG server
        try:
            import urllib.request

            r = urllib.request.urlopen("http://localhost:8081/api/telemetry", timeout=3)
            rag = json.loads(r.read())
            priv_level = rag.get("ego_privileges", {}).get("current_level", 0)
        except Exception:
            priv_level = 0

        if priv_level < 3:
            self._json_response(
                {
                    "ok": False,
                    "error": "Requires L3 (EXECUTE) privilege",
                    "current_level": priv_level,
                },
                status=403,
            )
            return

        # Check if training is already running
        r = subprocess.run(
            ["tmux", "has-session", "-t", "train"],
            capture_output=True,
            timeout=3,
        )
        if r.returncode == 0:
            self._json_response(
                {"ok": False, "error": "Training session already running"},
                status=409,
            )
            return

        # Launch training in tmux
        cmd = (
            f"tmux new-session -d -s train "
            f"'cd /home/claude/agi-hpc && bash scripts/daily_training_session.sh "
            f"--episodes {episodes} --difficulty {difficulty}'"
        )
        subprocess.run(["bash", "-c", cmd], capture_output=True, timeout=5)
        log.info(
            "Manual training started: %d episodes, difficulty %d",
            episodes,
            difficulty,
        )
        self._json_response(
            {
                "ok": True,
                "episodes": episodes,
                "difficulty": difficulty,
                "message": (f"Training started ({episodes} eps, L{difficulty})"),
            }
        )

    def log_message(self, fmt, *args):
        first = str(args[0]) if args else ""
        if "/api/telemetry" not in first:
            log.info(fmt % args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    log.info(f"Atlas Telemetry Server on port {args.port}")
    log.info(f"Static dir: {STATIC_DIR}")

    # Prime the cache before serving so the first request isn't slow.
    log.info("Priming telemetry snapshot cache...")
    try:
        with _SNAPSHOT_LOCK:
            _SNAPSHOT["telemetry"] = build_telemetry()
            _SNAPSHOT["events"] = _build_events()
            _SNAPSHOT["ts"] = time.time()
    except Exception as e:
        log.warning("initial snapshot failed (will retry in background): %s", e)
    threading.Thread(target=_refresher, daemon=True, name="snapshot").start()
    threading.Thread(target=_nrp_nats_listener, daemon=True, name="nrp-nats").start()
    threading.Thread(
        target=_tail_erebus_logs, daemon=True, name="activity-tail"
    ).start()
    _start_nats_subscriber()

    # Pre-load Erebus fingerprints in background so first chat is fast
    def _preload_fingerprints():
        try:
            from pathlib import Path as _P
            import importlib.util

            _base = _P(__file__).parent.parent / "src" / "agi" / "autonomous"
            _sci_path = _base / "arc_scientist.py"
            if _sci_path.exists():
                import types
                import sys as _s

                for mod_name in ("agi", "agi.autonomous"):
                    if mod_name not in _s.modules:
                        _s.modules[mod_name] = types.ModuleType(mod_name)
                _prim_path = _base / "primitives.py"
                if _prim_path.exists():
                    spec = importlib.util.spec_from_file_location(
                        "agi.autonomous.primitives", _prim_path
                    )
                    mod = importlib.util.module_from_spec(spec)
                    _s.modules["agi.autonomous.primitives"] = mod
                    spec.loader.exec_module(mod)
                spec2 = importlib.util.spec_from_file_location(
                    "agi.autonomous.arc_scientist", _sci_path
                )
                mod2 = importlib.util.module_from_spec(spec2)
                _s.modules["agi.autonomous.arc_scientist"] = mod2
                spec2.loader.exec_module(mod2)
                fps = {}
                task_dir = "/archive/neurogolf"
                for tn in range(1, 401):
                    tf = _P(task_dir) / f"task{tn:03d}.json"
                    if tf.exists():
                        try:
                            with open(tf) as f:
                                tk = json.load(f)
                            fps[tn] = mod2.fingerprint_task(tk, tn)
                        except Exception:
                            pass
                _erebus_fingerprints["fps"] = fps
                log.info(f"Erebus: pre-loaded {len(fps)} fingerprints")
        except Exception as e:
            log.warning(f"Fingerprint preload failed: {e}")

    threading.Thread(
        target=_preload_fingerprints, daemon=True, name="erebus-fp"
    ).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), TelemetryHandler)
    server.serve_forever()
