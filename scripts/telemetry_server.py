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
            s.sendall(b'SUB nrp.> 1\r\n')
            s.sendall(b'PING\r\n')
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
                        buf = buf[size + 2:]
                        try:
                            msg = json.loads(payload)
                        except Exception:
                            msg = {"raw": payload[:200]}
                        with _nrp_nats_lock:
                            if subject == "nrp.heartbeat":
                                _nrp_nats_cache["last_heartbeat"] = time.time()
                                _nrp_nats_cache["heartbeats"].append(msg)
                                _nrp_nats_cache["heartbeats"] = _nrp_nats_cache["heartbeats"][-10:]
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
        age = time.time() - _nrp_nats_cache["last_heartbeat"] if _nrp_nats_cache["last_heartbeat"] else None
        return {
            "leaf_alive": age is not None and age < 120,
            "last_heartbeat_age_s": round(age, 1) if age else None,
            "heartbeats": list(_nrp_nats_cache["heartbeats"]),
            "pods": dict(_nrp_nats_cache["pods"]),
        }


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
            ["kubectl", "--kubeconfig", kubeconfig, "-n", ns,
             "get", "jobs", "-o", "json"],
            capture_output=True, text=True, timeout=15,
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
                jobs_list.append({
                    "name": meta.get("name", ""),
                    "state": state,
                    "active": active,
                    "succeeded": succeeded,
                    "failed": failed,
                    "created": meta.get("creationTimestamp", ""),
                    "managed_by": meta.get("labels", {}).get(
                        "app.kubernetes.io/managed-by", ""),
                    "batch": meta.get("labels", {}).get(
                        "neurogolf.io/batch", ""),
                })

        # ── Pods ──
        pods_result = subprocess.run(
            ["kubectl", "--kubeconfig", kubeconfig, "-n", ns,
             "get", "pods", "-o", "json"],
            capture_output=True, text=True, timeout=15,
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

                pods_list.append({
                    "name": meta.get("name", ""),
                    "phase": phase,
                    "reason": reason,
                    "node": node_name,
                    "image": image.split("/")[-1] if image else "",
                    "resources": res,
                    "created": meta.get("creationTimestamp", ""),
                    "job": meta.get("labels", {}).get("job-name", ""),
                    "batch": batch,
                })

        # ── Pod metrics (kubectl top pods) ──
        usage_map = {}  # pod_name -> {cpu, memory}
        try:
            top_result = subprocess.run(
                ["kubectl", "--kubeconfig", kubeconfig, "-n", ns,
                 "top", "pods", "--no-headers"],
                capture_output=True, text=True, timeout=10,
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
        gpu_pods = [p for p in pods_list
                    if p.get("resources", {}).get("gpu")
                    and p.get("phase") == "Running"]
        for pod in gpu_pods[:8]:  # cap at 8 to avoid too many execs
            try:
                nv_result = subprocess.run(
                    ["kubectl", "--kubeconfig", kubeconfig, "-n", ns,
                     "exec", pod["name"], "--",
                     "nvidia-smi",
                     "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if nv_result.returncode == 0:
                    parts = [x.strip() for x in nv_result.stdout.strip().split(",")]
                    if len(parts) >= 4:
                        pod["resources"]["gpu_model"] = parts[0].replace(
                            "NVIDIA ", "").replace("GeForce ", "")
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
        jobs_list.sort(key=lambda j: (
            j["state"] not in ("Running", "Pending"), j["created"]
        ))
        pods_list.sort(key=lambda p: (
            p["phase"] not in ("Running", "Pending"), p["created"]
        ))

        out = {
            "succeeded": phases.get("Succeeded", 0),
            "running": phases.get("Running", 0),
            "pending": phases.get("Pending", 0) + phases.get("ContainerCreating", 0),
            "failed": phases.get("Failed", 0),
            "total": sum(phases.values()),
            "batches": [
                {"label": k, "total": v} for k, v in sorted(batches.items())
            ],
            "jobs": jobs_list,
            "pods": pods_list,
        }
        _nrp_cache["data"] = out
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
                nc = await nats_lib.connect("nats://localhost:4222", name="telemetry-live")
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
                    _NATS_LIVE_BUFFER.appendleft({
                        "ts": time.time(),
                        "subject": msg.subject,
                        "size": len(msg.data),
                        "summary": summary,
                        "reply": msg.reply or "",
                    })

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
                connections.append({
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
                })
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
                leafs.append({
                    "name": lf.get("name", ""),
                    "ip": lf.get("ip", ""),
                    "rtt": lf.get("rtt", ""),
                    "in_msgs": lf.get("in_msgs", 0),
                    "out_msgs": lf.get("out_msgs", 0),
                    "subscriptions": lf.get("subscriptions", 0),
                    "compression": lf.get("compression", ""),
                })
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


def _get_erebus_status():
    """Check if Erebus is running and get recent log."""
    import shutil
    status = {"running": False, "recent_log": [], "memory_summary": {}}
    try:
        result = subprocess.run(
            ["pgrep", "-f", "arc_scientist"],
            capture_output=True, text=True, timeout=5
        )
        status["running"] = result.returncode == 0
    except Exception:
        pass
    try:
        with open(EREBUS_LOG_PATH) as f:
            lines = f.readlines()
            status["recent_log"] = [l.rstrip() for l in lines[-20:]]
    except Exception:
        pass
    try:
        mem = _get_erebus_memory()
        status["memory_summary"] = {
            "total_attempts": mem.get("total_attempts", 0),
            "total_solves": mem.get("total_solves", 0),
            "tasks_explored": len(mem.get("tasks", {})),
            "strategies": {
                k: {"attempts": v.get("attempts", 0), "successes": v.get("successes", 0)}
                for k, v in mem.get("strategies", {}).items()
            },
        }
    except Exception:
        pass
    return status


EREBUS_HELP_PATH = "/archive/neurogolf/erebus_help_queue.json"


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

        memory_context = f"I have made {total_a} attempts and solved {total_s} tasks so far.\n"
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
        client = OpenAI(api_key=token, base_url="https://ellm.nrp-nautilus.io/v1")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        extra = {"chat_template_kwargs": {"thinking": False}}

        # Try agentic mode with tools
        try:
            from pathlib import Path as _P
            import importlib.util
            _tools_path = _P(__file__).parent.parent / "src" / "agi" / "autonomous" / "tools.py"
            if _tools_path.exists():
                _spec = importlib.util.spec_from_file_location("tools", _tools_path)
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)

                _sci_path = _P(__file__).parent.parent / "src" / "agi" / "autonomous" / "arc_scientist.py"
                _sci_spec = importlib.util.spec_from_file_location("arc_scientist", _sci_path)
                _sci_mod = importlib.util.module_from_spec(_sci_spec)
                _sci_spec.loader.exec_module(_sci_mod)

                mem = _sci_mod.EpisodicMemory(EREBUS_MEMORY_PATH)
                task_dir = "/archive/neurogolf"
                # Build fingerprints (cached after first call)
                if not hasattr(_erebus_chat, "_fingerprints"):
                    fps = {}
                    for tn in range(1, 401):
                        tf = _P(task_dir) / f"task{tn:03d}.json"
                        if tf.exists():
                            with open(tf) as f:
                                task = json.load(f)
                            fps[tn] = _sci_mod.fingerprint_task(task, tn)
                    _erebus_chat._fingerprints = fps
                executor = _mod.ToolExecutor(task_dir, mem, _erebus_chat._fingerprints)
                return _mod.run_agentic_turn(
                    client, "kimi", messages, executor,
                    max_tool_rounds=3, extra_body=extra)
        except Exception as tool_err:
            log.warning(f"Agentic mode failed, falling back: {tool_err}")

        # Fallback: simple chat without tools
        r = client.chat.completions.create(
            model="kimi", max_tokens=1024, messages=messages,
            extra_body=extra,
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
        elif self.path == "/api/nrp-telemetry" or self.path.startswith("/api/nrp-telemetry?"):
            self._json_response(_get_nrp_nats_telemetry())
        elif self.path == "/api/nats-live" or self.path.startswith("/api/nats-live?"):
            self._json_response(_get_nats_live())
        elif self.path == "/api/erebus/memory" or self.path.startswith("/api/erebus/memory?"):
            self._json_response(_get_erebus_memory())
        elif self.path == "/api/erebus/status" or self.path.startswith("/api/erebus/status?"):
            self._json_response(_get_erebus_status())
        elif self.path == "/api/erebus/help" or self.path.startswith("/api/erebus/help?"):
            self._json_response(_get_erebus_help_queue())
        elif self.path.startswith("/api/"):
            self._json_response({})
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
        elif self.path == "/api/erebus/chat":
            self._handle_erebus_chat()
        else:
            self.send_response(404)
            self.end_headers()

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
    _start_nats_subscriber()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), TelemetryHandler)
    server.serve_forever()
