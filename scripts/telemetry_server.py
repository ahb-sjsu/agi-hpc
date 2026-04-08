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
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("telemetry")

STATIC_DIR = os.environ.get("ATLAS_STATIC", "/home/claude/atlas-chat")
PORT = int(os.environ.get("TELEMETRY_PORT", "8085"))
DB_DSN = os.environ.get("ATLAS_DB_DSN", "dbname=atlas user=claude")


def _run(cmd, timeout=3):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def _get_gpu():
    gpus = []
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    for line in out.split("\n"):
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 6:
            gpus.append({
                "index": int(p[0]), "name": p[1], "temp": int(p[2]),
                "util": int(p[3]), "mem_used": int(p[4]), "mem_total": int(p[5]),
            })
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
                result[f"disk_{key}_total_gb"] = int(
                    parts[1].rstrip("G")
                )
                result[f"disk_{key}_used_gb"] = int(
                    parts[2].rstrip("G")
                )
                result[f"disk_{key}_avail_gb"] = int(
                    parts[3].rstrip("G")
                )
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
                    result["net_rx_gb"] = round(
                        int(nums[0]) / 1073741824, 2
                    )
                    result["net_tx_gb"] = round(
                        int(nums[8]) / 1073741824, 2
                    )
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
    counts = {"semantic_chunks": 0, "repos": 0, "episodic_episodes": 0,
              "ethics_chunks": 0, "publications": 0}
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
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE embedding_pca384 IS NOT NULL")
                compressed = cur.fetchone()[0]
                if compressed > 0:
                    stats["enabled"] = True
                    orig_mb = round(total * 1024 * 4 / 1048576, 1)
                    comp_mb = round(compressed * 384 * 4 / 1048576, 1)
                    stats["tables"][table] = {
                        "total": total, "compressed": compressed,
                        "original_mb": orig_mb, "compressed_mb": comp_mb,
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
    services = {"lh": "offline", "rh": "offline", "ego": "offline",
                "safety": "offline", "metacognition": "offline", "dht": "offline"}
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
    for name, session in [("safety", "safety"),
                          ("metacognition", "metacognition"), ("dht", "dht")]:
        if services[name] != "online":
            r = subprocess.run(["tmux", "has-session", "-t", session],
                               capture_output=True, timeout=2)
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

        ps_out = _run(["ps", "--ppid",
                        _run(["tmux", "list-panes", "-t", name, "-F", "#{pane_pid}"]),
                        "-o", "pid,pcpu,rss,etime,comm", "--no-headers"])
        if ps_out:
            parts = ps_out.split()
            if len(parts) >= 5:
                job["cpu"] = parts[1] + "%"
                job["mem"] = str(round(int(parts[2]) / 1024)) + "MB"
                job["elapsed"] = parts[3]

        desc_map = {
            "spock": ("Superego: Gemma 4 31B", 0),
            "kirk": ("Id: Qwen 3 32B", 1),
            "ego": ("Ego: Gemma 4 E4B", None),
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
        ("atlas-ego", "Ego: Gemma 4 E4B", None),
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
                            job["mem"] = (
                                str(round(int(parts[1]) / 1024))
                                + "MB"
                            )
                            job["elapsed"] = parts[2]
                jobs.append(job)
        except Exception:
            pass

    return jobs


def _get_rag_telemetry():
    """Fetch enriched telemetry from the RAG server (short timeout)."""
    try:
        import urllib.request
        r = urllib.request.urlopen(
            "http://localhost:8081/api/telemetry", timeout=2
        )
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
                "SELECT COUNT(*), AVG(score), MAX(timestamp) "
                "FROM training_results"
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

        conn.close()
    except Exception:
        pass
    return stats


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
                    "Role-played by Superego"
                    if services["rh"] == "fallback"
                    else ""
                ),
            },
            "ego": {
                "status": services["ego"],
                "model": "Gemma 4 E4B",
                "role": "Ego (arbiter/DM)",
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
        "ego_privileges": rag_privs if rag_privs else {
            "current_level": 0,
            "level_name": "READ_ONLY",
        },
        "system_deep": _get_system_deep(),
        "attention": rag_data.get("attention", {"checks": 0, "last_intensity": "none"}),
        "training": _get_training_stats(),
        "dreaming": _get_dreaming_stats(),
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
            "FROM visitor_log ORDER BY timestamp DESC LIMIT %s", (limit,))
        for row in cur.fetchall():
            visitors.append({
                "email": row[0] or "",
                "ip": row[1] or "",
                "path": row[2] or "",
                "user_agent": row[3] or "",
                "timestamp": row[4].isoformat() if row[4] else "",
                "type": "login",
            })
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
            "INSERT INTO visitor_log (email, ip, path, user_agent) VALUES (%s, %s, %s, %s)",
            (email, ip, path, user_agent[:200] if user_agent else ""))
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
        events.append({
            "type": "login",
            "message": f"{name} logged in",
            "detail": v.get("path", "/"),
            "timestamp": v.get("timestamp", ""),
            "color": "var(--green)",
        })
    # NATS stats as system events
    nats = _get_nats()
    if nats.get("status") == "online":
        events.append({
            "type": "system",
            "message": f"NATS: {nats.get('in_msgs', 0)} msgs in, {nats.get('connections', 0)} connections",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "color": "var(--accent)",
        })
    return events


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
            self._json_response(build_telemetry())
        elif self.path == "/api/events" or self.path.startswith("/api/events?"):
            self._json_response(_build_events())
        elif self.path == "/api/visitors" or self.path.startswith("/api/visitors?"):
            self._json_response(_get_visitors())
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
            ip = data.get("ip") or self.headers.get("X-Forwarded-For", self.client_address[0])
            path = data.get("path", "/")
            ua = data.get("user_agent") or self.headers.get("User-Agent", "")
            if email:
                _log_visitor(email, ip, path, ua)
                log.info(f"Visitor logged: {email} from {ip}")
            self._json_response({"ok": True})
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        if "/api/telemetry" not in (args[0] if args else ""):
            log.info(fmt % args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    log.info(f"Atlas Telemetry Server on port {args.port}")
    log.info(f"Static dir: {STATIC_DIR}")
    server = HTTPServer(("0.0.0.0", args.port), TelemetryHandler)
    server.serve_forever()
