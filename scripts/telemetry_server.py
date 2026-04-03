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
    for line in out.split("\n"):
        if line.startswith("Mem:"):
            p = line.split()
            return {
                "total_gb": round(int(p[1]) / 1073741824, 1),
                "used_gb": round(int(p[2]) / 1073741824, 1),
                "available_gb": round(int(p[6]) / 1073741824, 1),
            }
    return {}


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


def _get_services():
    services = {"lh": "offline", "rh": "offline", "safety": "planned",
                "metacognition": "planned", "dht": "planned"}
    for name, session in [("lh", "spock"), ("rh", "kirk"), ("safety", "safety"),
                          ("metacognition", "metacognition"), ("dht", "dht")]:
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
            "spock": "LH: Gemma 4 31B", "kirk": "RH: Qwen 3 32B",
            "nats": "NATS JetStream", "rag": "RAG Server",
            "caddy": "HTTPS / Let's Encrypt", "oauth2": "Google OAuth",
            "safety": "Safety Gateway", "memory": "Memory Service",
            "dht": "DHT Registry", "train": "Training",
            "indexer": "RAG Indexer", "embed": "Ethics Embedding",
        }
        if name in desc_map:
            job["description"] = desc_map[name]
        elif name.startswith("dl-"):
            job["description"] = f"Download: {name[3:]}"

        jobs.append(job)
    return jobs


def build_telemetry():
    gpus = _get_gpu()
    services = _get_services()
    memory = _get_db_counts()
    online = sum(1 for v in services.values() if v == "online") + 1  # +1 for integration

    return {
        "timestamp": time.time(),
        "hemispheres": {
            "lh": {"status": services["lh"], "model": "Gemma 4 31B", "role": "Spock"},
            "rh": {"status": services["rh"], "model": "Qwen 3 32B", "role": "Kirk"},
        },
        "nats": _get_nats(),
        "memory": memory,
        "safety": {"status": services["safety"], "vetoes": 0},
        "metacognition": {"status": services["metacognition"]},
        "environment": {"gpu": gpus, "cpu": _get_cpu(), "ram": _get_ram()},
        "integration": {"sessions": 0, "routed": 0},
        "dht": {"status": services["dht"], "services_online": online, "services_total": 10},
        "jobs": _get_jobs(),
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
