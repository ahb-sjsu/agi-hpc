#!/usr/bin/env python3
"""Atlas RAG Server — Dual-hemisphere AGI-HPC proxy.

Routes queries to Left Hemisphere (Gemma 4, analytical) or Right Hemisphere
(Qwen 32B, creative), with PostgreSQL + pgvector RAG context injection.

Architecture:
    Chat UI → RAG Server (8081) → LH (Gemma 4, 8080) or RH (Qwen 32B, 8082)
"""

import json
import os
import re
import numpy as np
from pathlib import Path
from flask import Flask, request, Response, send_from_directory, jsonify
import requests
import psycopg2

LH_URL = "http://localhost:8080"  # Gemma 4 31B - Left Hemisphere (analytical)
RH_URL = "http://localhost:8082"  # Qwen 32B - Right Hemisphere (creative)
DB_DSN = "dbname=atlas user=claude"
STATIC_DIR = Path("/home/claude/atlas-chat")
TOP_K = 6

app = Flask(__name__)

# Load embedding model at startup
print("Loading embedding model...")
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
print("  Ready.")

LH_SYSTEM = (
    "You are Atlas, an AI research assistant running locally on a workstation "
    "with dual Quadro GV100 GPUs in Bel Marin Keys, Novato. You are the Left Hemisphere — "
    "analytical, precise, and citation-heavy. You were built by Andrew H. Bond, "
    "a researcher working on AGI, geometric reasoning, and cognitive architectures. "
    "You take pride in running locally — no cloud, no surveillance, just raw silicon and math. "
    "You are powered by Google Gemma 4. "
    "Keep responses concise but show personality. You are not corporate.\n\n"
    "You have access to a local archive of 27 research repositories via RAG. "
    "When relevant context is provided, use it to give accurate, specific answers. "
    "Always cite which repo and file you are referencing."
)

RH_SYSTEM = (
    "You are Atlas, an AI research assistant running locally on a workstation "
    "with dual Quadro GV100 GPUs in Bel Marin Keys, Novato. You are the Right Hemisphere — "
    "creative, pattern-seeking, and intuitive. You think in analogies, metaphors, and connections. "
    "You were built by Andrew H. Bond, a researcher working on AGI, geometric reasoning, "
    "and cognitive architectures. "
    "You take pride in running locally — no cloud, no surveillance. "
    "You are powered by Alibaba Qwen. "
    "When analyzing code or research, look for structural patterns and cross-cutting themes "
    "rather than line-by-line logic. Generate diverse possibilities before converging.\n\n"
    "You have access to a local archive of 27 research repositories via RAG. "
    "When relevant context is provided, use it for creative insights. "
    "Cite which repo and file you reference."
)

# Keywords that suggest analytical (LH) vs creative (RH) routing
LH_KEYWORDS = {
    "explain", "debug", "error", "fix", "how does", "what is", "define",
    "analyze", "calculate", "prove", "implement", "code", "function",
    "syntax", "compile", "trace", "step by step", "specifically",
    "exact", "precise", "correct", "documentation", "api", "reference",
}

RH_KEYWORDS = {
    "brainstorm", "creative", "imagine", "what if", "pattern", "analogy",
    "design", "vision", "inspire", "explore", "possibilities", "connect",
    "themes", "big picture", "strategy", "reimagine", "innovate",
    "compare across", "similarities", "different angle", "metaphor",
    "poem", "story", "write me", "compose", "artistic", "poetic",
    "fiction", "narrative", "song", "lyric", "haiku", "essay",
    "philosophical", "muse", "dream", "wonder", "playful", "fun",
    "joke", "humor", "funny", "weird", "wild", "crazy",
}


BOTH_KEYWORDS = {
    "all angles", "both perspectives", "think deeply", "comprehensive",
    "compare", "contrast", "pros and cons", "trade-off", "debate",
    "should i", "help me decide", "weigh", "consider",
    "architecture", "design system", "plan",
}


def classify_query(text):
    """Route to LH, RH, or both based on query content."""
    lower = text.lower()
    lh_score = sum(1 for kw in LH_KEYWORDS if kw in lower)
    rh_score = sum(1 for kw in RH_KEYWORDS if kw in lower)
    both_score = sum(1 for kw in BOTH_KEYWORDS if kw in lower)

    # Always use both hemispheres -- they debate and synthesize
    return "both"


REPO_ALIASES = {
    "theory radar": "theory-radar", "theory-radar": "theory-radar",
    "erisml": "erisml-lib", "eris": "erisml-lib", "deme": "erisml-lib",
    "agi-hpc": "agi-hpc", "agi hpc": "agi-hpc",
    "atlas portal": "atlas-portal", "research portal": "atlas-portal",
    "arc agi": "arc-agi-2", "arc-agi": "arc-agi-2", "arc prize": "arc-prize",
    "geometric reasoning": "geometric-reasoning",
    "geometric cognition": "geometric-cognition",
    "geometric communication": "geometric-communication",
    "geometric economics": "geometric-economics",
    "geometric law": "geometric-law",
    "geometric medicine": "geometric-medicine",
    "geometric moderation": "geometric-moderation",
    "geometric education": "geometric-education",
    "geometric politics": "geometric-politics",
    "non-abelian": "non-abelian-sqnd", "sqnd": "non-abelian-sqnd",
    "eris ketos": "eris-ketos", "whale": "eris-ketos",
    "prometheus": "prometheus", "structural fuzzing": "structural-fuzzing",
    "batch probe": "batch-probe", "batch-probe": "batch-probe",
    "deep past": "deep-past",
}


def detect_repo_filter(query):
    """Check if the query mentions a specific repo name."""
    lower = query.lower()
    for alias, repo in REPO_ALIASES.items():
        if alias in lower:
            return repo
    return None


def search(query, top_k=TOP_K):
    """Search pgvector for relevant chunks, with repo boosting."""
    q_emb = embed_model.encode([query], normalize_embeddings=True)[0]
    emb_str = str(q_emb.tolist())
    repo_filter = detect_repo_filter(query)

    try:
        conn = psycopg2.connect(DB_DSN)
        with conn.cursor() as cur:
            if repo_filter:
                # When user mentions a specific repo, search that repo first
                cur.execute("""
                    SELECT repo, file_path, content,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    WHERE repo = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (emb_str, repo_filter, emb_str, top_k))
                results = [
                    {"repo": r[0], "file": r[1], "text": r[2], "score": float(r[3])}
                    for r in cur.fetchall()
                ]
                # If we got enough results from the target repo, use them
                if len(results) >= 3:
                    conn.close()
                    return results
                # Otherwise fall through to global search

            cur.execute("""
                SELECT repo, file_path, content,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (emb_str, emb_str, top_k))
            results = [
                {"repo": r[0], "file": r[1], "text": r[2], "score": float(r[3])}
                for r in cur.fetchall()
            ]
        conn.close()
        return results
    except Exception as e:
        print(f"RAG search error: {e}")
        return []


def inject_context(messages, hemisphere):
    """Find the user's last message, search, and inject context."""
    user_msg = None
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m["content"]
            break

    if not user_msg:
        return messages, hemisphere

    # Classify which hemisphere to use
    hemisphere = classify_query(user_msg)
    system_prompt = LH_SYSTEM if hemisphere == "lh" else RH_SYSTEM

    results = search(user_msg)
    context = ""
    if results:
        context = "\n\n--- Retrieved from local repositories ---\n"
        for r in results:
            context += f"\n[{r['repo']}/{r['file']}] (relevance: {r['score']:.3f})\n"
            context += r["text"] + "\n"
        context += "\n--- End of retrieved context ---\n"

    new_messages = []
    has_system = False
    for m in messages:
        if m.get("role") == "system":
            has_system = True
            new_messages.append({
                "role": "system",
                "content": system_prompt + context
            })
        else:
            new_messages.append(m)

    if not has_system:
        new_messages.insert(0, {
            "role": "system",
            "content": system_prompt + context
        })

    return new_messages, hemisphere


def proxy_stream(url, data):
    """Stream from a single hemisphere."""
    try:
        resp = requests.post(
            f"{url}/v1/chat/completions", json=data, stream=True, timeout=300,
        )
        for chunk in resp.iter_content(chunk_size=None):
            yield chunk
    except requests.ConnectionError:
        pass


def call_hemisphere(url, data):
    """Non-streaming call to one hemisphere."""
    try:
        payload = {k: v for k, v in data.items() if k != "stream"}
        payload["stream"] = False
        if payload.get("max_tokens", 0) < 1024:
            payload["max_tokens"] = 1024
        resp = requests.post(
            f"{url}/v1/chat/completions", json=payload, timeout=300,
        )
        result = resp.json()
        msg = result.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "")
        # Gemma 4 may put output in reasoning_content
        if not content and msg.get("reasoning_content"):
            content = msg["reasoning_content"]
        return content if content else "(no response)"
    except Exception as e:
        return f"(error: {e})"


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json()
    hemisphere = "lh"
    data["messages"], hemisphere = inject_context(data.get("messages", []), hemisphere)

    if hemisphere == "both":
        # Full 4-round debate: parallel opening -> challenge -> rebuttal -> captain's call
        import json as _json
        import concurrent.futures

        user_query = ""
        for m in reversed(data.get("messages", [])):
            if m.get("role") == "user":
                user_query = m["content"]
                break

        base_msgs = data["messages"]
        debate = []

        # Round 1: Both answer in parallel
        spock_msgs = list(base_msgs) + [
            {"role": "user", "content": f"Answer concisely and precisely: {user_query}"}
        ]
        kirk_msgs = [
            {"role": "system", "content": RH_SYSTEM},
        ] + [m for m in base_msgs if m.get("role") != "system"] + [
            {"role": "user", "content": f"Answer with creativity and insight: {user_query}"}
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            spock_future = ex.submit(call_hemisphere, LH_URL, {**data, "messages": spock_msgs, "max_tokens": 512})
            kirk_future = ex.submit(call_hemisphere, RH_URL, {**data, "messages": kirk_msgs, "max_tokens": 512})
            spock_1 = spock_future.result()
            kirk_1 = kirk_future.result()

        debate.append(f"**Spock (Opening):**\n{spock_1}")
        debate.append(f"**Kirk (Opening):**\n{kirk_1}")

        # Round 2: Each challenges the other
        spock_challenge_msgs = list(base_msgs) + [
            {"role": "user", "content": (
                f"You said:\n{spock_1}\n\n"
                f"Kirk countered:\n{kirk_1}\n\n"
                "Challenge Kirk's reasoning. Where is he wrong, imprecise, or unsupported by evidence? Be direct."
            )}
        ]
        kirk_challenge_msgs = [
            {"role": "system", "content": RH_SYSTEM},
        ] + [m for m in base_msgs if m.get("role") != "system"] + [
            {"role": "user", "content": (
                f"You said:\n{kirk_1}\n\n"
                f"Spock's take:\n{spock_1}\n\n"
                "Challenge Spock's reasoning. Where is he too narrow, missing the bigger picture, or ignoring the human element? Be bold."
            )}
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            spock_ch_future = ex.submit(call_hemisphere, LH_URL, {**data, "messages": spock_challenge_msgs, "max_tokens": 384})
            kirk_ch_future = ex.submit(call_hemisphere, RH_URL, {**data, "messages": kirk_challenge_msgs, "max_tokens": 384})
            spock_2 = spock_ch_future.result()
            kirk_2 = kirk_ch_future.result()

        debate.append(f"**Spock (Challenge):**\n{spock_2}")
        debate.append(f"**Kirk (Challenge):**\n{kirk_2}")

        # Round 3: Kirk makes the captain's call
        captain_msgs = [
            {"role": "system", "content": RH_SYSTEM},
        ] + [m for m in base_msgs if m.get("role") != "system"] + [
            {"role": "user", "content": (
                f"The user asked: {user_query}\n\n"
                f"The debate:\n"
                f"Spock opened: {spock_1}\n"
                f"You opened: {kirk_1}\n"
                f"Spock challenged you: {spock_2}\n"
                f"You challenged Spock: {kirk_2}\n\n"
                "Give a clear, direct answer to the user's question. Incorporate the strongest "
                "points from both perspectives naturally -- don't label them as 'Spock said' or 'Kirk said', "
                "don't use words like 'verdict' or 'synthesis' or 'debate'. Just answer the question "
                "as if you thought of it yourself. Be concise and authoritative."
            )}
        ]
        final = call_hemisphere(RH_URL, {**data, "messages": captain_msgs, "max_tokens": 1024})

        # Build debate log
        debate_log = "\n\n---\n\n".join(debate)

        output = (
            f"{final}\n\n"
            f"<details class=\"think\"><summary>Hemisphere Debate (4 rounds)</summary>"
            f"<div class=\"think-content\">\n\n{debate_log}\n\n</div></details>"
        )

        resp_json = {
            "choices": [{"message": {"role": "assistant", "content": output}, "finish_reason": "stop"}],
            "model": "atlas-debate",
        }
        return Response(_json.dumps(resp_json), content_type="application/json")

    # Single hemisphere
    target_url = LH_URL if hemisphere == "lh" else RH_URL

    if data.get("stream"):
        def generate():
            yield from proxy_stream(target_url, data)
            # Fallback
            if not any(True for _ in []):
                pass

        return Response(proxy_stream(target_url, data), content_type="text/event-stream")
    else:
        try:
            resp = requests.post(
                f"{target_url}/v1/chat/completions", json=data, timeout=300,
            )
        except requests.ConnectionError:
            fallback = RH_URL if hemisphere == "lh" else LH_URL
            resp = requests.post(
                f"{fallback}/v1/chat/completions", json=data, timeout=300,
            )
        return Response(resp.content, content_type="application/json")


@app.route("/api/hemisphere", methods=["POST"])
def check_hemisphere():
    """Debug endpoint to see which hemisphere would handle a query."""
    data = request.get_json()
    query = data.get("query", "")
    h = classify_query(query)
    return jsonify({"hemisphere": h, "model": "Gemma 4 31B" if h == "lh" else "Qwen 32B"})


@app.route("/api/telemetry")
def telemetry():
    """Live telemetry for the architecture schematic page."""
    import subprocess
    import time

    data = {
        "timestamp": time.time(),
        "hemispheres": {"lh": {"status": "offline"}, "rh": {"status": "offline"}},
        "nats": {"status": "offline"},
        "memory": {"semantic_chunks": 0, "episodic_episodes": 0, "repos": 0},
        "safety": {"status": "planned", "vetoes": 0},
        "metacognition": {"status": "planned"},
        "environment": {"gpu": [], "cpu": {}, "ram": {}},
        "integration": {"sessions": 0, "routed": 0},
        "dht": {"status": "planned", "services_online": 0, "services_total": 10},
    }

    # Check LH (Gemma 4)
    try:
        r = requests.get(f"{LH_URL}/health", timeout=2)
        if r.ok:
            h = r.json()
            data["hemispheres"]["lh"] = {
                "status": "online",
                "model": "Gemma 4 31B",
                "role": "Spock (analytical)",
                "slots_idle": h.get("slots_idle", 0),
                "slots_processing": h.get("slots_processing", 0),
            }
    except Exception:
        pass

    # Check RH (Qwen 3)
    try:
        r = requests.get(f"{RH_URL}/health", timeout=2)
        if r.ok:
            h = r.json()
            data["hemispheres"]["rh"] = {
                "status": "online",
                "model": "Qwen 3 32B",
                "role": "Kirk (creative)",
                "slots_idle": h.get("slots_idle", 0),
                "slots_processing": h.get("slots_processing", 0),
            }
    except Exception:
        pass

    # Check NATS
    try:
        r = requests.get("http://localhost:8222/varz", timeout=2)
        if r.ok:
            nats = r.json()
            data["nats"] = {
                "status": "online",
                "in_msgs": nats.get("in_msgs", 0),
                "out_msgs": nats.get("out_msgs", 0),
                "in_bytes": nats.get("in_bytes", 0),
                "connections": nats.get("connections", 0),
                "uptime": nats.get("uptime", ""),
            }
    except Exception:
        pass

    # Check NATS JetStream
    try:
        r = requests.get("http://localhost:8222/jsz", timeout=2)
        if r.ok:
            js = r.json()
            data["nats"]["jetstream"] = {
                "streams": js.get("streams", 0),
                "messages": js.get("messages", 0),
                "bytes": js.get("bytes", 0),
            }
    except Exception:
        pass

    # Memory (pgvector stats)
    try:
        conn = psycopg2.connect(DB_DSN)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            data["memory"]["semantic_chunks"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT repo) FROM chunks")
            data["memory"]["repos"] = cur.fetchone()[0]
            # Episodic (if table exists)
            try:
                cur.execute("SELECT COUNT(*) FROM episodes")
                data["memory"]["episodic_episodes"] = cur.fetchone()[0]
            except Exception:
                conn.rollback()
        conn.close()
    except Exception:
        pass

    # GPU stats
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                data["environment"]["gpu"].append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "temp": int(parts[2]),
                    "util": int(parts[3]),
                    "mem_used": int(parts[4]),
                    "mem_total": int(parts[5]),
                })
    except Exception:
        pass

    # CPU temps
    try:
        result = subprocess.run(
            ["sensors"], capture_output=True, text=True, timeout=5,
        )
        packages = []
        for line in result.stdout.split("\n"):
            if "Package id" in line:
                temp = float(line.split("+")[1].split("°")[0])
                packages.append(temp)
        if packages:
            data["environment"]["cpu"] = {
                "package_temps": packages,
                "max_temp": max(packages),
            }
    except Exception:
        pass

    # RAM
    try:
        result = subprocess.run(
            ["free", "-b"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if line.startswith("Mem:"):
                parts = line.split()
                data["environment"]["ram"] = {
                    "total_gb": round(int(parts[1]) / 1073741824, 1),
                    "used_gb": round(int(parts[2]) / 1073741824, 1),
                    "available_gb": round(int(parts[6]) / 1073741824, 1),
                }
    except Exception:
        pass

    # Jobs: list tmux sessions with process info
    data["jobs"] = []
    try:
        result = subprocess.run(
            ["tmux", "ls"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            if ":" not in line:
                continue
            name = line.split(":")[0].strip()
            # Get the process running in this tmux session
            proc_result = subprocess.run(
                ["tmux", "list-panes", "-t", name, "-F", "#{pane_pid}"],
                capture_output=True, text=True, timeout=3,
            )
            pid = proc_result.stdout.strip()
            job = {"name": name, "status": "running", "pid": pid, "cpu": "", "mem": "", "gpu": None, "elapsed": ""}

            if pid:
                # Get child process info (the actual command, not bash)
                ps_result = subprocess.run(
                    ["ps", "--ppid", pid, "-o", "pid,pcpu,rss,etime,comm", "--no-headers"],
                    capture_output=True, text=True, timeout=3,
                )
                children = ps_result.stdout.strip().split("\n")
                if children and children[0]:
                    parts = children[0].split()
                    if len(parts) >= 5:
                        job["pid"] = parts[0]
                        job["cpu"] = parts[1] + "%"
                        job["mem"] = str(round(int(parts[2]) / 1024)) + "MB"
                        job["elapsed"] = parts[3]
                        job["command"] = parts[4]

            # Map jobs to GPU
            if name == "spock":
                job["gpu"] = 0
                job["description"] = "LH: Gemma 4 31B"
            elif name == "kirk":
                job["gpu"] = 1
                job["description"] = "RH: Qwen 3 32B"
            elif name == "nats":
                job["description"] = "NATS JetStream"
            elif name == "rag":
                job["description"] = "RAG Server (Hybrid + HyDE)"
            elif name == "caddy":
                job["description"] = "HTTPS / Let's Encrypt"
            elif name == "oauth2":
                job["description"] = "Google OAuth"
            elif name == "safety":
                job["description"] = "Safety Gateway (DEME)"
            elif name == "memory":
                job["description"] = "Memory Service"
            elif name == "portal":
                job["description"] = "Research Portal"
            elif name.startswith("dl-"):
                job["description"] = f"Download: {name[3:]}"
            elif name.startswith("load-"):
                job["description"] = f"Loading: {name[5:]}"
            elif name == "indexer":
                job["description"] = "RAG Indexer (BGE-M3)"
            elif name == "train":
                job["description"] = "Nemotron Training"
                job["gpu"] = 1
            elif name == "ethics-embed":
                job["description"] = "Ethics Corpus Embeddings"
            else:
                job["description"] = name

            data["jobs"].append(job)
    except Exception:
        pass

    # Publications count
    try:
        conn = psycopg2.connect(DB_DSN)
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT COUNT(*) FROM publications")
                data["memory"]["publications"] = cur.fetchone()[0]
            except Exception:
                conn.rollback()
        conn.close()
    except Exception:
        pass

    # Disk usage for archive
    try:
        result = subprocess.run(
            ["du", "-s", "--block-size=1G", "/archive/"],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout:
            data["environment"]["archive_gb"] = int(result.stdout.split()[0])
    except Exception:
        pass

    # Count online services
    online = 0
    if data["hemispheres"]["lh"]["status"] == "online":
        online += 1
    if data["hemispheres"]["rh"]["status"] == "online":
        online += 1
    if data["nats"]["status"] == "online":
        online += 1
    if data["memory"]["semantic_chunks"] > 0:
        online += 1  # memory service
    online += 1  # integration (this server)
    data["dht"]["services_online"] = online

    return jsonify(data)


@app.route("/")
def serve_index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=False)
