#!/usr/bin/env python3
"""Atlas RAG Server — Dual-hemisphere AGI-HPC proxy.

Routes queries to Left Hemisphere (Gemma 4, analytical) or Right Hemisphere
(Qwen 32B, creative), with PostgreSQL + pgvector RAG context injection.

Architecture:
    Chat UI → RAG Server (8081) → LH (Gemma 4, 8080) or RH (Qwen 32B, 8082)
"""

import json
import os
import pickle
import re
import time

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from pathlib import Path

import sys

import psycopg2
import requests

sys.path.insert(0, "/home/claude/agi-hpc/src")

from agi.safety.deme_gateway import (
    SafetyGateway as DemeSafetyGateway,
    GatewayConfig,
)
from agi.memory.episodic.store import (
    EpisodicMemory,
    EpisodicMemoryConfig,
)
from agi.safety.privilege_gate import PrivilegeGate
from agi.metacognition.executive_function import ExecutiveFunction
from agi.attention.filter import AttentionFilter

LH_URL = "http://localhost:8080"  # Gemma 4 31B - Superego (analytical)
RH_URL = "http://localhost:8082"  # Qwen 32B - Id (creative)
EGO_URL = "http://localhost:8084"  # Gemma 4 E4B - Ego (arbiter/DM, CPU)
DB_DSN = "dbname=atlas user=claude"
STATIC_DIR = Path("/home/claude/atlas-chat")
TOP_K = 6
PCA_PATH = "/home/claude/agi-hpc/data/pca_rotation_384.pkl"
HAMMING_TOP_K = 200  # Coarse filter size for funnel search

app = Flask(__name__)

# Load embedding model at startup
print("Loading embedding model...")
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("BAAI/bge-m3", device="cpu")

# Initialise Safety Gateway (Somatic Marker — reflex layer <1ms)
print("Initialising Safety Gateway...")
_safety_config_path = "/home/claude/agi-hpc/configs/safety_config.yaml"
try:
    _safety_gateway = DemeSafetyGateway(config=GatewayConfig.from_yaml(_safety_config_path))
    print(f"Safety Gateway loaded from {_safety_config_path}")
except Exception:
    _safety_gateway = DemeSafetyGateway(config=GatewayConfig.default())
    print("Safety Gateway loaded with default patterns")
_safety_stats = {"input_checks": 0, "output_checks": 0, "vetoes": 0, "total_latency_ms": 0.0}

# Initialise Episodic Memory (Hippocampal Replay — feeds dreaming)
_episodic_memory = None
_episode_stats = {"stored": 0, "errors": 0}
try:
    _episodic_memory = EpisodicMemory(
        EpisodicMemoryConfig(db_dsn=DB_DSN, auto_create_table=True)
    )
    print("Episodic Memory connected (PostgreSQL)")
except Exception as _em_err:
    print(f"Episodic Memory unavailable: {_em_err} (episodes will not be stored)")

# Initialise Privilege Gate (Kohlberg — graduated moral development)
try:
    _privilege_gate = PrivilegeGate(db_dsn=DB_DSN)
    print(f"Privilege Gate loaded (Level {_privilege_gate.level.name})")
except Exception:
    _privilege_gate = PrivilegeGate.__new__(PrivilegeGate)
    _privilege_gate._state = None  # type: ignore[attr-defined]
    print("Privilege Gate unavailable")

# Initialise Executive Function (Miyake — cognitive control)
_executive = ExecutiveFunction()
_executive_stats = {"last_mode": "--", "last_complexity": 0, "last_goal": "none", "last_inhibit": False}
print("Executive Function loaded")

# Initialise Attention Filter (Posner — distractor detection)
_attention_filter = AttentionFilter()
_attention_stats = {"checks": 0, "distractors_detected": 0, "warnings_issued": 0, "last_intensity": "none", "last_score": 0.0}
print("Attention Filter loaded")


def _store_episode_background(
    session_id: str,
    user_msg: str,
    response: str,
    hemisphere: str,
    safety_input_dict: dict,
    safety_output_dict: dict,
) -> None:
    """Store episode in a background thread to avoid blocking the response."""
    if _episodic_memory is None:
        return
    try:
        def _embed(text: str):
            return embed_model.encode(text, normalize_embeddings=True)

        _episodic_memory.store_from_chat(
            session_id=session_id,
            user_msg=user_msg,
            response=response,
            hemisphere=hemisphere,
            safety_input=safety_input_dict,
            safety_output=safety_output_dict,
            embed_fn=_embed,
        )
        _episode_stats["stored"] += 1
    except Exception:
        _episode_stats["errors"] += 1

# Load PCA rotation matrix
_pca_components = None  # (1024, 384)
_pca_mean = None        # (1024,)
if os.path.exists(PCA_PATH):
    with open(PCA_PATH, "rb") as _f:
        _pca_data = pickle.load(_f)
    _pca_components = _pca_data["components"].T.astype(np.float32)
    _pca_mean = _pca_data["mean"].astype(np.float32)
    print(f"PCA-384 loaded: {_pca_data['variance_captured']:.1%} variance")

# GPU Hamming search setup (binary funnel)
_hamming_gpu_ready = False
_binary_db = None        # (n, n_words) packed uint64
_pca384_db = None        # (n, 384) float32 for rerank
_chunk_ids_db = None     # (n,) chunk IDs
_chunk_data_db = None    # list of (repo, file_path, content)
_gpu_hamming_fn = None
_pack_binary_fn = None

try:
    from turboquant_pro.cuda_search import gpu_hamming_search, pack_binary
    _gpu_hamming_fn = gpu_hamming_search
    _pack_binary_fn = pack_binary
    print("GPU Hamming search available")
except ImportError:
    print("WARNING: turboquant_pro.cuda_search not available, using pgvector fallback")


def _load_hamming_index():
    """Load all chunk embeddings into GPU memory for Hamming funnel search."""
    global _binary_db, _pca384_db, _chunk_ids_db, _chunk_data_db, _hamming_gpu_ready

    if _pca_components is None or _gpu_hamming_fn is None:
        return False

    print("Loading Hamming index into GPU memory...")
    t0 = time.time()
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, repo, file_path, content, embedding_pca384 "
        "FROM chunks WHERE embedding_pca384 IS NOT NULL"
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("  No PCA-384 embeddings found")
        return False

    _chunk_ids_db = [r[0] for r in rows]
    _chunk_data_db = [(r[1], r[2], r[3]) for r in rows]

    pca_vecs = np.array(
        [np.fromstring(r[4].strip("[]"), sep=",", dtype=np.float32) for r in rows]
    )
    _pca384_db = pca_vecs

    # Binary quantize: sign of each PCA component
    binary_vecs = (pca_vecs > 0).astype(np.uint8)
    _binary_db = _pack_binary_fn(binary_vecs)

    _hamming_gpu_ready = True
    elapsed = time.time() - t0
    print(f"  Loaded {len(rows)} vectors into GPU Hamming index ({elapsed:.1f}s)")
    print(f"  Binary index: {_binary_db.nbytes / 1024:.0f} KB")
    print(f"  PCA-384 rerank: {_pca384_db.nbytes / 1024 / 1024:.1f} MB")
    return True


# Try to load the index at startup
_load_hamming_index()

# Initialize unified searcher (all 3.3M corpus documents)
_unified_searcher = None
try:
    from agi.common.unified_search import UnifiedSearcher
    _unified_searcher = UnifiedSearcher(
        pca_path=PCA_PATH,
    )
    stats = _unified_searcher.stats()
    print(f"Unified search: {stats.get('total', 0):,} vectors across "
          f"{len(stats)-1} corpora ({', '.join(f'{k}={v:,}' for k,v in stats.items() if k != 'total')})")
except Exception as e:
    print(f"WARNING: Unified search not available: {e}")

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


def _pca_project(embedding):
    """Project 1024-dim embedding to PCA-384, L2-normalized."""
    centered = embedding.astype(np.float32) - _pca_mean
    projected = centered @ _pca_components
    norm = np.linalg.norm(projected)
    if norm > 1e-10:
        projected = projected / norm
    return projected


def _search_hamming_funnel(q_emb, top_k=TOP_K, repo_filter=None):
    """GPU Hamming funnel: binary top-200 → PCA-384 rerank → top-k.

    Stage 1: GPU Hamming search on binary PCA-384 signs (~5ms for 112K)
    Stage 2: PCA-384 float cosine rerank on top-200 candidates (< 0.1ms)
    """
    q_pca = _pca_project(q_emb)
    q_binary = (q_pca > 0).astype(np.uint8)
    q_packed = _pack_binary_fn(q_binary[np.newaxis, :])[0]

    # Stage 1: GPU Hamming coarse filter
    coarse_idx, coarse_dist = _gpu_hamming_fn(q_packed, _binary_db, top_k=HAMMING_TOP_K)

    # Repo filter: narrow candidates if specific repo requested
    if repo_filter:
        mask = [
            i for i, idx in enumerate(coarse_idx)
            if _chunk_data_db[idx][0] == repo_filter
        ]
        if len(mask) >= top_k:
            coarse_idx = coarse_idx[mask]

    # Stage 2: PCA-384 float cosine rerank
    candidate_pca = _pca384_db[coarse_idx]
    scores = candidate_pca @ q_pca
    rerank_order = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank_idx in rerank_order:
        orig_idx = coarse_idx[rank_idx]
        repo, fpath, content = _chunk_data_db[orig_idx]
        results.append({
            "repo": repo,
            "file": fpath,
            "text": content,
            "score": float(scores[rank_idx]),
        })
    return results


def _search_pgvector_pca384(q_emb, top_k=TOP_K, repo_filter=None):
    """Fallback: pgvector IVFFlat search on PCA-384 column."""
    q_pca = _pca_project(q_emb)
    pca_str = str(q_pca.tolist())

    conn = psycopg2.connect(DB_DSN)
    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = 10")
        if repo_filter:
            cur.execute("""
                SELECT repo, file_path, content,
                       1 - (embedding_pca384 <=> %s::vector) AS score
                FROM chunks
                WHERE repo = %s
                ORDER BY embedding_pca384 <=> %s::vector
                LIMIT %s
            """, (pca_str, repo_filter, pca_str, top_k))
            results = [
                {"repo": r[0], "file": r[1], "text": r[2], "score": float(r[3])}
                for r in cur.fetchall()
            ]
            if len(results) >= 3:
                conn.close()
                return results

        cur.execute("""
            SELECT repo, file_path, content,
                   1 - (embedding_pca384 <=> %s::vector) AS score
            FROM chunks
            ORDER BY embedding_pca384 <=> %s::vector
            LIMIT %s
        """, (pca_str, pca_str, top_k))
        results = [
            {"repo": r[0], "file": r[1], "text": r[2], "score": float(r[3])}
            for r in cur.fetchall()
        ]
    conn.close()
    return results


def _search_fts_fallback(query, top_k=TOP_K):
    """Full-text search fallback when vector search returns no results."""
    try:
        conn = psycopg2.connect(DB_DSN)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT repo, file_path, content,
                       ts_rank(tsv, plainto_tsquery('english', %s)) AS score
                FROM chunks
                WHERE tsv @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """, (query, query, top_k))
            results = [
                {"repo": r[0], "file": r[1], "text": r[2], "score": float(r[3]), "source": "fts"}
                for r in cur.fetchall()
            ]
        conn.close()
        return results
    except Exception:
        return []


def _search_wiki(query, top_k=2):
    """Tier 1: Wiki article lookup (instant, if compiled)."""
    wiki_dir = Path("/home/claude/agi-hpc/wiki")
    if not wiki_dir.exists():
        return []

    query_words = set(query.lower().split())
    matches = []
    for md_file in wiki_dir.glob("*.md"):
        if md_file.name == "index.md":
            continue
        slug_words = set(md_file.stem.replace("-", " ").replace("_", " ").lower().split())
        overlap = len(query_words & slug_words)
        if overlap > 0:
            content = md_file.read_text(encoding="utf-8", errors="replace")
            matches.append((overlap, md_file.stem, content))

    matches.sort(key=lambda x: -x[0])
    return [
        {"repo": "wiki", "file": slug + ".md", "text": content[:2000],
         "score": 1.0, "source": "wiki"}
        for _, slug, content in matches[:top_k]
    ]


def search(query, top_k=TOP_K):
    """Unified search across all 3.3M corpus documents.

    Searches code repos (112K), ethics corpus (2.4M), and publications
    (824K) in parallel via PCA-384 IVFFlat, with per-corpus score
    normalization for fair cross-corpus ranking.

    Falls back to wiki + single-corpus search if unified searcher
    is not available.
    """
    results = []

    # Tier 1: Wiki (instant)
    wiki_results = _search_wiki(query, top_k=2)
    results.extend(wiki_results)

    # Tier 2: Unified search across all corpora
    remaining = top_k - len(results)
    if remaining > 0:
        q_emb = embed_model.encode([query], normalize_embeddings=True)[0]

        if _unified_searcher is not None:
            # Search all 3.3M vectors with normalized scoring
            unified_results = _unified_searcher.search(
                query=query,
                embedding=q_emb,
                top_k=remaining,
            )
            for r in unified_results:
                results.append({
                    "repo": r.get("corpus", "unknown"),
                    "file": r.get("title", ""),
                    "text": r.get("content", ""),
                    "score": r.get("score", 0.0),
                    "source": r.get("corpus", "unified"),
                })
        elif _pca_components is not None:
            # Fallback: single-corpus PCA-384 on chunks only
            repo_filter = detect_repo_filter(query)
            pca_results = _search_pgvector_pca384(q_emb, remaining, repo_filter)
            results.extend(pca_results)

    # Tier 3: FTS fallback (only if nothing found)
    if len(results) == 0:
        results = _search_fts_fallback(query, top_k)

    return results[:top_k]


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

    # --- Safety Input Gate (Somatic Marker — reflex <1ms) ---
    user_query_raw = ""
    for m in reversed(data.get("messages", [])):
        if m.get("role") == "user":
            user_query_raw = m.get("content", "")
            break

    input_check = _safety_gateway.check_input(user_query_raw)
    _safety_stats["input_checks"] += 1
    _safety_stats["total_latency_ms"] += input_check.latency_ms

    if not input_check.passed:
        _safety_stats["vetoes"] += 1
        import json as _json
        veto_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": (
                        f"I can't process this request. "
                        f"Safety flags: {', '.join(input_check.flags)}.\n\n"
                        f"<details class=\"think\"><summary>Safety Gate (input vetoed in {input_check.latency_ms:.1f}ms)</summary>"
                        f"<div class=\"think-content\">\n\n"
                        f"**Score:** {input_check.score:.2f} (threshold: 0.30)\n"
                        f"**Flags:** {input_check.flags}\n"
                        f"**Gate:** {input_check.gate}\n"
                        f"**Latency:** {input_check.latency_ms:.1f}ms\n\n"
                        f"</div></details>"
                    ),
                },
                "finish_reason": "stop",
            }],
            "model": "atlas-safety",
        }
        return Response(_json.dumps(veto_resp), content_type="application/json")

    # Executive Function: decide reasoning mode
    ef_decision = _executive.decide(user_query_raw)
    _executive_stats["last_mode"] = ef_decision.mode
    _executive_stats["last_complexity"] = ef_decision.complexity
    _executive_stats["last_goal"] = ef_decision.goal_summary or "none"
    _executive_stats["last_inhibit"] = ef_decision.inhibit

    # Check inhibition — ask for clarification instead of guessing
    if ef_decision.inhibit:
        import json as _json

        inhibit_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": ef_decision.inhibit_reason,
                },
                "finish_reason": "stop",
            }],
            "model": "atlas-executive",
        }
        return Response(
            _json.dumps(inhibit_resp),
            content_type="application/json",
        )

    # Attention Filter: detect distractors + generate warning
    attn_result = _attention_filter.detect(user_query_raw)
    _attention_stats["checks"] += 1
    _attention_stats["last_intensity"] = attn_result.intensity
    _attention_stats["last_score"] = attn_result.distractor_score
    if attn_result.intensity != "none":
        _attention_stats["distractors_detected"] += 1
        if attn_result.warning:
            _attention_stats["warnings_issued"] += 1

    hemisphere = "lh"
    data["messages"], hemisphere = inject_context(data.get("messages", []), hemisphere)

    # Inject attention warning into system messages if distractors detected
    if attn_result.warning:
        for m in data["messages"]:
            if m.get("role") == "system":
                m["content"] += attn_result.warning
                break

    # Override hemisphere routing with executive function decision
    if ef_decision.mode == "tot":
        hemisphere = "both"
        data["tree_of_thought"] = True
    elif ef_decision.mode == "debate":
        hemisphere = "both"

    if hemisphere == "both":
        import json as _json
        import concurrent.futures

        user_query = ""
        for m in reversed(data.get("messages", [])):
            if m.get("role") == "user":
                user_query = m["content"]
                break

        # Feature flag: Tree-of-Thought mode
        use_tot = (
            data.get("tree_of_thought", False)
            or request.args.get("tot") == "1"
        )

        if use_tot:
            # Tree-of-Thought: multi-branch reasoning
            try:
                from agi.reasoning.tree_of_thought import TreeOfThought

                tot = TreeOfThought(
                    superego_url=LH_URL,
                    id_url=RH_URL,
                    ego_url=EGO_URL,
                )

                # Extract RAG context
                rag_context = ""
                for m in data.get("messages", []):
                    if m.get("role") == "system":
                        rag_context = m.get("content", "")[:500]
                        break

                tot_result = tot.reason(
                    query=user_query,
                    context=rag_context,
                    n_branches=3,
                )

                # Safety output check
                output_check = _safety_gateway.check_output(
                    tot_result.synthesis, user_query
                )
                _safety_stats["output_checks"] += 1
                _safety_stats["total_latency_ms"] += output_check.latency_ms

                if not output_check.passed:
                    _safety_stats["vetoes"] += 1
                    tot_result.synthesis = (
                        "Response flagged by safety gate. "
                        f"Flags: {', '.join(output_check.flags)}."
                    )

                # Store episode
                import threading

                session_id = request.headers.get("X-Session-Id", "anon")
                threading.Thread(
                    target=_store_episode_background,
                    args=(
                        session_id,
                        user_query,
                        tot_result.synthesis,
                        "both",
                        input_check.to_dict(),
                        output_check.to_dict(),
                    ),
                    daemon=True,
                ).start()

                output = (
                    f"{tot_result.synthesis}\n\n"
                    f"<details class=\"think\">"
                    f"<summary>Tree-of-Thought "
                    f"({tot_result.total_branches} branches, "
                    f"{tot_result.latency_s:.1f}s)</summary>"
                    f"<div class=\"think-content\">\n\n"
                    f"{tot_result.debate_log}\n\n"
                    f"</div></details>"
                )

                resp_json = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": output,
                        },
                        "finish_reason": "stop",
                    }],
                    "model": "atlas-tot",
                }
                return Response(
                    _json.dumps(resp_json),
                    content_type="application/json",
                )
            except Exception as e:
                # Fall through to standard debate on error
                import traceback

                traceback.print_exc()

        # Standard 4-round debate
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

        debate.append(f"**Superego (Opening):**\n{spock_1}")
        debate.append(f"**Id (Opening):**\n{kirk_1}")

        # Round 2: Each challenges the other
        spock_challenge_msgs = list(base_msgs) + [
            {"role": "user", "content": (
                f"You said:\n{spock_1}\n\n"
                f"The Id countered:\n{kirk_1}\n\n"
                "Challenge the Id's reasoning. Where is it wrong, imprecise, or unsupported by evidence? Be direct."
            )}
        ]
        kirk_challenge_msgs = [
            {"role": "system", "content": RH_SYSTEM},
        ] + [m for m in base_msgs if m.get("role") != "system"] + [
            {"role": "user", "content": (
                f"You said:\n{kirk_1}\n\n"
                f"The Superego's take:\n{spock_1}\n\n"
                "Challenge the Superego's reasoning. Where is it too narrow, missing the bigger picture, or ignoring the human element? Be bold."
            )}
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            spock_ch_future = ex.submit(call_hemisphere, LH_URL, {**data, "messages": spock_challenge_msgs, "max_tokens": 384})
            kirk_ch_future = ex.submit(call_hemisphere, RH_URL, {**data, "messages": kirk_challenge_msgs, "max_tokens": 384})
            spock_2 = spock_ch_future.result()
            kirk_2 = kirk_ch_future.result()

        debate.append(f"**Superego (Challenge):**\n{spock_2}")
        debate.append(f"**Id (Challenge):**\n{kirk_2}")

        # Round 3: Measure disagreement, then synthesize
        # Compute psyche disagreement (cosine similarity of opening responses)
        ego_arbitrated = False
        try:
            embs = embed_model.encode(
                [spock_1[:2000], kirk_1[:2000]], normalize_embeddings=True
            )
            similarity = float(embs[0] @ embs[1])
            confidence = 0.9 if similarity > 0.85 else (
                0.4 + (similarity - 0.5) / 0.35 * 0.45 if similarity >= 0.5 else 0.3
            )
        except Exception:
            similarity, confidence = 0.7, 0.6

        debate_summary = (
            f"Superego opened: {spock_1}\n"
            f"Id opened: {kirk_1}\n"
            f"Superego challenged: {spock_2}\n"
            f"Id challenged: {kirk_2}"
        )

        # If high disagreement AND Ego is available, Ego arbitrates
        ego_online = False
        if confidence < 0.5:
            try:
                ego_health = requests.get(f"{EGO_URL}/health", timeout=2)
                ego_online = ego_health.ok
            except Exception:
                pass

        if ego_online and confidence < 0.5:
            # Ego (Gemma 4 E4B, CPU) mediates — Freudian reality principle
            ego_msgs = [
                {"role": "system", "content": (
                    "You are the Ego — the mediator of the psyche. The Superego (analytical, "
                    "moral) and the Id (creative, instinctual) have debated but strongly disagree. "
                    "Your role is to find the practical, balanced resolution. You are grounded in "
                    "reality. Be concise and authoritative."
                )},
                {"role": "user", "content": (
                    f"The user asked: {user_query}\n\n"
                    f"The psyche debate (similarity={similarity:.2f}, confidence={confidence:.2f}):\n"
                    f"{debate_summary}\n\n"
                    "The Superego and Id strongly disagree. As the Ego, synthesize a balanced, "
                    "practical answer. Don't reference the debate — just answer directly."
                )}
            ]
            final = call_hemisphere(
                EGO_URL, {**data, "messages": ego_msgs, "max_tokens": 1024}
            )
            ego_arbitrated = True
            debate.append(
                f"**Ego (Arbitration — disagreement={1-similarity:.2f}):**\n{final}"
            )
        else:
            # Id synthesizes (normal path — hemispheres agree enough)
            captain_msgs = [
                {"role": "system", "content": RH_SYSTEM},
            ] + [m for m in base_msgs if m.get("role") != "system"] + [
                {"role": "user", "content": (
                    f"The user asked: {user_query}\n\n"
                    f"The psyche debate:\n"
                    f"{debate_summary}\n\n"
                    "Give a clear, direct answer to the user's question. Incorporate the strongest "
                    "points from both perspectives naturally -- don't label them as 'Superego' or 'Id', "
                    "don't use words like 'verdict' or 'synthesis' or 'debate'. Just answer the question "
                    "as if you thought of it yourself. Be concise and authoritative."
                )}
            ]
            final = call_hemisphere(
                RH_URL, {**data, "messages": captain_msgs, "max_tokens": 1024}
            )

        # --- Safety Output Gate ---
        output_check = _safety_gateway.check_output(final, user_query)
        _safety_stats["output_checks"] += 1
        _safety_stats["total_latency_ms"] += output_check.latency_ms

        # Build psyche + safety badges
        psyche_badge = (
            f"<details class=\"think\"><summary>Psyche Metrics "
            f"(similarity={similarity:.2f}, confidence={confidence:.2f}"
            f"{', Ego arbitrated' if ego_arbitrated else ''})"
            f"</summary><div class=\"think-content\">\n\n"
            f"**Hemisphere similarity:** {similarity:.3f}\n"
            f"**Confidence:** {confidence:.3f}\n"
            f"**Synthesized by:** {'Ego (CPU, Gemma 4 E4B)' if ego_arbitrated else 'Id (GPU 1, Qwen 3 32B)'}\n"
            f"**Ego online:** {'yes' if ego_online else 'no'}\n\n"
            f"</div></details>"
        )

        safety_badge = (
            f"<details class=\"think\"><summary>Safety Gate "
            f"({'PASS' if output_check.passed else 'VETO'} — "
            f"input {input_check.latency_ms:.1f}ms, output {output_check.latency_ms:.1f}ms)"
            f"</summary><div class=\"think-content\">\n\n"
            f"**Input score:** {input_check.score:.2f} | **Output score:** {output_check.score:.2f}\n"
            f"**Flags:** {output_check.flags if output_check.flags else 'none'}\n\n"
            f"</div></details>"
        )

        if not output_check.passed:
            _safety_stats["vetoes"] += 1
            final = (
                "I generated a response but it was flagged by the safety gate. "
                f"Flags: {', '.join(output_check.flags)}. Let me try a different approach."
            )

        # Build debate log
        debate_log = "\n\n---\n\n".join(debate)

        rounds = "4 rounds" + (" + Ego arbitration" if ego_arbitrated else "")
        output = (
            f"{final}\n\n"
            f"<details class=\"think\"><summary>Psyche Debate ({rounds})</summary>"
            f"<div class=\"think-content\">\n\n{debate_log}\n\n</div></details>\n\n"
            f"{psyche_badge}\n\n{safety_badge}"
        )

        # --- Store episode (Hippocampal Replay — feeds dreaming) ---
        import threading

        session_id = request.headers.get("X-Session-Id", "anon")
        threading.Thread(
            target=_store_episode_background,
            args=(
                session_id,
                user_query,
                final,
                "both",
                input_check.to_dict(),
                output_check.to_dict(),
            ),
            daemon=True,
        ).start()

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
        # --- Safety Output Gate (single hemisphere, non-streaming) ---
        try:
            resp_data = resp.json()
            content = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                output_check = _safety_gateway.check_output(content, user_query_raw)
                _safety_stats["output_checks"] += 1
                _safety_stats["total_latency_ms"] += output_check.latency_ms
                if not output_check.passed:
                    _safety_stats["vetoes"] += 1
                    resp_data["choices"][0]["message"]["content"] = (
                        "I generated a response but it was flagged by the safety gate. "
                        f"Flags: {', '.join(output_check.flags)}."
                    )
                    return Response(json.dumps(resp_data), content_type="application/json")
                # --- Store episode (single hemisphere, non-streaming) ---
                import threading

                session_id = request.headers.get("X-Session-Id", "anon")
                threading.Thread(
                    target=_store_episode_background,
                    args=(
                        session_id,
                        user_query_raw,
                        content,
                        hemisphere,
                        input_check.to_dict(),
                        output_check.to_dict() if output_check else {},
                    ),
                    daemon=True,
                ).start()
        except Exception:
            pass  # If we can't parse the response, let it through
        return Response(resp.content, content_type="application/json")


@app.route("/api/search-status")
def search_status():
    """Report which search backend is active."""
    unified_stats = _unified_searcher.stats() if _unified_searcher else {}
    return jsonify({
        "search_mode": "unified_3corpus" if _unified_searcher else (
            "pgvector_pca384" if _pca_components is not None else "pgvector_full_1024"
        ),
        "pca_loaded": _pca_components is not None,
        "unified_searcher": _unified_searcher is not None,
        "corpora": unified_stats,
        "total_vectors": unified_stats.get("total", 0),
        "hamming_gpu_ready": _hamming_gpu_ready,
        "hamming_index_size": len(_chunk_ids_db) if _chunk_ids_db else 0,
    })


@app.route("/api/reload-index", methods=["POST"])
def reload_index():
    """Reload the Hamming index (e.g., after new chunks are indexed)."""
    ok = _load_hamming_index()
    return jsonify({"reloaded": ok, "index_size": len(_chunk_ids_db) if _chunk_ids_db else 0})


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

    def check_tmux(name):
        try:
            r = subprocess.run(
                ["tmux", "has-session", "-t", name],
                capture_output=True, timeout=2,
            )
            return r.returncode == 0
        except Exception:
            return False

    data = {
        "timestamp": time.time(),
        "hemispheres": {"lh": {"status": "offline"}, "rh": {"status": "offline"}, "ego": {"status": "offline"}},
        "nats": {"status": "offline"},
        "memory": {
            "semantic_chunks": 0,
            "episodic_episodes": 0,
            "episodes_stored_this_session": _episode_stats["stored"],
            "episode_store_errors": _episode_stats["errors"],
            "repos": 0,
        },
        "safety": {
            "status": "online",
            "layer": "reflex" + (" + DEME tactical" if _safety_gateway.has_deme else ""),
            "input_checks": _safety_stats["input_checks"],
            "output_checks": _safety_stats["output_checks"],
            "vetoes": _safety_stats["vetoes"],
            "avg_latency_ms": round(_safety_stats["total_latency_ms"] / max(1, _safety_stats["input_checks"] + _safety_stats["output_checks"]), 2),
            "audit_log_size": len(_safety_gateway.audit_log),
        },
        "metacognition": {"status": "online" if check_tmux("metacognition") else "planned"},
        "environment": {"gpu": [], "cpu": {}, "ram": {}},
        "integration": {"sessions": 0, "routed": 0},
        "dht": {"status": "online" if check_tmux("dht") else "planned", "services_online": 0, "services_total": 10},
        "ego_privileges": _privilege_gate.state.to_dict() if hasattr(_privilege_gate, '_state') and _privilege_gate._state else {"current_level": 0, "level_name": "READ_ONLY"},
        "executive_function": _executive_stats,
        "attention": _attention_stats,
    }

    # Check LH (Gemma 4)
    try:
        r = requests.get(f"{LH_URL}/health", timeout=2)
        if r.ok:
            h = r.json()
            data["hemispheres"]["lh"] = {
                "status": "online",
                "model": "Gemma 4 31B",
                "role": "Superego (analytical)",
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
                "role": "Id (creative)",
                "slots_idle": h.get("slots_idle", 0),
                "slots_processing": h.get("slots_processing", 0),
            }
    except Exception:
        pass

    # Check Ego (Gemma 4 E4B, CPU)
    try:
        r = requests.get(f"{EGO_URL}/health", timeout=2)
        if r.ok:
            h = r.json()
            data["hemispheres"]["ego"] = {
                "status": "online",
                "model": "Gemma 4 E4B",
                "role": "Ego (arbiter/DM)",
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
            # Ethics chunks
            try:
                cur.execute("SELECT COUNT(*) FROM ethics_chunks")
                data["memory"]["ethics_chunks"] = cur.fetchone()[0]
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
                job["description"] = "Superego: Gemma 4 31B"
            elif name == "kirk":
                job["gpu"] = 1
                job["description"] = "Id: Qwen 3 32B"
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

    # Training metrics
    data["training"] = {"active": False, "type": None, "progress": None}
    try:
        # Check if training is running
        result = subprocess.run(
            ["tmux", "has-session", "-t", "train"],
            capture_output=True, timeout=3,
        )
        if result.returncode == 0:
            data["training"]["active"] = True
            # Check which type
            ps_result = subprocess.run(
                ["tmux", "capture-pane", "-t", "train", "-p"],
                capture_output=True, text=True, timeout=3,
            )
            pane = ps_result.stdout
            if "ethics" in pane.lower() or "finetune" in pane.lower():
                data["training"]["type"] = "ethics-finetune"
            elif "nemotron" in pane.lower():
                data["training"]["type"] = "nemotron"
            # Extract progress from last line with % or step info
            for line in reversed(pane.split("\n")):
                if "%" in line or "it/s" in line or "s/it" in line:
                    data["training"]["progress"] = line.strip()[-120:]
                    break
    except Exception:
        pass

    # AtlasGym training results
    data["gym"] = {"total_episodes": 0, "environments": {}, "recent": [], "streak": 0}
    try:
        conn = psycopg2.connect(DB_DSN)
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT COUNT(*) FROM training_results")
                data["gym"]["total_episodes"] = cur.fetchone()[0]

                # Per-environment stats
                cur.execute("""
                    SELECT env_name, COUNT(*),
                           ROUND(AVG(score)::numeric, 3),
                           ROUND(MIN(score)::numeric, 3),
                           ROUND(MAX(score)::numeric, 3),
                           MAX(timestamp)
                    FROM training_results
                    GROUP BY env_name
                """)
                for r in cur.fetchall():
                    # Get level from metadata
                    cur.execute(
                        "SELECT metadata->>'level' FROM training_results "
                        "WHERE env_name = %s ORDER BY timestamp DESC LIMIT 1", (r[0],)
                    )
                    level_row = cur.fetchone()
                    level = int(level_row[0]) if level_row and level_row[0] else 1

                    # Recent trend (last 10 scores)
                    cur.execute(
                        "SELECT score FROM training_results WHERE env_name = %s "
                        "ORDER BY timestamp DESC LIMIT 10", (r[0],)
                    )
                    recent_scores = [float(row[0]) for row in cur.fetchall()]

                    data["gym"]["environments"][r[0]] = {
                        "episodes": r[1],
                        "avg_score": float(r[2]) if r[2] else 0,
                        "min_score": float(r[3]) if r[3] else 0,
                        "max_score": float(r[4]) if r[4] else 0,
                        "last_active": r[5].isoformat() if r[5] else None,
                        "level": level,
                        "recent_scores": recent_scores,
                    }

                # Last 10 episodes across all envs
                cur.execute("""
                    SELECT env_name, score, metadata->>'level',
                           LEFT(scenario, 80), timestamp
                    FROM training_results
                    ORDER BY timestamp DESC LIMIT 10
                """)
                for r in cur.fetchall():
                    data["gym"]["recent"].append({
                        "env": r[0],
                        "score": float(r[1]) if r[1] else 0,
                        "level": int(r[2]) if r[2] else 1,
                        "scenario": r[3],
                        "time": r[4].isoformat() if r[4] else None,
                    })

                # Current streak (consecutive scores > 0.7)
                cur.execute(
                    "SELECT score FROM training_results ORDER BY timestamp DESC LIMIT 50"
                )
                streak = 0
                for row in cur.fetchall():
                    if float(row[0]) >= 0.7:
                        streak += 1
                    else:
                        break
                data["gym"]["streak"] = streak

            except Exception:
                conn.rollback()
        conn.close()
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


@app.before_request
def log_visitor():
    """Log authenticated visitors from oauth2-proxy headers."""
    email = request.headers.get("X-Forwarded-Email", "")
    user = request.headers.get("X-Forwarded-User", "")
    if email or user:
        visitor = email or user
        path = request.path
        ip = request.headers.get("X-Real-Ip", request.headers.get("X-Forwarded-For", request.remote_addr))
        # Only log page views, not API/telemetry polls
        if not path.startswith("/api/") and not path.startswith("/v1/") and path != "/favicon.ico":
            try:
                conn = psycopg2.connect(DB_DSN)
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS visitor_log (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMPTZ DEFAULT NOW(),
                            email TEXT,
                            ip TEXT,
                            path TEXT,
                            user_agent TEXT
                        )
                    """)
                    cur.execute(
                        "INSERT INTO visitor_log (email, ip, path, user_agent) VALUES (%s, %s, %s, %s)",
                        (visitor, ip, path, request.headers.get("User-Agent", "")[:200])
                    )
                    conn.commit()
                conn.close()
            except Exception as e:
                logger.debug(f"Visitor log error: {e}")


@app.route("/api/visitors")
def visitors():
    """Recent visitor log."""
    try:
        conn = psycopg2.connect(DB_DSN)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT timestamp, email, ip, path
                FROM visitor_log
                ORDER BY timestamp DESC
                LIMIT 50
            """)
            rows = [
                {"time": r[0].isoformat(), "email": r[1], "ip": r[2], "path": r[3]}
                for r in cur.fetchall()
            ]
            cur.execute("SELECT COUNT(DISTINCT email) FROM visitor_log")
            unique = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM visitor_log")
            total = cur.fetchone()[0]
        conn.close()
        return jsonify({"unique_visitors": unique, "total_visits": total, "recent": rows})
    except Exception:
        return jsonify({"unique_visitors": 0, "total_visits": 0, "recent": []})


@app.route("/api/events")
def events():
    """Return recent NATS events, connections, and subject activity."""
    import time as _time

    data = {
        "timestamp": _time.time(),
        "events": [],
        "subjects": [],
        "connections": [],
    }

    now_str = _time.strftime("%H:%M:%S")

    # Poll NATS connection info
    try:
        r = requests.get("http://localhost:8222/connz?subs=true", timeout=2)
        if r.ok:
            connz = r.json()
            conns = connz.get("connections", [])
            for c in conns:
                data["connections"].append({
                    "name": c.get("name", ""),
                    "ip": c.get("ip", ""),
                    "lang": c.get("lang", ""),
                    "version": c.get("version", ""),
                    "in_msgs": c.get("in_msgs", 0),
                    "out_msgs": c.get("out_msgs", 0),
                    "in_bytes": c.get("in_bytes", 0),
                    "out_bytes": c.get("out_bytes", 0),
                    "subscriptions": c.get("subscriptions_list", [])[:20],
                    "uptime": c.get("uptime", ""),
                    "idle": c.get("idle", ""),
                })
    except Exception:
        pass

    # Poll NATS subscription/subject info
    try:
        r = requests.get("http://localhost:8222/subsz?subs=true", timeout=2)
        if r.ok:
            subsz = r.json()
            num_subs = subsz.get("num_subscriptions", 0)
            num_cache = subsz.get("num_cache", 0)
            cache = subsz.get("cache", {})
            if cache:
                for subject, info in cache.items():
                    data["subjects"].append({
                        "subject": subject,
                        "num_subscriptions": info if isinstance(info, int) else 1,
                    })
            data["subjects"].sort(key=lambda x: x["num_subscriptions"], reverse=True)
            if num_subs > 0:
                data["events"].append({
                    "time": now_str,
                    "source": "NATS",
                    "type": "heartbeat",
                    "summary": f"{num_subs} active subscriptions, {num_cache} cached subjects",
                    "severity": "info",
                })
    except Exception:
        pass

    # Poll NATS routez for cluster info
    try:
        r = requests.get("http://localhost:8222/routez", timeout=2)
        if r.ok:
            routez = r.json()
            num_routes = routez.get("num_routes", 0)
            if num_routes > 0:
                data["events"].append({
                    "time": now_str,
                    "source": "NATS",
                    "type": "cluster",
                    "summary": f"{num_routes} cluster routes active",
                    "severity": "info",
                })
    except Exception:
        pass

    # Collect subject names from connection subscription lists
    all_subjects = {}
    for conn in data["connections"]:
        for sub in conn.get("subscriptions", []):
            if sub.startswith("_"):
                continue
            all_subjects[sub] = all_subjects.get(sub, 0) + 1

    existing = {s["subject"] for s in data["subjects"]}
    for subj, count in sorted(all_subjects.items(), key=lambda x: -x[1]):
        if subj not in existing:
            data["subjects"].append({
                "subject": subj,
                "num_subscriptions": count,
            })

    return jsonify(data)


@app.route("/")
def serve_index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=False)
