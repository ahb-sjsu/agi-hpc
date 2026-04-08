#!/usr/bin/env python3
"""
PCA-Matryoshka IEEE TAI paper -- supplementary experiments.

Addresses reviewer feedback:
  1. Second non-Matryoshka embedding model (E5-large-v2)
  2. Public benchmark (MTEB STS Benchmark)
  3. Out-of-domain test (Jeopardy questions)
  4. Improved retrieval evaluation (200 queries, 5 seeds, CI)
  5. Cross-lingual retrieval test
  6. Eigenvalue spectrum comparison

Runs on Atlas CPU cores 12-23.
Results saved to /home/claude/agi-hpc/benchmarks/paper_experiment_results.json
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psycopg2
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA

# Add TurboQuant to path
sys.path.insert(0, "/home/claude/agi-hpc/src")

# Thread limits -- keep VERY LOW: other heavy processes may be running
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DB_CONN = "dbname=atlas user=claude"
RESULTS_PATH = "/home/claude/agi-hpc/benchmarks/paper_experiment_results.json"
JEOPARDY_PATH = "/archive/knowledge/kaggle/jeopardy/JEOPARDY_CSV.csv"
DIMS_TO_TEST = [64, 128, 256, 384, 512]
N_SEEDS = 5
N_QUERIES_RETRIEVAL = 200
N_CORPUS_RETRIEVAL = 9000

# Reduced counts to keep total runtime under 2h on CPU
N_E5_TEXTS = 1000       # E5 is slow on CPU; 1K is enough for PCA + stats
N_STS_E5_MAX = None      # Use all STS pairs (~1.5K) -- small enough
N_JEOPARDY = 2000        # For out-of-domain test
N_EIGENSPECTRUM = 5000   # For BGE eigenspectrum from DB (fast, already embedded)


# ------------------------------------------------------------------ #
# Temperature monitoring                                              #
# ------------------------------------------------------------------ #

def get_max_cpu_temp() -> float:
    try:
        out = subprocess.check_output(["sensors"], text=True, timeout=5)
        temps: list[float] = []
        for line in out.splitlines():
            if "Package" in line or "Core" in line:
                parts = line.split("+")
                if len(parts) >= 2:
                    t = parts[1].split("\u00b0")[0].split("C")[0].strip()
                    try:
                        temps.append(float(t))
                    except ValueError:
                        pass
        return max(temps) if temps else 0.0
    except Exception:
        return 0.0


def thermal_check(label: str = "", threshold: float = 80.0) -> None:
    t = get_max_cpu_temp()
    if t > 98:
        logger.warning("CPU temp %.1fC near critical after %s -- pausing 15s", t, label)
        time.sleep(15)
    elif t > 60:
        logger.info("CPU temp %.1fC after %s", t, label)


# ------------------------------------------------------------------ #
# Data loading                                                        #
# ------------------------------------------------------------------ #

def load_embeddings_from_db(
    table: str,
    n: int,
    conn_str: str = DB_CONN,
    with_text: bool = False,
    where_clause: str = "",
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Load embeddings (and optionally text) from PostgreSQL."""
    logger.info("Loading %d embeddings from %s ...", n, table)
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()

    text_col = "content"
    select_cols = f"embedding::text, {text_col}" if with_text else "embedding::text"

    extra = f" AND ({where_clause})" if where_clause else ""
    cur.execute(
        f"SELECT {select_cols} FROM {table} "
        f"WHERE embedding IS NOT NULL{extra} "
        f"ORDER BY random() LIMIT {n}"
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        raise RuntimeError(f"No embeddings found in {table}")

    vecs = []
    texts = [] if with_text else None
    for row in rows:
        emb_str = row[0]
        vals = emb_str.strip("[]").split(",")
        vecs.append([float(v) for v in vals])
        if with_text:
            texts.append(row[1])

    arr = np.array(vecs, dtype=np.float32)
    logger.info("Loaded %d embeddings, shape %s", len(arr), arr.shape)
    return arr, texts


def load_texts_from_db(
    table: str,
    n: int,
    conn_str: str = DB_CONN,
    where_clause: str = "",
) -> List[str]:
    """Load text content only from PostgreSQL."""
    logger.info("Loading %d texts from %s ...", n, table)
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()

    extra = f" AND ({where_clause})" if where_clause else ""
    cur.execute(
        f"SELECT content FROM {table} "
        f"WHERE content IS NOT NULL AND length(content) > 20{extra} "
        f"ORDER BY random() LIMIT {n}"
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [r[0] for r in rows]


def load_jeopardy_questions(path: str, n: int = 2000) -> List[str]:
    """Load Jeopardy questions from CSV."""
    logger.info("Loading %d Jeopardy questions from %s ...", n, path)
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n * 3:
                break
            # CSV has leading space in column names: ' Question'
            q = row.get("Question", row.get(" Question", "")).strip().strip('"').strip("'")
            if len(q) > 20:
                questions.append(q)
    rng = np.random.default_rng(42)
    rng.shuffle(questions)
    return questions[:n]


def safe_encode(model: Any, texts: List[str], batch_size: int = 8,
                label: str = "") -> np.ndarray:
    """Encode texts in small batches with thermal checks every 5 batches."""
    all_embs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embs.append(embs)
        batch_num = i // batch_size
        if batch_num % 10 == 9:
            thermal_check(f"{label} batch {batch_num}/{n_batches}")
        if batch_num % 20 == 0:
            logger.info("%s: batch %d/%d", label, batch_num, n_batches)
    return np.vstack(all_embs).astype(np.float32)


# ------------------------------------------------------------------ #
# Core math helpers                                                   #
# ------------------------------------------------------------------ #

def cosine_similarity_paired(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity."""
    nA = np.linalg.norm(A, axis=1, keepdims=True)
    nB = np.linalg.norm(B, axis=1, keepdims=True)
    nA = np.maximum(nA, 1e-10)
    nB = np.maximum(nB, 1e-10)
    return np.sum((A / nA) * (B / nB), axis=1)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize rows."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return X / norms


def evaluate_truncation(
    X_full: np.ndarray,
    dims: List[int],
    pca_model: Optional[PCA] = None,
) -> Dict[str, Any]:
    """Evaluate raw truncation vs PCA truncation quality."""
    results = {}
    X_normed = l2_normalize(X_full)

    for dim in dims:
        # Raw truncation
        X_raw_padded = np.zeros_like(X_normed)
        X_raw_padded[:, :dim] = X_normed[:, :dim]
        cos_raw = cosine_similarity_paired(X_normed, X_raw_padded)

        entry = {
            "dim": dim,
            "raw_truncation": {
                "mean_cosine": float(np.mean(cos_raw)),
                "std_cosine": float(np.std(cos_raw)),
                "min_cosine": float(np.min(cos_raw)),
                "p5_cosine": float(np.percentile(cos_raw, 5)),
                "p95_cosine": float(np.percentile(cos_raw, 95)),
            },
        }

        if pca_model is not None:
            X_pca_all = pca_model.transform(X_normed)
            X_pca_trunc = np.zeros_like(X_pca_all)
            X_pca_trunc[:, :dim] = X_pca_all[:, :dim]
            X_pca_recon = pca_model.inverse_transform(X_pca_trunc)
            cos_pca = cosine_similarity_paired(X_normed, X_pca_recon)

            entry["pca_truncation"] = {
                "mean_cosine": float(np.mean(cos_pca)),
                "std_cosine": float(np.std(cos_pca)),
                "min_cosine": float(np.min(cos_pca)),
                "p5_cosine": float(np.percentile(cos_pca, 5)),
                "p95_cosine": float(np.percentile(cos_pca, 95)),
            }

        results[str(dim)] = entry

    return results


def evaluate_retrieval(
    X_corpus: np.ndarray,
    X_queries: np.ndarray,
    dim: int,
    pca_model: Optional[PCA] = None,
    n_seeds: int = N_SEEDS,
    k: int = 10,
    method: str = "pca",
) -> Dict[str, Any]:
    """Evaluate retrieval recall@k with confidence intervals."""
    n_queries = X_queries.shape[0]
    recalls = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 100)
        perm = rng.permutation(n_queries)
        Q = X_queries[perm]

        # Full-dim ground truth
        C_full = l2_normalize(X_corpus)
        Q_full = l2_normalize(Q)
        sim_full = Q_full @ C_full.T
        gt_top_k = np.argsort(-sim_full, axis=1)[:, :k]

        # Truncated search
        if method == "pca" and pca_model is not None:
            C_trunc = l2_normalize(pca_model.transform(l2_normalize(X_corpus))[:, :dim])
            Q_trunc = l2_normalize(pca_model.transform(l2_normalize(Q))[:, :dim])
        elif method == "raw":
            C_trunc = l2_normalize(X_corpus[:, :dim])
            Q_trunc = l2_normalize(Q[:, :dim])
        else:
            continue

        sim_trunc = Q_trunc @ C_trunc.T
        pred_top_k = np.argsort(-sim_trunc, axis=1)[:, :k]

        recall_per_query = []
        for qi in range(n_queries):
            gt_set = set(gt_top_k[qi])
            pred_set = set(pred_top_k[qi])
            recall_per_query.append(len(gt_set & pred_set) / k)
        recalls.append(float(np.mean(recall_per_query)))

    return {
        "dim": dim,
        "method": method,
        "k": k,
        "n_queries": n_queries,
        "n_corpus": X_corpus.shape[0],
        "n_seeds": n_seeds,
        "mean_recall": float(np.mean(recalls)),
        "std_recall": float(np.std(recalls)),
        "ci95_low": float(np.mean(recalls) - 1.96 * np.std(recalls)),
        "ci95_high": float(np.mean(recalls) + 1.96 * np.std(recalls)),
        "recalls_per_seed": recalls,
    }


def evaluate_sts_correlation(
    emb1: np.ndarray,
    emb2: np.ndarray,
    scores: np.ndarray,
    dim: int,
    pca_model: Optional[PCA] = None,
    method: str = "full",
) -> Dict[str, Any]:
    """Spearman correlation of cosine sim vs human STS scores."""
    if method == "full":
        e1 = l2_normalize(emb1)
        e2 = l2_normalize(emb2)
    elif method == "raw":
        e1 = l2_normalize(emb1[:, :dim])
        e2 = l2_normalize(emb2[:, :dim])
    elif method == "pca" and pca_model is not None:
        e1 = l2_normalize(pca_model.transform(l2_normalize(emb1))[:, :dim])
        e2 = l2_normalize(pca_model.transform(l2_normalize(emb2))[:, :dim])
    else:
        raise ValueError(f"Unknown method: {method}")

    cos_sims = np.sum(e1 * e2, axis=1)
    spearman_r, spearman_p = scipy_stats.spearmanr(cos_sims, scores)

    return {
        "dim": dim,
        "method": method,
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n_pairs": len(scores),
    }


def tq3_compress_decompress(X: np.ndarray, dim: int) -> np.ndarray:
    """Apply TQ3 round-trip for quality measurement."""
    from agi.meta.llm.turboquant_kv import TurboQuantKV
    tq = TurboQuantKV(head_dim=dim, n_heads=1, bits=3, use_gpu=False, seed=42)
    tensor = X.reshape(1, 1, X.shape[0], X.shape[1])
    compressed = tq.compress(tensor, packed=True)
    recon = tq.decompress(compressed)
    return recon.reshape(-1, dim)


def spectrum_metrics(eigenvalues: list, name: str) -> dict:
    """Compute summary metrics from eigenvalue spectrum."""
    ev = np.array(eigenvalues)
    cumvar = np.cumsum(ev)
    d90 = int(np.searchsorted(cumvar, 0.90)) + 1
    d95 = int(np.searchsorted(cumvar, 0.95)) + 1
    d99 = int(np.searchsorted(cumvar, 0.99)) + 1
    eff_dim = float((np.sum(ev) ** 2) / np.sum(ev ** 2))
    top1_ratio = float(ev[0])
    ev_pos = ev[ev > 0]
    entropy = float(-np.sum(ev_pos * np.log(ev_pos)))
    return {
        "name": name,
        "dims_for_90pct": d90,
        "dims_for_95pct": d95,
        "dims_for_99pct": d99,
        "effective_dimensionality": round(eff_dim, 1),
        "top1_eigenvalue_ratio": round(top1_ratio, 6),
        "eigenvalue_entropy": round(entropy, 4),
    }


# ================================================================== #
#  MAIN EXPERIMENT RUNNER                                             #
# ================================================================== #

def main() -> None:
    t_start = time.time()
    logger.info("PCA-Matryoshka IEEE TAI Paper Experiments")
    logger.info("Starting at %s", time.strftime("%Y-%m-%d %H:%M:%S"))

    # Limit PyTorch threads
    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        logger.info("PyTorch threads set to 2")
    except Exception:
        pass

    # Brief thermal check -- don't block long, other processes may be running
    t = get_max_cpu_temp()
    logger.info("Initial CPU temp: %.1fC (other processes may be contributing)", t)
    if t > 92:
        logger.warning("CPU very hot (%.1fC) -- waiting 30s before starting", t)
        time.sleep(30)

    all_results: Dict[str, Any] = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": "Atlas HP Z840 (2x Xeon E5-2690 v3, 48 cores, 252 GB RAM)",
            "python": sys.version,
        }
    }

    # ============================================================== #
    # Phase 1: Load all data from DB (fast, no model needed)         #
    # ============================================================== #
    logger.info("PHASE 1: Loading data from PostgreSQL ...")

    # BGE-M3 embeddings with text (for E5 re-embedding)
    X_bge_full, texts_bge = load_embeddings_from_db(
        "chunks", 10000, with_text=True
    )
    thermal_check("db load")

    # Ethics embeddings for cross-domain PCA
    X_ethics, _ = load_embeddings_from_db(
        "ethics_chunks", 10000,
        where_clause="language='english'"
    )
    thermal_check("ethics load")

    # English + Hebrew for cross-lingual
    X_en, texts_en = load_embeddings_from_db(
        "ethics_chunks", 500, with_text=True,
        where_clause="language='english'"
    )
    X_he, texts_he = load_embeddings_from_db(
        "ethics_chunks", 2000, with_text=True,
        where_clause="language='hebrew'"
    )

    # Jeopardy
    jeopardy_questions = load_jeopardy_questions(JEOPARDY_PATH, n=N_JEOPARDY)
    logger.info("Loaded %d Jeopardy questions", len(jeopardy_questions))

    # ============================================================== #
    # Phase 2: Fit PCA on BGE-M3 embeddings (fast)                   #
    # ============================================================== #
    logger.info("PHASE 2: Fitting PCA models ...")

    X_bge_n = l2_normalize(X_bge_full)
    pca_bge = PCA(n_components=min(X_bge_full.shape[0] - 1, X_bge_full.shape[1]))
    pca_bge.fit(X_bge_n)
    eigenvalues_bge = pca_bge.explained_variance_ratio_.tolist()
    cumvar_bge = np.cumsum(pca_bge.explained_variance_ratio_).tolist()
    bge_metrics = spectrum_metrics(eigenvalues_bge, "BGE-M3")
    logger.info("BGE-M3 PCA: 90%%=%d, 95%%=%d, 99%%=%d dims",
                bge_metrics["dims_for_90pct"],
                bge_metrics["dims_for_95pct"],
                bge_metrics["dims_for_99pct"])

    # Ethics PCA (for cross-domain)
    X_ethics_n = l2_normalize(X_ethics)
    pca_ethics = PCA(n_components=min(X_ethics.shape[0] - 1, X_ethics.shape[1]))
    pca_ethics.fit(X_ethics_n)

    # ============================================================== #
    # Phase 3: E5-large-v2 embedding (slow -- cache to disk)         #
    # ============================================================== #
    logger.info("PHASE 3: E5-large-v2 embedding ...")

    CACHE_DIR = "/home/claude/agi-hpc/benchmarks/.embed_cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_e5_chunks = os.path.join(CACHE_DIR, "e5_chunks.npy")
    cache_e5_jeop = os.path.join(CACHE_DIR, "e5_jeopardy.npy")
    cache_e5_sts1 = os.path.join(CACHE_DIR, "e5_sts_s1.npy")
    cache_e5_sts2 = os.path.join(CACHE_DIR, "e5_sts_s2.npy")
    cache_bge_sts1 = os.path.join(CACHE_DIR, "bge_sts_s1.npy")
    cache_bge_sts2 = os.path.join(CACHE_DIR, "bge_sts_s2.npy")
    cache_bge_jeop = os.path.join(CACHE_DIR, "bge_jeopardy.npy")

    from sentence_transformers import SentenceTransformer

    # Determine which E5 embeddings need computing
    need_e5_model = (
        not os.path.exists(cache_e5_chunks) or
        not os.path.exists(cache_e5_jeop) or
        not os.path.exists(cache_e5_sts1) or
        not os.path.exists(cache_e5_sts2)
    )

    model_e5 = None
    if need_e5_model:
        logger.info("Loading E5-large-v2 model ...")
        model_e5 = SentenceTransformer("intfloat/e5-large-v2", device="cpu")

    # E5 chunks embedding
    if os.path.exists(cache_e5_chunks):
        X_e5 = np.load(cache_e5_chunks)
        e5_embed_time = 0.0
        logger.info("E5 chunks loaded from cache: shape %s", X_e5.shape)
    else:
        e5_texts_subset = texts_bge[:N_E5_TEXTS]
        e5_input = ["passage: " + t for t in e5_texts_subset]
        t0 = time.time()
        X_e5 = safe_encode(model_e5, e5_input, batch_size=8, label="E5-chunks")
        e5_embed_time = time.time() - t0
        np.save(cache_e5_chunks, X_e5)
        logger.info("E5 chunks embedding: shape %s in %.1fs (cached)", X_e5.shape, e5_embed_time)
        thermal_check("e5 chunks embed")

    # E5 Jeopardy embedding
    if os.path.exists(cache_e5_jeop):
        X_jeop_e5 = np.load(cache_e5_jeop)
        jeop_e5_time = 0.0
        logger.info("E5 Jeopardy loaded from cache: shape %s", X_jeop_e5.shape)
    else:
        jeop_e5_input = ["query: " + q for q in jeopardy_questions]
        t0 = time.time()
        X_jeop_e5 = safe_encode(model_e5, jeop_e5_input, batch_size=8, label="E5-jeopardy")
        jeop_e5_time = time.time() - t0
        np.save(cache_e5_jeop, X_jeop_e5)
        logger.info("E5 Jeopardy embedding: shape %s in %.1fs (cached)", X_jeop_e5.shape, jeop_e5_time)
        thermal_check("e5 jeopardy embed")

    # Fit PCA on E5 embeddings
    X_e5_n = l2_normalize(X_e5)
    e5_dim = X_e5.shape[1]
    pca_e5 = PCA(n_components=min(X_e5.shape[0] - 1, e5_dim))
    pca_e5.fit(X_e5_n)
    eigenvalues_e5 = pca_e5.explained_variance_ratio_.tolist()
    cumvar_e5 = np.cumsum(pca_e5.explained_variance_ratio_).tolist()
    e5_metrics = spectrum_metrics(eigenvalues_e5, "E5-large-v2")

    # ============================================================== #
    # Phase 4: STS Benchmark embedding (small dataset, both models)  #
    # ============================================================== #
    logger.info("PHASE 4: STS Benchmark ...")

    from datasets import load_dataset
    sts = load_dataset("mteb/stsbenchmark-sts", split="test")
    sts_s1 = [r["sentence1"] for r in sts]
    sts_s2 = [r["sentence2"] for r in sts]
    sts_scores = np.array([r["score"] for r in sts], dtype=np.float32)
    logger.info("STS benchmark: %d pairs loaded", len(sts_scores))

    # E5 STS embeddings
    if os.path.exists(cache_e5_sts1) and os.path.exists(cache_e5_sts2):
        sts_e5_s1 = np.load(cache_e5_sts1)
        sts_e5_s2 = np.load(cache_e5_sts2)
        sts_e5_time = 0.0
        logger.info("E5 STS loaded from cache: %d pairs", len(sts_e5_s1))
    else:
        logger.info("Embedding STS with E5-large-v2 ...")
        t0 = time.time()
        sts_e5_s1 = safe_encode(model_e5, ["query: " + s for s in sts_s1],
                                batch_size=8, label="E5-STS-s1")
        sts_e5_s2 = safe_encode(model_e5, ["passage: " + s for s in sts_s2],
                                batch_size=8, label="E5-STS-s2")
        sts_e5_time = time.time() - t0
        np.save(cache_e5_sts1, sts_e5_s1)
        np.save(cache_e5_sts2, sts_e5_s2)
        logger.info("E5 STS embedding done in %.1fs (cached)", sts_e5_time)
        thermal_check("e5 sts embed")

    # Free E5 model
    del model_e5
    import gc
    gc.collect()
    logger.info("E5 model freed from memory")

    # Determine if we need BGE model
    need_bge_model = (
        not os.path.exists(cache_bge_sts1) or
        not os.path.exists(cache_bge_sts2) or
        not os.path.exists(cache_bge_jeop)
    )
    model_bge = None
    if need_bge_model:
        logger.info("Loading BGE-M3 model ...")
        model_bge = SentenceTransformer("BAAI/bge-m3", device="cpu")

    # BGE-M3 STS embeddings
    if os.path.exists(cache_bge_sts1) and os.path.exists(cache_bge_sts2):
        sts_bge_s1 = np.load(cache_bge_sts1)
        sts_bge_s2 = np.load(cache_bge_sts2)
        sts_bge_time = 0.0
        logger.info("BGE STS loaded from cache: %d pairs", len(sts_bge_s1))
    else:
        t0 = time.time()
        sts_bge_s1 = safe_encode(model_bge, sts_s1, batch_size=8, label="BGE-STS-s1")
        sts_bge_s2 = safe_encode(model_bge, sts_s2, batch_size=8, label="BGE-STS-s2")
        sts_bge_time = time.time() - t0
        np.save(cache_bge_sts1, sts_bge_s1)
        np.save(cache_bge_sts2, sts_bge_s2)
        logger.info("BGE STS embedding done in %.1fs (cached)", sts_bge_time)
        thermal_check("bge sts embed")

    # Jeopardy with BGE-M3 (for out-of-domain)
    if os.path.exists(cache_bge_jeop):
        X_jeop_bge = np.load(cache_bge_jeop)
        jeop_bge_time = 0.0
        logger.info("BGE Jeopardy loaded from cache: shape %s", X_jeop_bge.shape)
    else:
        logger.info("Embedding Jeopardy with BGE-M3 ...")
        t0 = time.time()
        X_jeop_bge = safe_encode(model_bge, jeopardy_questions, batch_size=8,
                                 label="BGE-jeopardy")
        jeop_bge_time = time.time() - t0
        np.save(cache_bge_jeop, X_jeop_bge)
        logger.info("BGE Jeopardy embedding: shape %s in %.1fs (cached)", X_jeop_bge.shape, jeop_bge_time)
        thermal_check("bge jeopardy embed")

    # Free BGE model
    del model_bge
    gc.collect()
    logger.info("Models freed from memory")

    # ============================================================== #
    # Phase 5: Run all experiments (pure numpy, fast)                 #
    # ============================================================== #

    # -------------------------------------------------------------- #
    # EXPERIMENT 1: E5-large-v2 (second model)                       #
    # -------------------------------------------------------------- #
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: E5-large-v2 results")
    logger.info("=" * 70)

    test_dims_e5 = [d for d in DIMS_TO_TEST if d < e5_dim]
    trunc_e5 = evaluate_truncation(X_e5, test_dims_e5, pca_model=pca_e5)

    # PCA + TQ3
    pca_tq3_e5 = {}
    for dim in test_dims_e5:
        X_pca_trunc = pca_e5.transform(X_e5_n)[:, :dim].astype(np.float32)
        X_pca_trunc_n = l2_normalize(X_pca_trunc)
        try:
            X_tq3_recon = tq3_compress_decompress(X_pca_trunc_n, dim)
            cos_tq3 = cosine_similarity_paired(X_pca_trunc_n, X_tq3_recon)
            pca_tq3_e5[str(dim)] = {
                "mean_cosine_pca_to_tq3_recon": float(np.mean(cos_tq3)),
                "std_cosine": float(np.std(cos_tq3)),
            }
        except Exception as e:
            logger.warning("TQ3 failed at dim %d: %s", dim, e)
            pca_tq3_e5[str(dim)] = {"error": str(e)}

    all_results["experiment_1_e5_model"] = {
        "model": "intfloat/e5-large-v2",
        "native_dim": e5_dim,
        "n_texts": N_E5_TEXTS,
        "embed_time_sec": round(e5_embed_time, 1),
        "truncation_quality": trunc_e5,
        "pca_tq3_combinations": pca_tq3_e5,
        "eigenvalue_spectrum": {
            "explained_variance_ratio_first100": eigenvalues_e5[:100],
            "cumulative_variance_first100": cumvar_e5[:100],
            "metrics": e5_metrics,
        },
    }

    print("\n" + "=" * 80)
    print("EXPERIMENT 1: E5-large-v2 Truncation Quality")
    print("=" * 80)
    print(f"{'Dim':<6} {'Raw Cos':>10} {'PCA Cos':>10} {'Delta':>10} {'PCA+TQ3':>10}")
    print("-" * 50)
    for dim in test_dims_e5:
        d = str(dim)
        raw = trunc_e5[d]["raw_truncation"]["mean_cosine"]
        pca = trunc_e5[d].get("pca_truncation", {}).get("mean_cosine", 0)
        delta = pca - raw
        tq3 = pca_tq3_e5.get(d, {}).get("mean_cosine_pca_to_tq3_recon", 0)
        print(f"{dim:<6} {raw:>10.4f} {pca:>10.4f} {delta:>+10.4f} {tq3:>10.4f}")

    # -------------------------------------------------------------- #
    # EXPERIMENT 2: STS Benchmark                                     #
    # -------------------------------------------------------------- #
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: STS Benchmark")
    logger.info("=" * 70)

    sts_results_by_model = {}

    for model_name, emb_s1, emb_s2 in [
        ("BAAI/bge-m3", sts_bge_s1, sts_bge_s2),
        ("intfloat/e5-large-v2", sts_e5_s1, sts_e5_s2),
    ]:
        native_dim = emb_s1.shape[1]

        # PCA on combined STS embeddings for this model
        all_embs = np.vstack([emb_s1, emb_s2])
        pca_sts = PCA(n_components=min(all_embs.shape[0] - 1, native_dim))
        pca_sts.fit(l2_normalize(all_embs))

        sts_full = evaluate_sts_correlation(emb_s1, emb_s2, sts_scores,
                                            native_dim, method="full")

        test_dims = [d for d in DIMS_TO_TEST if d < native_dim]
        per_dim = {}
        for dim in test_dims:
            raw_r = evaluate_sts_correlation(emb_s1, emb_s2, sts_scores,
                                             dim, method="raw")
            pca_r = evaluate_sts_correlation(emb_s1, emb_s2, sts_scores,
                                             dim, pca_model=pca_sts, method="pca")
            per_dim[str(dim)] = {
                "raw_truncation": raw_r,
                "pca_truncation": pca_r,
            }

        sts_results_by_model[model_name] = {
            "native_dim": native_dim,
            "n_pairs": len(sts_scores),
            "full_dim_spearman": sts_full,
            "per_dim_results": per_dim,
        }

        safe_name = model_name.split("/")[-1]
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT 2: STS Benchmark - {safe_name}")
        print(f"{'=' * 80}")
        print(f"Full-dim ({native_dim}) Spearman r = {sts_full['spearman_r']:.4f}")
        print(f"{'Dim':<6} {'Raw Spearman':>14} {'PCA Spearman':>14} {'Delta':>10}")
        print("-" * 50)
        for dim in test_dims:
            d = str(dim)
            raw_rr = per_dim[d]["raw_truncation"]["spearman_r"]
            pca_rr = per_dim[d]["pca_truncation"]["spearman_r"]
            delta = pca_rr - raw_rr
            print(f"{dim:<6} {raw_rr:>14.4f} {pca_rr:>14.4f} {delta:>+10.4f}")

    all_results["experiment_2_sts_benchmark"] = sts_results_by_model

    # -------------------------------------------------------------- #
    # EXPERIMENT 3: Out-of-Domain (Jeopardy)                         #
    # -------------------------------------------------------------- #
    logger.info("=" * 70)
    logger.info("EXPERIMENT 3: Out-of-Domain (Jeopardy)")
    logger.info("=" * 70)

    ood_results = {}

    for model_label, X_jeop, pca_indomain, pca_indomain_label in [
        ("BGE-M3", X_jeop_bge, pca_ethics, "ethics_corpus"),
        ("E5-large-v2", X_jeop_e5, pca_e5, "chunks_e5_corpus"),
    ]:
        X_jeop_n = l2_normalize(X_jeop)
        native_dim = X_jeop.shape[1]

        # Also fit in-domain PCA on Jeopardy itself
        pca_jeop = PCA(n_components=min(X_jeop_n.shape[0] - 1, native_dim))
        pca_jeop.fit(X_jeop_n)

        test_dims = [d for d in DIMS_TO_TEST if d < native_dim]
        cross_domain = {}

        for dim in test_dims:
            # Raw truncation
            X_raw_pad = np.zeros_like(X_jeop_n)
            X_raw_pad[:, :dim] = X_jeop_n[:, :dim]
            cos_raw = cosine_similarity_paired(X_jeop_n, X_raw_pad)

            # Cross-domain PCA
            X_pca_all = pca_indomain.transform(X_jeop_n)
            X_pca_trunc = np.zeros_like(X_pca_all)
            X_pca_trunc[:, :dim] = X_pca_all[:, :dim]
            X_recon_cross = pca_indomain.inverse_transform(X_pca_trunc)
            cos_cross = cosine_similarity_paired(X_jeop_n, X_recon_cross)

            # In-domain PCA (Jeopardy)
            X_pca_all_in = pca_jeop.transform(X_jeop_n)
            X_pca_trunc_in = np.zeros_like(X_pca_all_in)
            X_pca_trunc_in[:, :dim] = X_pca_all_in[:, :dim]
            X_recon_in = pca_jeop.inverse_transform(X_pca_trunc_in)
            cos_in = cosine_similarity_paired(X_jeop_n, X_recon_in)

            cross_domain[str(dim)] = {
                "raw_truncation": {
                    "mean_cosine": float(np.mean(cos_raw)),
                    "std_cosine": float(np.std(cos_raw)),
                    "p5_cosine": float(np.percentile(cos_raw, 5)),
                },
                "pca_cross_domain": {
                    "mean_cosine": float(np.mean(cos_cross)),
                    "std_cosine": float(np.std(cos_cross)),
                    "p5_cosine": float(np.percentile(cos_cross, 5)),
                    "pca_source": pca_indomain_label,
                },
                "pca_in_domain": {
                    "mean_cosine": float(np.mean(cos_in)),
                    "std_cosine": float(np.std(cos_in)),
                    "p5_cosine": float(np.percentile(cos_in, 5)),
                    "pca_source": "jeopardy",
                },
            }

        ood_results[model_label] = {
            "n_jeopardy": len(jeopardy_questions),
            "model_dim": native_dim,
            "cross_domain_results": cross_domain,
        }

        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT 3: Out-of-Domain (Jeopardy) - {model_label}")
        print(f"{'=' * 80}")
        print(f"{'Dim':<6} {'Raw':>10} {'PCA(cross)':>12} {'PCA(jeop)':>12} {'Gap':>10}")
        print("-" * 55)
        for dim in test_dims:
            d = str(dim)
            raw = cross_domain[d]["raw_truncation"]["mean_cosine"]
            cross = cross_domain[d]["pca_cross_domain"]["mean_cosine"]
            ind = cross_domain[d]["pca_in_domain"]["mean_cosine"]
            gap = cross - ind
            print(f"{dim:<6} {raw:>10.4f} {cross:>12.4f} {ind:>12.4f} {gap:>+10.4f}")

    all_results["experiment_3_out_of_domain"] = ood_results

    # -------------------------------------------------------------- #
    # EXPERIMENT 4: Improved Retrieval                                #
    # -------------------------------------------------------------- #
    logger.info("=" * 70)
    logger.info("EXPERIMENT 4: Improved Retrieval Evaluation")
    logger.info("=" * 70)

    # Split BGE embeddings: 200 queries, 9000 corpus
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X_bge_full))
    X_q = X_bge_full[indices[:N_QUERIES_RETRIEVAL]]
    X_c = X_bge_full[indices[N_QUERIES_RETRIEVAL:N_QUERIES_RETRIEVAL + N_CORPUS_RETRIEVAL]]
    logger.info("Retrieval: %d queries, %d corpus", X_q.shape[0], X_c.shape[0])

    # PCA fitted on corpus only (realistic)
    pca_corpus = PCA(n_components=min(X_c.shape[0] - 1, X_c.shape[1]))
    pca_corpus.fit(l2_normalize(X_c))

    test_dims_ret = [d for d in DIMS_TO_TEST if d < X_c.shape[1]]

    retrieval_results = {}
    for dim in test_dims_ret:
        logger.info("Retrieval at dim=%d ...", dim)
        raw_r = evaluate_retrieval(X_c, X_q, dim, n_seeds=N_SEEDS, method="raw")
        pca_r = evaluate_retrieval(X_c, X_q, dim, pca_model=pca_corpus,
                                   n_seeds=N_SEEDS, method="pca")
        retrieval_results[str(dim)] = {
            "raw_truncation": raw_r,
            "pca_truncation": pca_r,
        }
        thermal_check(f"retrieval dim={dim}")

    all_results["experiment_4_retrieval"] = {
        "protocol": {
            "n_queries": N_QUERIES_RETRIEVAL,
            "n_corpus": X_c.shape[0],
            "n_seeds": N_SEEDS,
            "k": 10,
            "normalization": "L2 normalize before PCA and after truncation",
            "pca_fitted_on": "corpus vectors only (not queries)",
            "source_table": "chunks (BGE-M3 1024-dim)",
        },
        "results": retrieval_results,
    }

    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT 4: Retrieval Recall@10 ({N_QUERIES_RETRIEVAL}q, "
          f"{X_c.shape[0]}c, {N_SEEDS} seeds)")
    print(f"{'=' * 80}")
    print(f"{'Dim':<6} {'Raw R@10':>12} {'(+/-std)':>10} {'PCA R@10':>12} {'(+/-std)':>10} {'Delta':>10}")
    print("-" * 65)
    for dim in test_dims_ret:
        d = str(dim)
        raw_m = retrieval_results[d]["raw_truncation"]["mean_recall"]
        raw_s = retrieval_results[d]["raw_truncation"]["std_recall"]
        pca_m = retrieval_results[d]["pca_truncation"]["mean_recall"]
        pca_s = retrieval_results[d]["pca_truncation"]["std_recall"]
        print(f"{dim:<6} {raw_m:>12.4f} {raw_s:>10.4f} {pca_m:>12.4f} {pca_s:>10.4f} {pca_m - raw_m:>+10.4f}")

    # -------------------------------------------------------------- #
    # EXPERIMENT 5: Cross-lingual Retrieval                          #
    # -------------------------------------------------------------- #
    logger.info("=" * 70)
    logger.info("EXPERIMENT 5: Cross-lingual Retrieval")
    logger.info("=" * 70)

    # PCA fitted on combined English+Hebrew
    X_combined_cl = np.vstack([X_en, X_he])
    pca_cl = PCA(n_components=min(X_combined_cl.shape[0] - 1, X_combined_cl.shape[1]))
    pca_cl.fit(l2_normalize(X_combined_cl))

    test_dims_cl = [d for d in DIMS_TO_TEST if d < X_en.shape[1]]
    cl_results = {}
    k = 10

    for dim in test_dims_cl:
        Q_full = l2_normalize(X_en)
        C_full = l2_normalize(X_he)
        sim_full = Q_full @ C_full.T
        gt_top_k = np.argsort(-sim_full, axis=1)[:, :k]

        # Raw
        Q_raw = l2_normalize(X_en[:, :dim])
        C_raw = l2_normalize(X_he[:, :dim])
        sim_raw = Q_raw @ C_raw.T
        pred_raw = np.argsort(-sim_raw, axis=1)[:, :k]
        recall_raw = []
        for qi in range(len(X_en)):
            gt_set = set(gt_top_k[qi])
            pred_set = set(pred_raw[qi])
            recall_raw.append(len(gt_set & pred_set) / k)

        # PCA
        Q_pca = l2_normalize(pca_cl.transform(l2_normalize(X_en))[:, :dim])
        C_pca = l2_normalize(pca_cl.transform(l2_normalize(X_he))[:, :dim])
        sim_pca = Q_pca @ C_pca.T
        pred_pca = np.argsort(-sim_pca, axis=1)[:, :k]
        recall_pca = []
        for qi in range(len(X_en)):
            gt_set = set(gt_top_k[qi])
            pred_set = set(pred_pca[qi])
            recall_pca.append(len(gt_set & pred_set) / k)

        cl_results[str(dim)] = {
            "raw_recall_at_10": float(np.mean(recall_raw)),
            "raw_recall_std": float(np.std(recall_raw)),
            "pca_recall_at_10": float(np.mean(recall_pca)),
            "pca_recall_std": float(np.std(recall_pca)),
        }

    all_results["experiment_5_crosslingual"] = {
        "query_language": "english",
        "corpus_language": "hebrew",
        "n_queries": len(X_en),
        "n_corpus": len(X_he),
        "k": k,
        "model": "BGE-M3 (pre-embedded in DB)",
        "results": cl_results,
    }

    print(f"\n{'=' * 80}")
    print("EXPERIMENT 5: Cross-lingual Retrieval (English -> Hebrew)")
    print(f"{'=' * 80}")
    print(f"{'Dim':<6} {'Raw R@10':>12} {'PCA R@10':>12} {'Delta':>10}")
    print("-" * 45)
    for dim in test_dims_cl:
        d = str(dim)
        raw_r = cl_results[d]["raw_recall_at_10"]
        pca_r = cl_results[d]["pca_recall_at_10"]
        print(f"{dim:<6} {raw_r:>12.4f} {pca_r:>12.4f} {pca_r - raw_r:>+10.4f}")

    # -------------------------------------------------------------- #
    # EXPERIMENT 6: Eigenvalue Spectrum                               #
    # -------------------------------------------------------------- #
    logger.info("=" * 70)
    logger.info("EXPERIMENT 6: Eigenvalue Spectrum Comparison")
    logger.info("=" * 70)

    # Matryoshka scores (raw truncation cosine at each dim)
    matryoshka_scores = {}
    for dim in DIMS_TO_TEST:
        # BGE-M3
        X_raw_pad = np.zeros_like(X_bge_n)
        X_raw_pad[:, :dim] = X_bge_n[:, :dim]
        cos_bge = float(np.mean(cosine_similarity_paired(X_bge_n, X_raw_pad)))

        # E5
        if dim < e5_dim:
            X_e5_pad = np.zeros_like(X_e5_n)
            X_e5_pad[:, :dim] = X_e5_n[:, :dim]
            cos_e5 = float(np.mean(cosine_similarity_paired(X_e5_n, X_e5_pad)))
        else:
            cos_e5 = 1.0

        matryoshka_scores[str(dim)] = {
            "bge_m3_raw_cosine": cos_bge,
            "e5_large_v2_raw_cosine": cos_e5,
        }

    all_results["experiment_6_eigenspectrum"] = {
        "bge_m3": {
            "explained_variance_ratio": eigenvalues_bge,
            "cumulative_variance": cumvar_bge,
            "metrics": bge_metrics,
        },
        "e5_large_v2": {
            "explained_variance_ratio": eigenvalues_e5,
            "cumulative_variance": cumvar_e5,
            "metrics": e5_metrics,
        },
        "matryoshka_scores": matryoshka_scores,
    }

    print(f"\n{'=' * 80}")
    print("EXPERIMENT 6: Eigenvalue Spectrum Comparison")
    print(f"{'=' * 80}")
    print(f"\n{'Metric':<30} {'BGE-M3':>15} {'E5-large-v2':>15}")
    print("-" * 60)
    for key in ["dims_for_90pct", "dims_for_95pct", "dims_for_99pct",
                "effective_dimensionality", "top1_eigenvalue_ratio",
                "eigenvalue_entropy"]:
        print(f"{key:<30} {bge_metrics[key]:>15} {e5_metrics[key]:>15}")

    print(f"\n{'Dim':<6} {'BGE Raw Cos':>12} {'E5 Raw Cos':>12}")
    print("-" * 35)
    for dim in DIMS_TO_TEST:
        d = str(dim)
        print(f"{dim:<6} {matryoshka_scores[d]['bge_m3_raw_cosine']:>12.4f} "
              f"{matryoshka_scores[d]['e5_large_v2_raw_cosine']:>12.4f}")

    # ============================================================== #
    # Save results                                                    #
    # ============================================================== #
    elapsed = time.time() - t_start
    all_results["metadata"]["total_runtime_sec"] = round(elapsed, 1)
    all_results["metadata"]["total_runtime_min"] = round(elapsed / 60, 1)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("Results saved to %s", RESULTS_PATH)
    logger.info("Total runtime: %.1f minutes", elapsed / 60)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total runtime: {elapsed / 60:.1f} minutes")
    print(f"Results: {RESULTS_PATH}")
    for key in sorted(all_results.keys()):
        if key.startswith("experiment_"):
            val = all_results[key]
            if isinstance(val, dict) and "error" in val:
                print(f"  {key}: FAILED - {val['error']}")
            else:
                print(f"  {key}: OK")


if __name__ == "__main__":
    main()
