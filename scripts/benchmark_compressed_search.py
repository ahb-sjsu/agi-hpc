#!/usr/bin/env python3
"""Benchmark compressed-space search: TQ3 packed search vs pgvector float.

Tests searching directly on TurboQuant 3-bit packed representations
without decompressing to float, using the centroid inner product trick.

Run on Atlas:
    taskset -c 24-35 python benchmark_compressed_search.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np
import psycopg2

sys.path.insert(0, "/home/claude/agi-hpc/src")
sys.path.insert(0, "/home/claude/turboquant-pro")

DB_DSN = "dbname=atlas user=claude"
PCA_PATH = "/home/claude/agi-hpc/data/pca_rotation_384.pkl"


def main():
    from turboquant_pro import TurboQuantPGVector

    # Load PCA
    with open(PCA_PATH, "rb") as f:
        pca = pickle.load(f)
    comp384 = pca["components"].T.astype(np.float32)
    mean = pca["mean"].astype(np.float32)

    # Initialize TQ3 for 384-dim
    tq = TurboQuantPGVector(dim=384, bits=3, seed=42)
    tq1024 = TurboQuantPGVector(dim=1024, bits=3, seed=42)

    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    # Load corpus
    N_CORPUS = 10000  # start smaller for speed
    print("Loading %d embeddings..." % N_CORPUS)
    cur.execute(
        "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL "
        "ORDER BY random() LIMIT %s",
        (N_CORPUS,),
    )
    rows = cur.fetchall()
    ids = [r[0] for r in rows]
    embs = np.array(
        [np.fromstring(r[1].strip("[]"), sep=",", dtype=np.float32) for r in rows]
    )
    print("Loaded %d vectors" % len(embs))

    # Precompute representations
    norms_full = np.linalg.norm(embs, axis=1)

    # PCA-384
    centered = embs - mean
    pca384 = centered @ comp384
    pca384_normed = pca384 / (np.linalg.norm(pca384, axis=1, keepdims=True) + 1e-10)

    # TQ3 compress full 1024-dim
    print("Compressing 1024-dim to TQ3...")
    t0 = time.time()
    compressed_1024 = tq1024.compress_batch(embs)
    print("  %d vectors in %.1fs (%.0f vec/s)" % (
        len(compressed_1024), time.time() - t0,
        len(compressed_1024) / (time.time() - t0),
    ))

    # TQ3 compress PCA-384
    print("Compressing PCA-384 to TQ3...")
    t0 = time.time()
    compressed_384 = tq.compress_batch(pca384_normed)
    print("  %d vectors in %.1fs (%.0f vec/s)" % (
        len(compressed_384), time.time() - t0,
        len(compressed_384) / (time.time() - t0),
    ))

    # Queries
    N_Q = 100
    rng = np.random.default_rng(42)
    q_idx = rng.choice(len(ids), N_Q, replace=False)

    results = {}

    # --- Method 1: Full 1024-dim exact brute force ---
    print("\n--- Full 1024-dim brute force ---")
    recalls_ref = []
    t0 = time.time()
    for qi in q_idx:
        q = embs[qi]
        sims = embs @ q / (norms_full * np.linalg.norm(q) + 1e-10)
        gt = set(np.argsort(sims)[-10:])
        recalls_ref.append(gt)
    ref_time = (time.time() - t0) / N_Q * 1000
    print("  Latency: %.2f ms" % ref_time)
    results["full_1024_bruteforce"] = {"recall": 1.0, "latency_ms": round(ref_time, 2)}

    # --- Method 2: PCA-384 float brute force ---
    print("\n--- PCA-384 float brute force ---")
    recalls = []
    t0 = time.time()
    for i, qi in enumerate(q_idx):
        q384 = pca384_normed[qi]
        sims = pca384_normed @ q384
        top10 = set(np.argsort(sims)[-10:])
        recalls.append(len(recalls_ref[i] & top10) / 10)
    elapsed = (time.time() - t0) / N_Q * 1000
    recalls = np.array(recalls)
    print("  Recall@10: %.4f  Latency: %.2f ms" % (recalls.mean(), elapsed))
    results["pca384_float"] = {"recall": float(recalls.mean()), "latency_ms": round(elapsed, 2)}

    # --- Method 3: TQ3 compressed search (1024-dim) ---
    print("\n--- TQ3 compressed cosine (1024-dim) ---")
    recalls = []
    t0 = time.time()
    for i, qi in enumerate(q_idx):
        scores = tq1024.compressed_cosine_similarity(embs[qi], compressed_1024)
        top10 = set(np.argsort(scores)[-10:])
        recalls.append(len(recalls_ref[i] & top10) / 10)
    elapsed = (time.time() - t0) / N_Q * 1000
    recalls = np.array(recalls)
    print("  Recall@10: %.4f  Latency: %.2f ms" % (recalls.mean(), elapsed))
    results["tq3_1024_compressed"] = {"recall": float(recalls.mean()), "latency_ms": round(elapsed, 2)}

    # --- Method 4: TQ3 compressed search (PCA-384) ---
    print("\n--- TQ3 compressed cosine (PCA-384) ---")
    recalls = []
    t0 = time.time()
    for i, qi in enumerate(q_idx):
        scores = tq.compressed_cosine_similarity(pca384_normed[qi], compressed_384)
        top10 = set(np.argsort(scores)[-10:])
        recalls.append(len(recalls_ref[i] & top10) / 10)
    elapsed = (time.time() - t0) / N_Q * 1000
    recalls = np.array(recalls)
    print("  Recall@10: %.4f  Latency: %.2f ms" % (recalls.mean(), elapsed))
    results["tq3_pca384_compressed"] = {"recall": float(recalls.mean()), "latency_ms": round(elapsed, 2)}

    # --- Storage comparison ---
    print("\n--- Storage per vector ---")
    sizes = {
        "full_1024_float32": 1024 * 4,
        "pca384_float32": 384 * 4,
        "tq3_1024_packed": compressed_1024[0].size_bytes,
        "tq3_pca384_packed": compressed_384[0].size_bytes,
    }
    for k, v in sizes.items():
        print("  %-25s %5d bytes  (%.1fx)" % (k, v, 4096 / v))
    results["storage_bytes_per_vec"] = sizes

    # Summary
    print("\n" + "=" * 65)
    print("  COMPRESSED SEARCH BENCHMARK (%d corpus, %d queries)" % (N_CORPUS, N_Q))
    print("=" * 65)
    print("%-35s  recall@10  latency  storage" % "Method")
    print("-" * 65)
    for k in ["full_1024_bruteforce", "pca384_float", "tq3_1024_compressed", "tq3_pca384_compressed"]:
        v = results[k]
        sz = sizes.get(k.replace("_bruteforce", "_float32").replace("_compressed", "_packed").replace("_float", "_float32"), 0)
        print("%-35s  %.4f     %.1f ms   %d B" % (k, v["recall"], v["latency_ms"], sz))

    with open("/tmp/compressed_search_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /tmp/compressed_search_benchmark.json")

    conn.close()


if __name__ == "__main__":
    main()
