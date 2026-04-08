#!/usr/bin/env python3
"""Benchmark funnel search: PCA-64 coarse -> PCA-384 rerank.

Tests whether a two-stage search (coarse low-dim filter + fine high-dim rerank)
is faster than single-stage PCA-384 search while maintaining recall.

Run on Atlas:
    taskset -c 24-35 python benchmark_funnel_search.py
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import psycopg2

DB_DSN = "dbname=atlas user=claude"
PCA_PATH = "/home/claude/agi-hpc/data/pca_rotation_384.pkl"
RESULTS_PATH = "/tmp/funnel_benchmark.json"


def main():
    with open(PCA_PATH, "rb") as f:
        pca = pickle.load(f)
    comp384 = pca["components"].T.astype(np.float32)  # (1024, 384)
    mean = pca["mean"].astype(np.float32)
    comp64 = comp384[:, :64]  # First 64 PCA dims
    comp128 = comp384[:, :128]

    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    # Use chunks table (fully migrated, 112K)
    TABLE = "chunks"
    cur.execute("SELECT count(*) FROM %s" % TABLE)
    total = cur.fetchone()[0]
    print("Table: %s (%d rows)" % (TABLE, total))

    # Load all embeddings into memory for numpy-based search
    print("Loading embeddings...")
    cur.execute(
        "SELECT id, embedding FROM %s WHERE embedding IS NOT NULL" % TABLE
    )
    rows = cur.fetchall()
    corpus_ids = np.array([r[0] for r in rows])
    corpus_embs = np.array(
        [np.fromstring(r[1].strip("[]"), sep=",", dtype=np.float32) for r in rows]
    )
    N = len(corpus_ids)
    print("Loaded %d vectors, shape %s" % (N, corpus_embs.shape))

    # Precompute all PCA projections
    print("Computing PCA projections...")
    centered = corpus_embs - mean
    pca384 = centered @ comp384
    pca384 /= np.linalg.norm(pca384, axis=1, keepdims=True) + 1e-10
    pca128 = centered @ comp128
    pca128 /= np.linalg.norm(pca128, axis=1, keepdims=True) + 1e-10
    pca64 = centered @ comp64
    pca64 /= np.linalg.norm(pca64, axis=1, keepdims=True) + 1e-10
    print("Done.")

    # Norms for full-dim cosine
    corpus_norms = np.linalg.norm(corpus_embs, axis=1)

    # Select 100 random queries
    N_Q = 100
    rng = np.random.default_rng(42)
    q_idx = rng.choice(N, N_Q, replace=False)

    results = {}

    # --- Method 1: Full 1024-dim exact search (ground truth) ---
    print("\n--- Ground truth: full 1024-dim exact search ---")

    # --- Method 2: PCA-384 brute force ---
    print("\n--- PCA-384 brute force ---")
    recalls = []
    t0 = time.time()
    for qi in q_idx:
        q = corpus_embs[qi]
        q384 = pca384[qi]

        # Ground truth
        sims = corpus_embs @ q / (corpus_norms * np.linalg.norm(q) + 1e-10)
        gt = set(np.argsort(sims)[-10:])

        # PCA-384
        sims384 = pca384 @ q384
        top10 = set(np.argsort(sims384)[-10:])
        recalls.append(len(gt & top10) / 10)

    elapsed = (time.time() - t0) / N_Q * 1000
    recalls = np.array(recalls)
    print("  Recall@10: %.4f  Latency: %.2f ms" % (recalls.mean(), elapsed))
    results["pca384_bruteforce"] = {
        "recall": float(recalls.mean()),
        "latency_ms": round(elapsed, 2),
    }

    # --- Method 3: PCA-64 brute force ---
    print("\n--- PCA-64 brute force ---")
    recalls = []
    t0 = time.time()
    for qi in q_idx:
        q = corpus_embs[qi]
        q64 = pca64[qi]

        sims = corpus_embs @ q / (corpus_norms * np.linalg.norm(q) + 1e-10)
        gt = set(np.argsort(sims)[-10:])

        sims64 = pca64 @ q64
        top10 = set(np.argsort(sims64)[-10:])
        recalls.append(len(gt & top10) / 10)

    elapsed = (time.time() - t0) / N_Q * 1000
    recalls = np.array(recalls)
    print("  Recall@10: %.4f  Latency: %.2f ms" % (recalls.mean(), elapsed))
    results["pca64_bruteforce"] = {
        "recall": float(recalls.mean()),
        "latency_ms": round(elapsed, 2),
    }

    # --- Method 4: Funnel PCA-64 top-K -> PCA-384 rerank ---
    for top_k_coarse in [50, 100, 200, 500]:
        label = "funnel_64to384_top%d" % top_k_coarse
        print("\n--- Funnel: PCA-64 top-%d -> PCA-384 rerank ---" % top_k_coarse)
        recalls = []
        t0 = time.time()
        for qi in q_idx:
            q = corpus_embs[qi]
            q64 = pca64[qi]
            q384 = pca384[qi]

            sims = corpus_embs @ q / (corpus_norms * np.linalg.norm(q) + 1e-10)
            gt = set(np.argsort(sims)[-10:])

            # Stage 1: PCA-64 coarse
            sims64 = pca64 @ q64
            topK = np.argsort(sims64)[-top_k_coarse:]

            # Stage 2: PCA-384 rerank
            sims384 = pca384[topK] @ q384
            top10_in_K = np.argsort(sims384)[-10:]
            final = set(topK[top10_in_K])

            recalls.append(len(gt & final) / 10)

        elapsed = (time.time() - t0) / N_Q * 1000
        recalls = np.array(recalls)
        print("  Recall@10: %.4f  Latency: %.2f ms" % (recalls.mean(), elapsed))
        results[label] = {
            "recall": float(recalls.mean()),
            "latency_ms": round(elapsed, 2),
            "coarse_k": top_k_coarse,
        }

    # --- Method 5: Funnel PCA-128 top-K -> PCA-384 rerank ---
    for top_k_coarse in [50, 100, 200]:
        label = "funnel_128to384_top%d" % top_k_coarse
        print("\n--- Funnel: PCA-128 top-%d -> PCA-384 rerank ---" % top_k_coarse)
        recalls = []
        t0 = time.time()
        for qi in q_idx:
            q = corpus_embs[qi]
            q128 = pca128[qi]
            q384 = pca384[qi]

            sims = corpus_embs @ q / (corpus_norms * np.linalg.norm(q) + 1e-10)
            gt = set(np.argsort(sims)[-10:])

            sims128 = pca128 @ q128
            topK = np.argsort(sims128)[-top_k_coarse:]

            sims384 = pca384[topK] @ q384
            top10_in_K = np.argsort(sims384)[-10:]
            final = set(topK[top10_in_K])

            recalls.append(len(gt & final) / 10)

        elapsed = (time.time() - t0) / N_Q * 1000
        recalls = np.array(recalls)
        print("  Recall@10: %.4f  Latency: %.2f ms" % (recalls.mean(), elapsed))
        results[label] = {
            "recall": float(recalls.mean()),
            "latency_ms": round(elapsed, 2),
            "coarse_k": top_k_coarse,
        }

    # Summary
    print("\n" + "=" * 65)
    print("  FUNNEL SEARCH BENCHMARK SUMMARY (%d corpus, %d queries)" % (N, N_Q))
    print("=" * 65)
    print("%-40s  recall@10  latency" % "Method")
    print("-" * 65)
    for k, v in sorted(results.items(), key=lambda x: -x[1]["recall"]):
        print("%-40s  %.4f     %.2f ms" % (k, v["recall"], v["latency_ms"]))

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to %s" % RESULTS_PATH)

    conn.close()


if __name__ == "__main__":
    main()
