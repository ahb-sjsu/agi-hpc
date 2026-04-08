#!/usr/bin/env python3
"""Benchmark all search tiers and validate each improvement.

Tests:
  Tier 0: Full 1024-dim pgvector brute force (baseline)
  Tier 1: Wiki article lookup (instant, if article exists)
  Tier 2a: PCA-384 IVFFlat (pgvector)
  Tier 2b: GPU Hamming funnel (binary top-200 → PCA-384 rerank)
  Tier 2c: tsvector FTS (keyword search)
  Tier 2d: Hybrid RRF (Hamming + FTS fusion)
  Tier 3: Vector-only on ethics corpus

For each tier: measures latency, recall@k vs baseline, and decides
whether the improvement is "worth it" based on the speed/recall tradeoff.

Also serves as a regression test — run after any search pipeline changes.

Usage:
    python benchmark_search_tiers.py         # full benchmark
    python benchmark_search_tiers.py --quick  # fast smoke test (10 queries)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import psycopg2

sys.path.insert(0, "/home/claude/agi-hpc/src")

DB_DSN = "dbname=atlas user=claude"
PCA_PATH = "/home/claude/agi-hpc/data/pca_rotation_384.pkl"
WIKI_DIR = "/archive/wiki"
RESULTS_PATH = "/home/claude/agi-hpc/benchmarks/search_tiers_benchmark.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    args = parser.parse_args()

    N_Q = 10 if args.quick else 50
    results = {}
    verdicts = {}

    # Load PCA
    with open(PCA_PATH, "rb") as f:
        pca = pickle.load(f)
    comp384 = pca["components"].T.astype(np.float32)
    mean = pca["mean"].astype(np.float32)

    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    # Load query set
    cur.execute(
        "SELECT id, embedding, content FROM chunks "
        "WHERE embedding IS NOT NULL ORDER BY random() LIMIT %s",
        (N_Q,),
    )
    query_rows = cur.fetchall()
    print("Loaded %d queries" % len(query_rows))

    norms_cache = {}

    def get_ground_truth(qemb):
        """Exact 1024-dim cosine search (ground truth)."""
        emb_str = str(qemb.tolist())
        cur.execute("SET enable_indexscan = off")
        cur.execute(
            "SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT 10",
            (emb_str,),
        )
        gt = [r[0] for r in cur.fetchall()]
        cur.execute("SET enable_indexscan = on")
        return gt

    def recall_at_k(retrieved, ground_truth, k=10):
        gt_set = set(ground_truth[:k])
        ret_set = set(retrieved[:k])
        return len(gt_set & ret_set) / len(gt_set) if gt_set else 0.0

    # ------------------------------------------------------------------
    # Tier 0: Full 1024-dim brute force (baseline)
    # ------------------------------------------------------------------
    print("\n=== Tier 0: Full 1024-dim exact (baseline) ===")
    t0 = time.time()
    ground_truths = []
    for qid, qemb_str, qcontent in query_rows:
        qemb = np.fromstring(qemb_str.strip("[]"), sep=",", dtype=np.float32)
        gt = get_ground_truth(qemb)
        ground_truths.append(gt)
    baseline_lat = (time.time() - t0) / len(query_rows) * 1000
    print("  Latency: %.1f ms (sequential scan, ground truth)" % baseline_lat)
    results["tier0_baseline"] = {"latency_ms": round(baseline_lat, 1), "recall": 1.0}

    # ------------------------------------------------------------------
    # Tier 2a: PCA-384 IVFFlat
    # ------------------------------------------------------------------
    print("\n=== Tier 2a: PCA-384 IVFFlat (probes=10) ===")
    cur.execute("SET ivfflat.probes = 10")
    recalls = []
    t0 = time.time()
    for i, (qid, qemb_str, qcontent) in enumerate(query_rows):
        qemb = np.fromstring(qemb_str.strip("[]"), sep=",", dtype=np.float32)
        centered = qemb - mean
        projected = centered @ comp384
        norm = np.linalg.norm(projected)
        if norm > 1e-10:
            projected = projected / norm
        pca_str = str(projected.tolist())
        cur.execute(
            "SELECT id FROM chunks ORDER BY embedding_pca384 <=> %s::vector LIMIT 10",
            (pca_str,),
        )
        retrieved = [r[0] for r in cur.fetchall()]
        recalls.append(recall_at_k(retrieved, ground_truths[i]))
    pca_lat = (time.time() - t0) / len(query_rows) * 1000
    pca_recall = np.mean(recalls)
    print("  Recall@10: %.4f  Latency: %.1f ms" % (pca_recall, pca_lat))
    results["tier2a_pca384_ivfflat"] = {
        "recall": float(pca_recall),
        "latency_ms": round(pca_lat, 1),
    }
    verdicts["pca384_ivfflat"] = (
        "WORTH IT" if pca_lat < baseline_lat * 0.5 and pca_recall > 0.85
        else "MARGINAL" if pca_recall > 0.80
        else "NOT WORTH IT"
    )

    # ------------------------------------------------------------------
    # Tier 2b: GPU Hamming funnel
    # ------------------------------------------------------------------
    print("\n=== Tier 2b: GPU Hamming funnel ===")
    try:
        from turboquant_pro.cuda_search import gpu_hamming_search, pack_binary

        # Load index
        cur.execute(
            "SELECT id, embedding_pca384 FROM chunks "
            "WHERE embedding_pca384 IS NOT NULL"
        )
        idx_rows = cur.fetchall()
        chunk_ids = [r[0] for r in idx_rows]
        pca_vecs = np.array(
            [np.fromstring(r[1].strip("[]"), sep=",", dtype=np.float32) for r in idx_rows]
        )
        binary_vecs = (pca_vecs > 0).astype(np.uint8)
        packed_binary = pack_binary(binary_vecs)

        recalls = []
        t0 = time.time()
        for i, (qid, qemb_str, qcontent) in enumerate(query_rows):
            qemb = np.fromstring(qemb_str.strip("[]"), sep=",", dtype=np.float32)
            centered = qemb - mean
            projected = centered @ comp384
            norm = np.linalg.norm(projected)
            if norm > 1e-10:
                projected = projected / norm

            q_binary = (projected > 0).astype(np.uint8)
            q_packed = pack_binary(q_binary[np.newaxis, :])[0]
            coarse_idx, _ = gpu_hamming_search(q_packed, packed_binary, top_k=200)

            candidate_pca = pca_vecs[coarse_idx]
            scores = candidate_pca @ projected
            rerank = np.argsort(scores)[::-1][:10]
            retrieved = [chunk_ids[coarse_idx[j]] for j in rerank]
            recalls.append(recall_at_k(retrieved, ground_truths[i]))

        hamming_lat = (time.time() - t0) / len(query_rows) * 1000
        hamming_recall = np.mean(recalls)
        print("  Recall@10: %.4f  Latency: %.1f ms" % (hamming_recall, hamming_lat))
        results["tier2b_hamming_funnel"] = {
            "recall": float(hamming_recall),
            "latency_ms": round(hamming_lat, 1),
        }
        verdicts["hamming_funnel"] = (
            "WORTH IT" if hamming_lat < pca_lat and hamming_recall > 0.80
            else "MARGINAL"
        )
    except ImportError:
        print("  SKIPPED (no CuPy/turboquant_pro)")

    # ------------------------------------------------------------------
    # Tier 2c: tsvector FTS
    # ------------------------------------------------------------------
    print("\n=== Tier 2c: tsvector FTS (keyword) ===")
    cur.execute("SELECT count(*) FROM chunks WHERE tsv IS NOT NULL")
    tsv_count = cur.fetchone()[0]
    if tsv_count > 0:
        recalls = []
        t0 = time.time()
        for i, (qid, qemb_str, qcontent) in enumerate(query_rows):
            # Use content as query text (first 100 chars)
            query_text = qcontent[:100]
            cur.execute(
                "SELECT id FROM chunks "
                "WHERE tsv @@ plainto_tsquery('english', %s) "
                "ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC "
                "LIMIT 10",
                (query_text, query_text),
            )
            retrieved = [r[0] for r in cur.fetchall()]
            recalls.append(recall_at_k(retrieved, ground_truths[i]))

        fts_lat = (time.time() - t0) / len(query_rows) * 1000
        fts_recall = np.mean(recalls)
        print("  Recall@10: %.4f  Latency: %.1f ms" % (fts_recall, fts_lat))
        results["tier2c_fts"] = {
            "recall": float(fts_recall),
            "latency_ms": round(fts_lat, 1),
        }
        verdicts["fts"] = (
            "COMPLEMENTARY" if fts_recall > 0.1
            else "NOT USEFUL"
        )
    else:
        print("  SKIPPED (tsv column not populated)")

    # ------------------------------------------------------------------
    # Tier 2d: Hybrid RRF (vector + FTS)
    # ------------------------------------------------------------------
    print("\n=== Tier 2d: Hybrid RRF (Hamming + FTS) ===")
    if tsv_count > 0 and "tier2b_hamming_funnel" in results:
        from agi.common.hybrid_search import reciprocal_rank_fusion

        recalls = []
        t0 = time.time()
        for i, (qid, qemb_str, qcontent) in enumerate(query_rows):
            qemb = np.fromstring(qemb_str.strip("[]"), sep=",", dtype=np.float32)
            centered = qemb - mean
            projected = centered @ comp384
            norm = np.linalg.norm(projected)
            if norm > 1e-10:
                projected = projected / norm

            # Vector results
            q_binary = (projected > 0).astype(np.uint8)
            q_packed = pack_binary(q_binary[np.newaxis, :])[0]
            coarse_idx, _ = gpu_hamming_search(q_packed, packed_binary, top_k=200)
            candidate_pca = pca_vecs[coarse_idx]
            scores = candidate_pca @ projected
            rerank = np.argsort(scores)[::-1][:30]
            vector_ids = [chunk_ids[coarse_idx[j]] for j in rerank]

            # FTS results
            query_text = qcontent[:100]
            cur.execute(
                "SELECT id FROM chunks "
                "WHERE tsv @@ plainto_tsquery('english', %s) "
                "ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC "
                "LIMIT 30",
                (query_text, query_text),
            )
            fts_ids = [r[0] for r in cur.fetchall()]

            # RRF fusion
            rrf = reciprocal_rank_fusion([vector_ids, fts_ids])
            hybrid_top10 = list(rrf.keys())[:10]
            recalls.append(recall_at_k(hybrid_top10, ground_truths[i]))

        hybrid_lat = (time.time() - t0) / len(query_rows) * 1000
        hybrid_recall = np.mean(recalls)
        print("  Recall@10: %.4f  Latency: %.1f ms" % (hybrid_recall, hybrid_lat))
        results["tier2d_hybrid_rrf"] = {
            "recall": float(hybrid_recall),
            "latency_ms": round(hybrid_lat, 1),
        }
        verdicts["hybrid_rrf"] = (
            "WORTH IT" if hybrid_recall > hamming_recall + 0.01
            else "MARGINAL — FTS adds latency without recall gain"
        )
    else:
        print("  SKIPPED (requires both Hamming and FTS)")

    # ------------------------------------------------------------------
    # Tier 1: Wiki lookup
    # ------------------------------------------------------------------
    print("\n=== Tier 1: Wiki article lookup ===")
    from agi.common.hybrid_search import WikiIndex
    wiki = WikiIndex(WIKI_DIR)
    n_articles = wiki.load()
    print("  Wiki articles: %d" % n_articles)
    if n_articles > 0:
        t0 = time.time()
        hits = 0
        for qid, qemb_str, qcontent in query_rows:
            articles = wiki.lookup(qcontent[:50], top_k=1)
            if articles:
                hits += 1
        wiki_lat = (time.time() - t0) / len(query_rows) * 1000
        print("  Hit rate: %d/%d (%.0f%%)" % (hits, len(query_rows), hits / len(query_rows) * 100))
        print("  Latency: %.2f ms" % wiki_lat)
        results["tier1_wiki"] = {
            "articles": n_articles,
            "hit_rate": hits / len(query_rows),
            "latency_ms": round(wiki_lat, 2),
        }
        verdicts["wiki"] = (
            "WORTH IT" if n_articles > 5 and hits > 0
            else "NOT YET — compile wiki first"
        )
    else:
        print("  No wiki compiled yet. Run: python compile_wiki.py")
        verdicts["wiki"] = "NOT YET — compile wiki first"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SEARCH TIER BENCHMARK SUMMARY (%d queries)" % N_Q)
    print("=" * 70)
    print("%-30s  recall  latency  verdict" % "Tier")
    print("-" * 70)
    for key in sorted(results.keys()):
        v = results[key]
        r = v.get("recall", v.get("hit_rate", 0))
        lat = v.get("latency_ms", 0)
        verdict_key = key.replace("tier0_", "").replace("tier1_", "").replace("tier2a_", "").replace("tier2b_", "").replace("tier2c_", "").replace("tier2d_", "")
        verd = verdicts.get(verdict_key, "")
        print("%-30s  %.4f  %6.1f   %s" % (key, r, lat, verd))

    # Save
    output = {"results": results, "verdicts": verdicts, "n_queries": N_Q}
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to %s" % RESULTS_PATH)

    conn.close()


if __name__ == "__main__":
    main()
