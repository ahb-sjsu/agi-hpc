#!/usr/bin/env python3
"""Benchmark TurboQuant Pro integration across AGI-HPC on Atlas.

Measures:
  1. PCA-384 quality (cosine similarity vs full 1024-dim)
  2. RAG search accuracy (PCA-384 vs full embedding, recall@k)
  3. Search latency (PCA-384 vs full embedding)
  4. TurboQuant compression quality (TQ3, PCA+TQ3)
  5. NATS codec payload sizes
  6. Storage savings (actual table sizes)
  7. Embedding service throughput

Run on Atlas:
    python benchmark_tqpro_integration.py 2>&1 | tee /tmp/tqpro_benchmark.log
"""
from __future__ import annotations

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
RESULTS_PATH = "/home/claude/agi-hpc/benchmarks/tqpro_integration_benchmark.json"

results = {}


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ------------------------------------------------------------------
# 1. PCA-384 Quality
# ------------------------------------------------------------------
section("1. PCA-384 QUALITY (cosine similarity vs full 1024-dim)")

with open(PCA_PATH, "rb") as f:
    pca_data = pickle.load(f)

components = pca_data["components"].T.astype(np.float32)  # (1024, 384)
mean = pca_data["mean"].astype(np.float32)

conn = psycopg2.connect(DB_DSN)
cur = conn.cursor()

# Sample embeddings from chunks
cur.execute(
    "SELECT embedding FROM chunks WHERE embedding IS NOT NULL "
    "ORDER BY random() LIMIT 5000"
)
rows = cur.fetchall()
embeddings = np.array(
    [np.fromstring(r[0].strip("[]"), sep=",", dtype=np.float32) for r in rows]
)
print(f"Loaded {len(embeddings)} embeddings, shape {embeddings.shape}")

# PCA project and reconstruct
centered = embeddings - mean
projected = centered @ components  # (N, 384)
reconstructed = projected @ components.T + mean  # (N, 1024)

# Cosine similarity
dot = np.sum(embeddings * reconstructed, axis=1)
norm_a = np.linalg.norm(embeddings, axis=1)
norm_b = np.linalg.norm(reconstructed, axis=1)
cos = dot / (norm_a * norm_b + 1e-10)

print(f"PCA-384 round-trip cosine similarity:")
print(f"  Mean:   {cos.mean():.6f}")
print(f"  Median: {np.median(cos):.6f}")
print(f"  Min:    {cos.min():.6f}")
print(f"  Std:    {cos.std():.6f}")
print(f"  P5:     {np.percentile(cos, 5):.6f}")
print(f"  P95:    {np.percentile(cos, 95):.6f}")

results["pca_quality"] = {
    "n_samples": len(embeddings),
    "mean_cosine": float(cos.mean()),
    "median_cosine": float(np.median(cos)),
    "min_cosine": float(cos.min()),
    "std_cosine": float(cos.std()),
    "p5_cosine": float(np.percentile(cos, 5)),
    "p95_cosine": float(np.percentile(cos, 95)),
    "variance_captured": float(pca_data["variance_captured"]),
}

# ------------------------------------------------------------------
# 2. RAG Search Accuracy (PCA-384 vs Full 1024)
# ------------------------------------------------------------------
section("2. RAG SEARCH ACCURACY (PCA-384 vs full 1024-dim, recall@k)")

# Use 200 random queries, compare top-10 from both columns
N_QUERIES = 200
TOP_K = 10

cur.execute(
    "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL "
    "ORDER BY random() LIMIT %s",
    (N_QUERIES,),
)
query_rows = cur.fetchall()

recalls = []
for qid, qemb_str in query_rows:
    qemb = np.fromstring(qemb_str.strip("[]"), sep=",", dtype=np.float32)

    # Full 1024-dim search
    emb_str = str(qemb.tolist())
    cur.execute(
        "SELECT id FROM chunks "
        "ORDER BY embedding <=> %s::vector LIMIT %s",
        (emb_str, TOP_K),
    )
    full_ids = set(r[0] for r in cur.fetchall())

    # PCA-384 search
    centered_q = qemb - mean
    projected_q = centered_q @ components
    pnorm = np.linalg.norm(projected_q)
    if pnorm > 1e-10:
        projected_q = projected_q / pnorm
    pca_str = str(projected_q.tolist())
    cur.execute(
        "SELECT id FROM chunks "
        "ORDER BY embedding_pca384 <=> %s::vector LIMIT %s",
        (pca_str, TOP_K),
    )
    pca_ids = set(r[0] for r in cur.fetchall())

    recall = len(full_ids & pca_ids) / len(full_ids)
    recalls.append(recall)

recalls = np.array(recalls)
print(f"Recall@{TOP_K} (PCA-384 vs full 1024-dim):")
print(f"  N queries:  {N_QUERIES}")
print(f"  Mean:       {recalls.mean():.4f}")
print(f"  Median:     {np.median(recalls):.4f}")
print(f"  Min:        {recalls.min():.4f}")
print(f"  P10:        {np.percentile(recalls, 10):.4f}")
print(f"  % perfect:  {(recalls == 1.0).mean() * 100:.1f}%")

results["rag_recall"] = {
    "n_queries": N_QUERIES,
    "top_k": TOP_K,
    "mean_recall": float(recalls.mean()),
    "median_recall": float(np.median(recalls)),
    "min_recall": float(recalls.min()),
    "p10_recall": float(np.percentile(recalls, 10)),
    "pct_perfect": float((recalls == 1.0).mean() * 100),
}

# ------------------------------------------------------------------
# 3. Search Latency
# ------------------------------------------------------------------
section("3. SEARCH LATENCY (PCA-384 vs full 1024-dim)")

N_LATENCY = 50
qemb = embeddings[0]
emb_str = str(qemb.tolist())

# Full 1024-dim
t0 = time.perf_counter()
for i in range(N_LATENCY):
    cur.execute(
        "SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT 10",
        (emb_str,),
    )
    cur.fetchall()
full_lat = (time.perf_counter() - t0) / N_LATENCY * 1000

# PCA-384
centered_q = qemb - mean
projected_q = centered_q @ components
pnorm = np.linalg.norm(projected_q)
if pnorm > 1e-10:
    projected_q = projected_q / pnorm
pca_str = str(projected_q.tolist())

t0 = time.perf_counter()
for i in range(N_LATENCY):
    cur.execute(
        "SELECT id FROM chunks ORDER BY embedding_pca384 <=> %s::vector LIMIT 10",
        (pca_str,),
    )
    cur.fetchall()
pca_lat = (time.perf_counter() - t0) / N_LATENCY * 1000

print(f"Search latency (avg over {N_LATENCY} queries):")
print(f"  Full 1024-dim: {full_lat:.1f} ms")
print(f"  PCA-384:       {pca_lat:.1f} ms")
print(f"  Speedup:       {full_lat / pca_lat:.2f}x")

results["latency"] = {
    "n_queries": N_LATENCY,
    "full_1024_ms": round(full_lat, 2),
    "pca_384_ms": round(pca_lat, 2),
    "speedup": round(full_lat / pca_lat, 2),
}

# ------------------------------------------------------------------
# 4. TurboQuant Compression Quality
# ------------------------------------------------------------------
section("4. TURBOQUANT COMPRESSION QUALITY")

try:
    from turboquant_pro import TurboQuantPGVector

    sample = embeddings[:1000]

    for bits in [2, 3, 4]:
        tq = TurboQuantPGVector(dim=1024, bits=bits, seed=42)
        compressed = tq.compress_batch(sample)
        decompressed = tq.decompress_batch(compressed)

        cos_tq = np.sum(sample * decompressed, axis=1) / (
            np.linalg.norm(sample, axis=1) * np.linalg.norm(decompressed, axis=1) + 1e-10
        )
        size_per = compressed[0].size_bytes
        ratio = 1024 * 4 / size_per

        print(f"  TQ{bits}-bit: cosine={cos_tq.mean():.6f} min={cos_tq.min():.4f} "
              f"size={size_per}B ratio={ratio:.1f}x")

        results[f"tq{bits}_quality"] = {
            "bits": bits,
            "mean_cosine": float(cos_tq.mean()),
            "min_cosine": float(cos_tq.min()),
            "bytes_per_vec": size_per,
            "ratio": round(ratio, 1),
        }

    # PCA-384 + TQ3 combined
    sample_pca = (sample - mean) @ components
    norms = np.linalg.norm(sample_pca, axis=1, keepdims=True)
    sample_pca_normed = sample_pca / (norms + 1e-10)

    tq384 = TurboQuantPGVector(dim=384, bits=3, seed=42)
    compressed_pca = tq384.compress_batch(sample_pca_normed)
    decompressed_pca = tq384.decompress_batch(compressed_pca)

    # Reconstruct to 1024-dim for comparison
    recon_full = decompressed_pca * norms  # undo normalization
    recon_full = recon_full @ components.T + mean

    cos_combo = np.sum(sample * recon_full, axis=1) / (
        np.linalg.norm(sample, axis=1) * np.linalg.norm(recon_full, axis=1) + 1e-10
    )
    size_combo = compressed_pca[0].size_bytes
    ratio_combo = 1024 * 4 / size_combo

    print(f"\n  PCA-384 + TQ3: cosine={cos_combo.mean():.6f} min={cos_combo.min():.4f} "
          f"size={size_combo}B ratio={ratio_combo:.1f}x")

    results["pca384_tq3_quality"] = {
        "mean_cosine": float(cos_combo.mean()),
        "min_cosine": float(cos_combo.min()),
        "bytes_per_vec": size_combo,
        "ratio": round(ratio_combo, 1),
    }

except ImportError:
    print("  turboquant_pro not available, skipping")

# ------------------------------------------------------------------
# 5. NATS Codec Payload Sizes
# ------------------------------------------------------------------
section("5. NATS CODEC PAYLOAD SIZES")

try:
    from agi.common.embedding_codec import EmbeddingCodec

    codec = EmbeddingCodec(dim=1024, bits=3, pca_path=PCA_PATH)
    stats = codec.payload_size(embeddings[0])
    print(f"  Mode:             {codec.mode}")
    print(f"  Raw JSON payload: {stats['raw_bytes']:,} bytes")
    print(f"  Compressed:       {stats['compressed_bytes']:,} bytes")
    print(f"  Ratio:            {stats['ratio']}x")

    results["nats_codec"] = {
        "mode": codec.mode,
        "raw_bytes": stats["raw_bytes"],
        "compressed_bytes": stats["compressed_bytes"],
        "ratio": stats["ratio"],
    }
except Exception as e:
    print(f"  Error: {e}")

# ------------------------------------------------------------------
# 6. Storage Savings
# ------------------------------------------------------------------
section("6. STORAGE SAVINGS (actual table/index sizes)")

tables_info = {}
for table in ["chunks", "ethics_chunks", "episodes"]:
    try:
        cur.execute("SELECT pg_total_relation_size(%s)", (table,))
        total_size = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM %s" % table)
        row_count = cur.fetchone()[0]

        # Check for PCA column
        cur.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name=%s AND column_name='embedding_pca384'",
            (table,),
        )
        has_pca = cur.fetchone()[0] > 0

        if has_pca:
            cur.execute("SELECT count(*) FROM %s WHERE embedding_pca384 IS NOT NULL" % table)
            pca_count = cur.fetchone()[0]
        else:
            pca_count = 0

        info = {
            "rows": row_count,
            "total_size_mb": round(total_size / 1048576, 1),
            "has_pca384": has_pca,
            "pca384_populated": pca_count,
        }
        tables_info[table] = info
        print(f"  {table}:")
        print(f"    Rows:          {row_count:,}")
        print(f"    Total size:    {info['total_size_mb']:,.1f} MB")
        print(f"    PCA-384:       {pca_count:,} / {row_count:,}")
    except Exception as e:
        print(f"  {table}: error: {e}")
        conn.rollback()

# Index sizes
print("\n  Indexes:")
cur.execute("""
    SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid)),
           pg_relation_size(indexrelid) as bytes,
           idx_scan
    FROM pg_stat_user_indexes
    WHERE indexrelname LIKE '%%embed%%' OR indexrelname LIKE '%%pca%%'
    ORDER BY bytes DESC
""")
index_info = []
for row in cur.fetchall():
    name, size, bytes_val, scans = row
    print(f"    {name}: {size} ({scans} scans)")
    index_info.append({
        "name": name,
        "size": size,
        "bytes": bytes_val,
        "scans": scans,
    })

results["storage"] = {"tables": tables_info, "indexes": index_info}

# ------------------------------------------------------------------
# 7. Embedding Throughput
# ------------------------------------------------------------------
section("7. EMBEDDING THROUGHPUT")

# PCA projection throughput
batch_sizes = [100, 500, 1000]
for bs in batch_sizes:
    batch = embeddings[:bs]
    t0 = time.perf_counter()
    for _ in range(10):
        centered_b = batch - mean
        projected_b = centered_b @ components
        norms_b = np.linalg.norm(projected_b, axis=1, keepdims=True)
        projected_b = projected_b / (norms_b + 1e-10)
    elapsed = (time.perf_counter() - t0) / 10
    rate = bs / elapsed
    print(f"  PCA-384 projection: batch={bs}, {rate:,.0f} vec/s, {elapsed*1000:.1f} ms/batch")

results["throughput"] = {}
batch = embeddings[:1000]
t0 = time.perf_counter()
for _ in range(10):
    centered_b = batch - mean
    projected_b = centered_b @ components
    norms_b = np.linalg.norm(projected_b, axis=1, keepdims=True)
    projected_b = projected_b / (norms_b + 1e-10)
elapsed = (time.perf_counter() - t0) / 10
results["throughput"]["pca_384_vecs_per_sec"] = round(1000 / elapsed)

# TQ3 compression throughput
try:
    from turboquant_pro import TurboQuantPGVector
    tq = TurboQuantPGVector(dim=1024, bits=3, seed=42)
    t0 = time.perf_counter()
    for _ in range(5):
        tq.compress_batch(batch)
    elapsed = (time.perf_counter() - t0) / 5
    rate = 1000 / elapsed
    print(f"  TQ3 compress: 1000 vec, {rate:,.0f} vec/s, {elapsed*1000:.1f} ms/batch")
    results["throughput"]["tq3_compress_vecs_per_sec"] = round(rate)
except ImportError:
    pass

conn.close()

# ------------------------------------------------------------------
# Save results
# ------------------------------------------------------------------
section("RESULTS SAVED")

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"  -> {RESULTS_PATH}")
print(f"\nBenchmark complete.")
