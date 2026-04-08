#!/usr/bin/env python3
"""
GPU-accelerated embedding generation for ethics_chunks table.
Uses BGE-M3 on GPU 1 (CPU 1 package, cooler) for ~50x speedup over CPU.

Run: CUDA_VISIBLE_DEVICES=1 /home/claude/env/bin/python /archive/ethics-corpora/gpu_embed.py
"""
from __future__ import annotations

import logging
import math
import os
import sys
import time
from pathlib import Path

import psycopg2
import psycopg2.extras

BASE_DIR = Path("/archive/ethics-corpora")
DB_NAME = "atlas"
DB_USER = "claude"
BATCH_SIZE = 32  # Small batch to fit alongside llama-server (~24GB VRAM used)
EMBEDDING_DIM = 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "gpu_embed.log", mode="a"),
    ],
)
log = logging.getLogger("gpu_embed")


def get_db_conn():
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER)


def main():
    log.info("=" * 70)
    log.info("GPU EMBEDDING: BGE-M3 on ethics_chunks")
    log.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    log.info("=" * 70)

    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        device = "cuda"
    else:
        log.warning("No GPU available, falling back to CPU (will be slow)")
        device = "cpu"

    # Load model
    log.info("Loading BGE-M3 model...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    log.info(f"  Model loaded in {time.time() - t0:.1f}s")

    # Count chunks needing embeddings
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM ethics_chunks WHERE embedding IS NULL")
    total = cur.fetchone()[0]
    log.info(f"  {total} chunks need embeddings")

    if total == 0:
        log.info("Nothing to embed!")
        cur.close()
        conn.close()
        return

    # Process in batches
    processed = 0
    start_time = time.time()

    while processed < total:
        cur.execute(
            "SELECT id, content FROM ethics_chunks WHERE embedding IS NULL ORDER BY id LIMIT %s",
            (BATCH_SIZE,),
        )
        rows = cur.fetchall()
        if not rows:
            break

        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]

        # Generate embeddings on GPU
        try:
            embeddings = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.warning(f"OOM at batch size {len(texts)}, clearing cache and retrying with smaller batch")
                import torch
                torch.cuda.empty_cache()
                # Retry with half batch
                half = len(texts) // 2
                emb1 = model.encode(texts[:half], batch_size=half, show_progress_bar=False, normalize_embeddings=True)
                emb2 = model.encode(texts[half:], batch_size=half, show_progress_bar=False, normalize_embeddings=True)
                import numpy as np
                embeddings = np.concatenate([emb1, emb2], axis=0)
                torch.cuda.empty_cache()
            else:
                raise

        # Update database
        update_cur = conn.cursor()
        for chunk_id, emb in zip(ids, embeddings):
            emb_list = emb.tolist()
            update_cur.execute(
                "UPDATE ethics_chunks SET embedding = %s WHERE id = %s",
                (str(emb_list), chunk_id),
            )
        conn.commit()
        update_cur.close()

        processed += len(rows)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta_s = (total - processed) / rate if rate > 0 else 0
        pct = (processed / total) * 100
        log.info(
            f"  Embedded {processed}/{total} ({pct:.1f}%) "
            f"| {rate:.0f} chunks/s | ETA: {eta_s/60:.1f} min"
        )

    # Create IVFFlat index
    log.info("\nCreating IVFFlat index...")
    cur.execute("SELECT count(*) FROM ethics_chunks WHERE embedding IS NOT NULL")
    n_embedded = cur.fetchone()[0]
    n_lists = max(10, min(1000, int(math.sqrt(n_embedded))))

    cur.execute("DROP INDEX IF EXISTS idx_ethics_embedding;")
    cur.execute(f"""
        CREATE INDEX idx_ethics_embedding ON ethics_chunks
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = {n_lists});
    """)
    conn.commit()
    log.info(f"  IVFFlat index: {n_lists} lists for {n_embedded} rows")

    # Final stats
    cur.execute("""
        SELECT corpus, count(*), count(embedding)
        FROM ethics_chunks
        GROUP BY corpus
        ORDER BY corpus
    """)
    log.info("\n--- Final embedding stats ---")
    for corpus, total_c, embedded_c in cur.fetchall():
        log.info(f"  {corpus:20s} total={total_c:>8} embedded={embedded_c:>8}")

    cur.execute("SELECT count(*), count(embedding) FROM ethics_chunks")
    total_all, embedded_all = cur.fetchone()
    log.info(f"\n  TOTAL: {total_all} chunks, {embedded_all} embedded")

    elapsed_total = time.time() - start_time
    log.info(f"  Time: {elapsed_total/60:.1f} minutes ({processed/elapsed_total:.0f} chunks/s)")

    cur.close()
    conn.close()
    log.info("\nGPU EMBEDDING COMPLETE")


if __name__ == "__main__":
    main()
