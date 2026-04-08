#!/usr/bin/env python3
"""Embed publications table (824K papers) with BGE-M3.

Concatenates title + author + topic for each publication, generates
1024-dim BGE-M3 embeddings, and also computes PCA-384 projections.

Run on Atlas GPU 1:
    tmux new-session -d -s pub-embed
    tmux send-keys -t pub-embed \
        'CUDA_VISIBLE_DEVICES=1 taskset -c 36-43 \
         /home/claude/env/bin/python3 \
         /home/claude/agi-hpc/scripts/embed_publications.py \
         2>&1 | tee /tmp/pub_embed.log' Enter
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
import time

import numpy as np
import psycopg2
from psycopg2.extras import execute_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/pub_embed.log", mode="a"),
    ],
)
log = logging.getLogger("pub-embed")

DB_DSN = "dbname=atlas user=claude"
PCA_PATH = "/home/claude/agi-hpc/data/pca_rotation_384.pkl"
BATCH_SIZE = 128
MODEL_NAME = "BAAI/bge-m3"


def main():
    log.info("=" * 60)
    log.info("PUBLICATIONS EMBEDDING")
    log.info("=" * 60)

    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    # Add columns if needed
    for col, dim in [("embedding", 1024), ("embedding_pca384", 384)]:
        cur.execute(
            """
            DO $$ BEGIN
                ALTER TABLE publications ADD COLUMN %s vector(%s);
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
            """
            % (col, dim)
        )
    conn.commit()
    log.info("Columns ready")

    # Count work
    cur.execute("SELECT count(*) FROM publications WHERE embedding IS NULL")
    remaining = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM publications")
    total = cur.fetchone()[0]
    log.info("Total: %d, remaining: %d", total, remaining)

    if remaining == 0:
        log.info("All publications already embedded!")
        conn.close()
        return

    # Load model
    log.info("Loading %s...", MODEL_NAME)
    from sentence_transformers import SentenceTransformer

    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    log.info("Model loaded on %s", device)

    # Load PCA
    pca_components = None
    pca_mean = None
    if os.path.exists(PCA_PATH):
        with open(PCA_PATH, "rb") as f:
            pca = pickle.load(f)
        pca_components = pca["components"].T.astype(np.float32)
        pca_mean = pca["mean"].astype(np.float32)
        log.info("PCA-384 loaded (%.1f%% variance)", pca["variance_captured"] * 100)

    # Process in batches
    embedded = 0
    start_time = time.time()

    while True:
        cur.execute(
            "SELECT id, title, author, topic FROM publications "
            "WHERE embedding IS NULL ORDER BY id LIMIT %s",
            (BATCH_SIZE,),
        )
        rows = cur.fetchall()
        if not rows:
            break

        ids = [r[0] for r in rows]
        texts = []
        for r in rows:
            parts = []
            if r[1]:
                parts.append(r[1])  # title
            if r[2]:
                parts.append("by " + r[2])  # author
            if r[3]:
                parts.append("(" + r[3] + ")")  # topic
            texts.append(" ".join(parts) if parts else "untitled")

        # Embed
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=BATCH_SIZE)
        embeddings = np.array(embeddings, dtype=np.float32)

        # PCA project
        pca_embeddings = None
        if pca_components is not None:
            centered = embeddings - pca_mean
            projected = centered @ pca_components
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            pca_embeddings = projected / (norms + 1e-10)

        # Update DB
        updates = []
        for i in range(len(ids)):
            emb_str = str(embeddings[i].tolist())
            pca_str = str(pca_embeddings[i].tolist()) if pca_embeddings is not None else None
            if pca_str:
                updates.append((emb_str, pca_str, ids[i]))
            else:
                updates.append((emb_str, ids[i]))

        if pca_embeddings is not None:
            execute_batch(
                cur,
                "UPDATE publications SET embedding = %s::vector, "
                "embedding_pca384 = %s::vector WHERE id = %s",
                updates,
                page_size=100,
            )
        else:
            execute_batch(
                cur,
                "UPDATE publications SET embedding = %s::vector WHERE id = %s",
                updates,
                page_size=100,
            )
        conn.commit()

        embedded += len(rows)
        elapsed = time.time() - start_time
        rate = embedded / elapsed if elapsed > 0 else 0
        eta = (remaining - embedded) / rate if rate > 0 else 0

        if embedded % (BATCH_SIZE * 10) < BATCH_SIZE:
            log.info(
                "Embedded %d/%d (%.1f%%) @ %.1f vec/s, ETA %.0fs (%.1f min)",
                embedded,
                remaining,
                embedded / remaining * 100,
                rate,
                eta,
                eta / 60,
            )

    elapsed = time.time() - start_time
    log.info("Done! Embedded %d publications in %.1f min", embedded, elapsed / 60)

    # Create indexes
    log.info("Creating IVFFlat indexes...")
    n_lists = min(500, max(50, total // 2000))

    for col in ["embedding", "embedding_pca384"]:
        dim = 1024 if col == "embedding" else 384
        idx_name = "idx_publications_%s" % col
        cur.execute("DROP INDEX IF EXISTS %s" % idx_name)
        cur.execute(
            "SET maintenance_work_mem = '512MB'"
        )
        cur.execute(
            "CREATE INDEX %s ON publications "
            "USING ivfflat (%s vector_cosine_ops) WITH (lists = %d)"
            % (idx_name, col, n_lists)
        )
        conn.commit()
        cur.execute(
            "SELECT pg_size_pretty(pg_relation_size('%s'))" % idx_name
        )
        size = cur.fetchone()[0]
        log.info("Index %s: %s", idx_name, size)

    conn.close()
    log.info("All done!")


if __name__ == "__main__":
    main()
