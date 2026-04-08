#!/usr/bin/env python3
"""Migrate ethics_chunks table: add PCA-384 column + IVFFlat index.

Run AFTER ethics embedding is complete (all rows have embedding IS NOT NULL).

Uses the same PCA rotation matrix fitted on the chunks table. The paper
(Sec. 5.7) shows that PCA basis fitted on 10K vectors achieves within 0.001
cosine similarity of the full-corpus basis, so cross-table transfer is safe.

Usage:
    # Check readiness
    python migrate_ethics_pca384.py --check

    # Run migration (in tmux, will take ~hours for 2.4M rows)
    tmux new-session -d -s migrate
    tmux send-keys -t migrate 'taskset -c 12-23 python migrate_ethics_pca384.py 2>&1 | tee /tmp/ethics_pca_migrate.log' Enter
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
)
log = logging.getLogger("ethics-pca-migrate")

DB_DSN = "dbname=atlas user=claude"
PCA_PATH = "/home/claude/agi-hpc/data/pca_rotation_384.pkl"
PCA_DIM = 384
BATCH_SIZE = 1000
TABLE = "ethics_chunks"
COL = "embedding_pca384"


def parse_embedding(val):
    if isinstance(val, str):
        return np.fromstring(val.strip("[]"), sep=",", dtype=np.float32)
    return np.array(val, dtype=np.float32)


def check_readiness():
    """Check if ethics_chunks is ready for migration."""
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM ethics_chunks")
    total = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM ethics_chunks WHERE embedding IS NULL")
    null_count = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM ethics_chunks WHERE embedding IS NOT NULL")
    has_embed = cur.fetchone()[0]

    pca_exists = os.path.exists(PCA_PATH)

    # Check if column already exists
    cur.execute(
        "SELECT count(*) FROM information_schema.columns "
        "WHERE table_name='ethics_chunks' AND column_name=%s",
        (COL,),
    )
    col_exists = cur.fetchone()[0] > 0

    if col_exists:
        cur.execute(
            "SELECT count(*) FROM ethics_chunks WHERE %s IS NOT NULL" % COL
        )
        pca_done = cur.fetchone()[0]
    else:
        pca_done = 0

    conn.close()

    print("Ethics PCA-384 Migration Readiness")
    print("=" * 50)
    print("Total rows:          %d" % total)
    print("With embedding:      %d" % has_embed)
    print("Missing embedding:   %d" % null_count)
    print("PCA model exists:    %s" % pca_exists)
    print("PCA column exists:   %s" % col_exists)
    print("PCA populated:       %d" % pca_done)
    print()
    if null_count > 0:
        print("NOT READY: %d rows still missing embeddings." % null_count)
        print("Wait for embed_v2.py to finish.")
        return False
    if not pca_exists:
        print("NOT READY: PCA model not found at %s" % PCA_PATH)
        print("Run: python tqpro_migrate.py --fit-only")
        return False
    if pca_done == has_embed:
        print("ALREADY DONE: All rows have PCA-384.")
        return True
    print("READY: %d rows to migrate." % (has_embed - pca_done))
    return True


def migrate():
    # Load PCA
    with open(PCA_PATH, "rb") as f:
        pca_data = pickle.load(f)
    components = pca_data["components"].T.astype(np.float32)  # (1024, 384)
    mean = pca_data["mean"].astype(np.float32)
    log.info(
        "PCA loaded: %dd -> %dd, %.1f%% variance",
        pca_data["original_dim"],
        pca_data["n_components"],
        pca_data["variance_captured"] * 100,
    )

    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    # Add column
    cur.execute(
        """
        DO $$ BEGIN
            ALTER TABLE ethics_chunks ADD COLUMN %s vector(%s);
        EXCEPTION WHEN duplicate_column THEN NULL;
        END $$;
        """
        % (COL, PCA_DIM)
    )
    conn.commit()
    log.info("Column %s ready", COL)

    # Count
    cur.execute(
        "SELECT count(*) FROM ethics_chunks "
        "WHERE embedding IS NOT NULL AND %s IS NULL" % COL
    )
    total = cur.fetchone()[0]
    log.info("%d rows to migrate", total)

    if total == 0:
        log.info("Nothing to do.")
        conn.close()
        return

    migrated = 0
    start = time.time()

    while True:
        cur.execute(
            "SELECT id, embedding FROM ethics_chunks "
            "WHERE embedding IS NOT NULL AND %s IS NULL "
            "ORDER BY id LIMIT %%s" % COL,
            (BATCH_SIZE,),
        )
        rows = cur.fetchall()
        if not rows:
            break

        ids = [r[0] for r in rows]
        embeddings = np.array(
            [parse_embedding(r[1]) for r in rows], dtype=np.float32
        )

        # PCA project + L2 normalize
        centered = embeddings - mean
        projected = centered @ components
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        projected = projected / (norms + 1e-10)

        # Batch update
        updates = [
            (str(projected[i].tolist()), ids[i]) for i in range(len(ids))
        ]
        execute_batch(
            cur,
            "UPDATE ethics_chunks SET %s = %%s::vector WHERE id = %%s" % COL,
            updates,
            page_size=200,
        )
        conn.commit()

        migrated += len(rows)
        elapsed = time.time() - start
        rate = migrated / elapsed if elapsed > 0 else 0
        eta = (total - migrated) / rate if rate > 0 else 0
        if migrated % 10000 < BATCH_SIZE:
            log.info(
                "%d/%d (%.1f%%) @ %.0f rows/s, ETA %.0fs (%.1f min)",
                migrated,
                total,
                migrated / total * 100,
                rate,
                eta,
                eta / 60,
            )

    elapsed = time.time() - start
    log.info("Migration complete: %d rows in %.1f min", migrated, elapsed / 60)

    # Create IVFFlat index
    n_lists = min(1000, max(100, total // 2000))
    log.info(
        "Creating IVFFlat index with %d lists (this may take a while)...",
        n_lists,
    )
    cur.execute("DROP INDEX IF EXISTS idx_ethics_%s" % COL)
    cur.execute(
        "CREATE INDEX idx_ethics_%s ON ethics_chunks "
        "USING ivfflat (%s vector_cosine_ops) WITH (lists = %d)"
        % (COL, COL, n_lists)
    )
    conn.commit()
    log.info("Index created.")

    # Stats
    cur.execute(
        "SELECT pg_size_pretty(pg_relation_size('idx_ethics_%s'))" % COL
    )
    idx_size = cur.fetchone()[0]
    log.info("New index size: %s (vs ~16 GB for full 1024-dim index)", idx_size)

    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        check_readiness()
    else:
        if not check_readiness():
            sys.exit(1)
        migrate()
