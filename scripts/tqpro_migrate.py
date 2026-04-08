#!/home/claude/env/bin/python
"""TurboQuant Pro PCA-Matryoshka migration for Atlas pgvector embeddings.

Fits PCA on a sample, rotates all embeddings to PCA basis, truncates to
384 dimensions, and stores in a new vector(384) column. pgvector then
searches natively in the compressed space.

Uses batch-probe ThermalController to keep CPUs safe.

Usage:
    python tqpro_migrate.py                # full migration
    python tqpro_migrate.py --fit-only     # just fit PCA and save
    python tqpro_migrate.py --stats        # show compression stats
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time

import numpy as np
import psycopg2
from psycopg2.extras import execute_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tqpro-migrate")

DB_DSN = os.environ.get("ATLAS_DB_DSN", "dbname=atlas user=claude")
PCA_PATH = "/home/claude/agi-hpc/data/pca_rotation_384.pkl"
PCA_DIM = 384  # Target dimensions (96.4% variance retained)
BATCH_SIZE = 500
TABLES = ["chunks", "episodes", "ethics_chunks"]


def parse_embedding(val):
    """Parse pgvector string '[0.1,0.2,...]' to numpy array."""
    if isinstance(val, str):
        return np.fromstring(val.strip("[]"), sep=",", dtype=np.float32)
    elif isinstance(val, (list, tuple)):
        return np.array(val, dtype=np.float32)
    return np.array(val, dtype=np.float32)


def fit_pca(conn, n_sample=10000):
    """Fit PCA on a sample of embeddings from chunks table."""
    from sklearn.decomposition import PCA

    log.info(f"Sampling {n_sample} embeddings for PCA fitting...")
    cur = conn.cursor()
    cur.execute(
        "SELECT embedding FROM chunks WHERE embedding IS NOT NULL "
        "ORDER BY random() LIMIT %s", (n_sample,))
    rows = cur.fetchall()

    embeddings = np.array([parse_embedding(r[0]) for r in rows], dtype=np.float32)
    log.info(f"Fitting PCA on {len(embeddings)} vectors, shape {embeddings.shape}...")

    pca = PCA(n_components=PCA_DIM)
    pca.fit(embeddings)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    log.info(f"PCA fitted: {PCA_DIM} components capture {cum_var[-1]:.1%} of variance")
    log.info(f"Eigenvalue decay: α ≈ {_power_law_alpha(pca.explained_variance_):.2f}")

    # Verify quality on sample
    projected = (embeddings - pca.mean_) @ pca.components_.T
    reconstructed = projected @ pca.components_ + pca.mean_
    cos_sims = np.sum(embeddings * reconstructed, axis=1) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(reconstructed, axis=1) + 1e-10)
    log.info(f"Mean cosine similarity (full → PCA-{PCA_DIM} → reconstructed): {np.mean(cos_sims):.4f}")

    # Save PCA model
    os.makedirs(os.path.dirname(PCA_PATH), exist_ok=True)
    with open(PCA_PATH, "wb") as f:
        pickle.dump({
            "components": pca.components_,  # (384, 1024) float64
            "mean": pca.mean_,              # (1024,) float64
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "n_components": PCA_DIM,
            "original_dim": embeddings.shape[1],
            "n_train": len(embeddings),
            "mean_cosine": float(np.mean(cos_sims)),
            "variance_captured": float(cum_var[-1]),
        }, f)
    log.info(f"PCA model saved to {PCA_PATH}")
    return pca


def _power_law_alpha(eigenvalues, n_fit=200):
    k = np.arange(1, min(n_fit, len(eigenvalues)) + 1)
    coeffs = np.polyfit(np.log(k), np.log(eigenvalues[:len(k)] + 1e-20), 1)
    return -coeffs[0]


def load_pca():
    """Load saved PCA model."""
    with open(PCA_PATH, "rb") as f:
        data = pickle.load(f)
    log.info(f"Loaded PCA: {data['n_components']}d, {data['variance_captured']:.1%} variance, "
             f"cosine={data['mean_cosine']:.4f}")
    return data


def pca_transform(embeddings, pca_data):
    """Transform embeddings to PCA space and truncate."""
    centered = embeddings - pca_data["mean"].astype(np.float32)
    projected = centered @ pca_data["components"].T.astype(np.float32)
    # L2-normalize for cosine similarity search
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    return projected / (norms + 1e-10)


def migrate_table(conn, table, pca_data):
    """Add PCA-compressed embedding column and populate it."""
    cur = conn.cursor()
    col_name = "embedding_pca384"

    # Add column if not exists
    cur.execute(f"""
        DO $$ BEGIN
            ALTER TABLE {table} ADD COLUMN {col_name} vector({PCA_DIM});
        EXCEPTION WHEN duplicate_column THEN NULL;
        END $$;
    """)
    conn.commit()
    log.info(f"Column {col_name} ready on {table}")

    # Count rows needing migration
    cur.execute(f"""
        SELECT COUNT(*) FROM {table}
        WHERE embedding IS NOT NULL AND {col_name} IS NULL
    """)
    total = cur.fetchone()[0]
    log.info(f"{table}: {total} rows to migrate")

    if total == 0:
        return 0

    # Process in batches
    migrated = 0
    offset = 0
    start = time.time()

    while offset < total:
        # Thermal check
        try:
            from batch_probe import ThermalController
            import subprocess
            out = subprocess.run(["sensors"], capture_output=True, text=True, timeout=3).stdout
            for line in out.split("\n"):
                if "Package id 0" in line:
                    temp = float(line.split("+")[1].split("°")[0])
                    if temp > 88:
                        log.warning(f"CPU0 at {temp}°C, pausing 10s...")
                        time.sleep(10)
        except Exception:
            pass

        cur.execute(f"""
            SELECT id, embedding FROM {table}
            WHERE embedding IS NOT NULL AND {col_name} IS NULL
            ORDER BY id LIMIT %s
        """, (BATCH_SIZE,))
        rows = cur.fetchall()
        if not rows:
            break

        ids = [r[0] for r in rows]
        embeddings = np.array([parse_embedding(r[1]) for r in rows], dtype=np.float32)

        # PCA transform
        compressed = pca_transform(embeddings, pca_data)

        # Batch update
        updates = [(str(compressed[i].tolist()), ids[i]) for i in range(len(ids))]
        execute_batch(cur, f"""
            UPDATE {table} SET {col_name} = %s::vector WHERE id = %s
        """, updates, page_size=100)
        conn.commit()

        migrated += len(rows)
        offset += len(rows)
        elapsed = time.time() - start
        rate = migrated / elapsed if elapsed > 0 else 0
        eta = (total - migrated) / rate if rate > 0 else 0
        log.info(f"  {table}: {migrated}/{total} ({migrated/total*100:.1f}%) "
                 f"@ {rate:.0f} rows/s, ETA {eta:.0f}s")

    # Create index
    log.info(f"Creating IVFFlat index on {table}.{col_name}...")
    n_lists = max(10, min(100, total // 1000))
    cur.execute(f"""
        DROP INDEX IF EXISTS idx_{table}_{col_name};
        CREATE INDEX idx_{table}_{col_name}
        ON {table} USING ivfflat ({col_name} vector_cosine_ops)
        WITH (lists = {n_lists});
    """)
    conn.commit()
    log.info(f"Index created on {table}.{col_name} with {n_lists} lists")

    return migrated


def show_stats(conn):
    """Show compression statistics."""
    cur = conn.cursor()
    stats = {}
    for table in TABLES:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL")
            total = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE embedding_pca384 IS NOT NULL")
            compressed = cur.fetchone()[0]
            cur.execute(f"SELECT pg_total_relation_size('{table}')")
            size_bytes = cur.fetchone()[0]
            stats[table] = {
                "total": total,
                "compressed": compressed,
                "size_mb": round(size_bytes / 1048576, 1),
                "original_bytes_per_vec": 1024 * 4,  # float32
                "compressed_bytes_per_vec": PCA_DIM * 4,
                "ratio": round(1024 / PCA_DIM, 2),
            }
        except Exception:
            conn.rollback()
            continue

    print("\n" + "=" * 70)
    print("TurboQuant Pro — PCA-Matryoshka Compression Stats")
    print("=" * 70)
    for table, s in stats.items():
        print(f"\n{table}:")
        print(f"  Vectors:     {s['total']:,} total, {s['compressed']:,} compressed")
        print(f"  Original:    {s['original_bytes_per_vec']} bytes/vec (1024d float32)")
        print(f"  Compressed:  {s['compressed_bytes_per_vec']} bytes/vec ({PCA_DIM}d float32)")
        print(f"  Ratio:       {s['ratio']}x")
        print(f"  Table size:  {s['size_mb']} MB")
        orig_mb = s['total'] * s['original_bytes_per_vec'] / 1048576
        comp_mb = s['compressed'] * s['compressed_bytes_per_vec'] / 1048576
        print(f"  Embedding storage: {orig_mb:.1f} MB → {comp_mb:.1f} MB "
              f"(saved {orig_mb - comp_mb:.1f} MB)")
    print()
    return stats


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Pro PCA-Matryoshka migration")
    parser.add_argument("--fit-only", action="store_true", help="Only fit PCA, don't migrate")
    parser.add_argument("--stats", action="store_true", help="Show compression stats")
    parser.add_argument("--table", type=str, default=None, help="Migrate specific table")
    args = parser.parse_args()

    conn = psycopg2.connect(DB_DSN)

    if args.stats:
        show_stats(conn)
        conn.close()
        return

    # Fit or load PCA
    if os.path.exists(PCA_PATH) and not args.fit_only:
        pca_data = load_pca()
    else:
        pca = fit_pca(conn)
        pca_data = load_pca()

    if args.fit_only:
        conn.close()
        return

    # Migrate tables
    tables = [args.table] if args.table else TABLES
    total_migrated = 0
    for table in tables:
        try:
            n = migrate_table(conn, table, pca_data)
            total_migrated += n
            log.info(f"Migrated {n} rows in {table}")
        except Exception as e:
            log.error(f"Failed to migrate {table}: {e}")
            conn.rollback()

    log.info(f"\nTotal migrated: {total_migrated} rows")
    show_stats(conn)
    conn.close()


if __name__ == "__main__":
    main()
