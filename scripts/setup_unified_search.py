#!/usr/bin/env python3
"""Set up unified search across all 3.3M corpus documents.

Creates tsvector columns, IVFFlat indexes, and a unified search view
for chunks + ethics_chunks + publications.

Run on Atlas:
    python setup_unified_search.py
    python setup_unified_search.py --status   # check index status
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("unified-search")

DB_DSN = "dbname=atlas user=claude"


def setup_tsvectors(conn):
    """Add and populate tsvector columns where missing."""
    cur = conn.cursor()

    # ethics_chunks: content + tradition + source_ref
    cur.execute(
        "SELECT count(*) FROM information_schema.columns "
        "WHERE table_name='ethics_chunks' AND column_name='tsv'"
    )
    if cur.fetchone()[0] == 0:
        log.info("Adding tsv column to ethics_chunks...")
        cur.execute("ALTER TABLE ethics_chunks ADD COLUMN tsv tsvector")
        conn.commit()

    cur.execute("SELECT count(*) FROM ethics_chunks WHERE tsv IS NULL")
    null_count = cur.fetchone()[0]
    if null_count > 0:
        log.info("Populating tsv on ethics_chunks (%d rows)...", null_count)
        t0 = time.time()
        cur.execute(
            "UPDATE ethics_chunks SET tsv = "
            "to_tsvector('english', "
            "  coalesce(content, '') || ' ' || "
            "  coalesce(tradition, '') || ' ' || "
            "  coalesce(source_ref, '') || ' ' || "
            "  coalesce(language, '')"
            ") WHERE tsv IS NULL"
        )
        conn.commit()
        log.info("  Updated %d rows in %.0fs", cur.rowcount, time.time() - t0)

        log.info("Creating GIN index on ethics_chunks.tsv...")
        cur.execute("DROP INDEX IF EXISTS idx_ethics_tsv")
        cur.execute(
            "CREATE INDEX idx_ethics_tsv ON ethics_chunks USING gin(tsv)"
        )
        conn.commit()
        log.info("  Done")

    # publications: title + author + topic
    cur.execute(
        "SELECT count(*) FROM information_schema.columns "
        "WHERE table_name='publications' AND column_name='tsv'"
    )
    if cur.fetchone()[0] == 0:
        log.info("Adding tsv column to publications...")
        cur.execute("ALTER TABLE publications ADD COLUMN tsv tsvector")
        conn.commit()

    cur.execute("SELECT count(*) FROM publications WHERE tsv IS NULL")
    null_count = cur.fetchone()[0]
    if null_count > 0:
        log.info("Populating tsv on publications (%d rows)...", null_count)
        t0 = time.time()
        cur.execute(
            "UPDATE publications SET tsv = "
            "to_tsvector('english', "
            "  coalesce(title, '') || ' ' || "
            "  coalesce(author, '') || ' ' || "
            "  coalesce(topic, '')"
            ") WHERE tsv IS NULL"
        )
        conn.commit()
        log.info("  Updated %d rows in %.0fs", cur.rowcount, time.time() - t0)

        log.info("Creating GIN index on publications.tsv...")
        cur.execute("DROP INDEX IF EXISTS idx_publications_tsv")
        cur.execute(
            "CREATE INDEX idx_publications_tsv ON publications USING gin(tsv)"
        )
        conn.commit()
        log.info("  Done")


def verify_pca_indexes(conn):
    """Ensure PCA-384 IVFFlat indexes exist on all tables."""
    cur = conn.cursor()

    for table, expected_lists in [
        ("chunks", 300),
        ("ethics_chunks", 1000),
        ("publications", 400),
    ]:
        idx_name = "idx_%s_embedding_pca384" % table
        cur.execute(
            "SELECT count(*) FROM pg_indexes WHERE indexname=%s",
            (idx_name,),
        )
        if cur.fetchone()[0] == 0:
            # Check if there's a differently-named PCA index
            cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE relname=%s AND indexdef LIKE '%%pca384%%'",
                (table,),
            )
            existing = cur.fetchone()
            if existing:
                log.info("%s: PCA-384 index exists as %s", table, existing[0])
            else:
                cur.execute(
                    "SELECT count(*) FROM %s WHERE embedding_pca384 IS NOT NULL" % table
                )
                pca_count = cur.fetchone()[0]
                if pca_count == 0:
                    log.warning("%s: no PCA-384 embeddings, skipping index", table)
                    continue

                log.info("Creating IVFFlat index on %s.embedding_pca384 (%d lists)...",
                         table, expected_lists)
                cur.execute("SET maintenance_work_mem = '512MB'")
                t0 = time.time()
                cur.execute(
                    "CREATE INDEX %s ON %s "
                    "USING ivfflat (embedding_pca384 vector_cosine_ops) "
                    "WITH (lists = %d)" % (idx_name, table, expected_lists)
                )
                conn.commit()
                log.info("  Done in %.0fs", time.time() - t0)
        else:
            log.info("%s: PCA-384 index exists", table)


def show_status(conn):
    """Show unified search readiness status."""
    cur = conn.cursor()

    print("\n" + "=" * 70)
    print("  UNIFIED SEARCH STATUS")
    print("=" * 70)

    total_searchable = 0
    for table, desc in [
        ("chunks", "Code repos (27 repos)"),
        ("ethics_chunks", "Ethics corpus (9 traditions)"),
        ("publications", "Publications catalog"),
    ]:
        cur.execute("SELECT count(*) FROM %s" % table)
        total = cur.fetchone()[0]

        cur.execute(
            "SELECT count(*) FROM %s WHERE embedding_pca384 IS NOT NULL" % table
        )
        pca = cur.fetchone()[0]

        has_tsv = False
        tsv_count = 0
        cur.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name=%s AND column_name='tsv'",
            (table,),
        )
        if cur.fetchone()[0] > 0:
            has_tsv = True
            cur.execute("SELECT count(*) FROM %s WHERE tsv IS NOT NULL" % table)
            tsv_count = cur.fetchone()[0]

        # Index info
        cur.execute(
            "SELECT indexname, pg_size_pretty(pg_relation_size(quote_ident(indexname)::regclass)) "
            "FROM pg_indexes "
            "WHERE tablename=%s AND indexdef LIKE '%%pca384%%'",
            (table,),
        )
        idx = cur.fetchone()

        total_searchable += pca

        print("\n%s (%s):" % (table, desc))
        print("  Total rows:   %d" % total)
        print("  PCA-384:      %d (%.0f%%)" % (pca, pca / total * 100 if total else 0))
        print("  tsvector:     %d (%.0f%%)" % (tsv_count, tsv_count / total * 100 if total else 0))
        if idx:
            print("  PCA index:    %s (%s)" % (idx[0], idx[1]))
        else:
            print("  PCA index:    MISSING")

    print("\n  TOTAL SEARCHABLE: %d vectors across 3 corpora" % total_searchable)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    conn = psycopg2.connect(DB_DSN)

    if args.status:
        show_status(conn)
        conn.close()
        return

    log.info("Setting up unified search across all corpora...")
    setup_tsvectors(conn)
    verify_pca_indexes(conn)
    show_status(conn)
    conn.close()
    log.info("Done!")


if __name__ == "__main__":
    main()
