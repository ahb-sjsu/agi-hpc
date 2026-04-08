#!/usr/bin/env python3
"""Harvest academic paper metadata from open APIs and embed for search.

Sources:
  1. CORE (200M+ papers, free API, 10K/day free tier)
  2. PubMed Central (11.6M papers, OAI-PMH, unlimited)
  3. Unpaywall (20M papers, free database snapshot)
  4. DOAJ (21K journals, free API)

All sources produce the same schema: title + abstract + metadata,
stored in academic_papers table with embedding + PCA-384 + tsvector.

Usage:
    python harvest_academic_sources.py --source pubmed    # start PubMed harvest
    python harvest_academic_sources.py --source core      # start CORE harvest
    python harvest_academic_sources.py --source unpaywall # load Unpaywall snapshot
    python harvest_academic_sources.py --source doaj      # small, quick
    python harvest_academic_sources.py --source all       # everything
    python harvest_academic_sources.py --status           # show progress
    python harvest_academic_sources.py --embed            # embed unharvested rows

Architecture:
    Harvest and embedding are separated so harvesting can run on CPU
    while embedding runs on GPU. Harvest stores text-only rows; a
    separate --embed pass fills in the vector columns.

Best run as two parallel processes:
    # Terminal 1: harvest (CPU-only, network-bound)
    python harvest_academic_sources.py --source pubmed

    # Terminal 2: embed (GPU, runs continuously on un-embedded rows)
    CUDA_VISIBLE_DEVICES=1 python harvest_academic_sources.py --embed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("harvest")

DB_DSN = "dbname=atlas user=claude"
PCA_PATH = "/home/claude/agi-hpc/data/pca_rotation_384.pkl"
BATCH_SIZE = 128


def init_db(conn):
    """Create the unified academic_papers table."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS academic_papers (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            external_id TEXT,
            title TEXT NOT NULL,
            abstract TEXT,
            authors TEXT,
            year TEXT,
            journal TEXT,
            doi TEXT,
            categories TEXT,
            language TEXT DEFAULT 'en',
            harvested_at TIMESTAMP DEFAULT now(),
            embedding vector(1024),
            embedding_pca384 vector(384),
            tsv tsvector
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_acad_source ON academic_papers(source)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_acad_doi ON academic_papers(doi) "
        "WHERE doi IS NOT NULL"
    )
    conn.commit()
    log.info("academic_papers table ready")


# ------------------------------------------------------------------ #
# Source 1: PubMed Central (OAI-PMH, free, unlimited)                 #
# ------------------------------------------------------------------ #

PUBMED_OAI_BASE = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
PUBMED_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
}


def harvest_pubmed(conn, from_date="2020-01-01", max_records=None):
    """Harvest PubMed Central via OAI-PMH ListRecords."""
    cur = conn.cursor()
    total = 0
    resumption_token = None

    log.info("Harvesting PubMed Central from %s...", from_date)

    while True:
        # Build request URL
        if resumption_token:
            url = f"{PUBMED_OAI_BASE}?verb=ListRecords&resumptionToken={resumption_token}"
        else:
            url = (
                f"{PUBMED_OAI_BASE}?verb=ListRecords"
                f"&metadataPrefix=oai_dc&from={from_date}"
            )

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Atlas-AI/1.0 (research)"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                xml_data = resp.read()
        except Exception as e:
            log.warning("PubMed request failed: %s, retrying in 30s...", e)
            time.sleep(30)
            continue

        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError:
            log.warning("XML parse error, retrying in 30s...")
            time.sleep(30)
            continue

        # Extract records
        records = root.findall(".//oai:record", PUBMED_NS)
        batch = []
        for record in records:
            header = record.find("oai:header", PUBMED_NS)
            if header is None or header.get("status") == "deleted":
                continue

            identifier = header.findtext("oai:identifier", "", PUBMED_NS)
            datestamp = header.findtext("oai:datestamp", "", PUBMED_NS)

            metadata = record.find("oai:metadata", PUBMED_NS)
            if metadata is None:
                continue

            dc = metadata.find("oai_dc:dc", PUBMED_NS)
            if dc is None:
                continue

            title = dc.findtext("dc:title", "", PUBMED_NS).strip()
            if not title:
                continue

            descriptions = dc.findall("dc:description", PUBMED_NS)
            abstract = " ".join(d.text or "" for d in descriptions).strip()

            creators = dc.findall("dc:creator", PUBMED_NS)
            authors = "; ".join(c.text or "" for c in creators)

            subjects = dc.findall("dc:subject", PUBMED_NS)
            categories = ", ".join(s.text or "" for s in subjects)

            # Extract DOI from identifier
            doi = None
            identifiers = dc.findall("dc:identifier", PUBMED_NS)
            for ident in identifiers:
                if ident.text and "doi" in ident.text.lower():
                    doi = ident.text.strip()

            paper_id = hashlib.md5(f"pubmed:{identifier}".encode()).hexdigest()

            batch.append((
                paper_id, "pubmed", identifier, title, abstract[:5000],
                authors[:1000], datestamp[:4], None, doi,
                categories[:500], "en",
                f"{title} {abstract}",
            ))

        if batch:
            execute_batch(
                cur,
                "INSERT INTO academic_papers "
                "(id, source, external_id, title, abstract, authors, year, "
                "journal, doi, categories, language, tsv) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "to_tsvector('english', %s)) "
                "ON CONFLICT (id) DO NOTHING",
                batch,
                page_size=100,
            )
            conn.commit()
            total += len(batch)

        if total % 5000 < len(batch):
            log.info("  PubMed: %d records harvested", total)

        if max_records and total >= max_records:
            break

        # Check for resumption token
        token_elem = root.find(".//oai:resumptionToken", PUBMED_NS)
        if token_elem is not None and token_elem.text:
            resumption_token = token_elem.text
            time.sleep(1)  # Be polite
        else:
            break

    log.info("PubMed harvest complete: %d records", total)
    return total


# ------------------------------------------------------------------ #
# Source 2: CORE (API, free tier 10K/day)                             #
# ------------------------------------------------------------------ #

CORE_API_BASE = "https://api.core.ac.uk/v3"


def harvest_core(conn, api_key=None, max_records=50000):
    """Harvest CORE via search API (requires free API key)."""
    if api_key is None:
        api_key = os.environ.get("CORE_API_KEY", "")
    if not api_key:
        log.warning(
            "No CORE API key. Get one free at https://core.ac.uk/services/api "
            "and set CORE_API_KEY env var."
        )
        return 0

    cur = conn.cursor()
    total = 0

    # Search broad topics to get diverse papers
    queries = [
        "machine learning",
        "artificial intelligence",
        "natural language processing",
        "computer vision",
        "distributed systems",
        "quantum computing",
        "bioinformatics",
        "climate change",
        "neuroscience",
        "ethics",
        "cognitive science",
        "mathematics",
        "physics",
        "chemistry",
        "economics",
    ]

    for query in queries:
        offset = 0
        while offset < max_records // len(queries):
            url = (
                f"{CORE_API_BASE}/search/works?"
                f"q={urllib.request.quote(query)}"
                f"&limit=100&offset={offset}"
            )
            headers = {
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Atlas-AI/1.0 (research)",
            }

            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
            except Exception as e:
                log.warning("CORE request failed: %s", e)
                time.sleep(5)
                break

            results = data.get("results", [])
            if not results:
                break

            batch = []
            for paper in results:
                title = (paper.get("title") or "").strip()
                if not title:
                    continue

                abstract = (paper.get("abstract") or "").strip()
                authors_list = paper.get("authors", [])
                authors = "; ".join(
                    a.get("name", "") for a in authors_list if isinstance(a, dict)
                )
                year = str(paper.get("yearPublished", ""))
                doi = paper.get("doi", "")
                core_id = str(paper.get("id", ""))

                paper_id = hashlib.md5(f"core:{core_id}".encode()).hexdigest()

                batch.append((
                    paper_id, "core", core_id, title, abstract[:5000],
                    authors[:1000], year, None, doi, query, "en",
                    f"{title} {abstract}",
                ))

            if batch:
                execute_batch(
                    cur,
                    "INSERT INTO academic_papers "
                    "(id, source, external_id, title, abstract, authors, year, "
                    "journal, doi, categories, language, tsv) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                    "to_tsvector('english', %s)) "
                    "ON CONFLICT (id) DO NOTHING",
                    batch,
                    page_size=100,
                )
                conn.commit()
                total += len(batch)

            offset += 100
            time.sleep(1)  # Rate limit

            if total % 5000 < 100:
                log.info("  CORE: %d records harvested", total)

    log.info("CORE harvest complete: %d records", total)
    return total


# ------------------------------------------------------------------ #
# Source 3: DOAJ (free API, ~21K journals)                            #
# ------------------------------------------------------------------ #

DOAJ_API_BASE = "https://doaj.org/api"


def harvest_doaj(conn, max_records=50000):
    """Harvest DOAJ journal + article metadata."""
    cur = conn.cursor()
    total = 0
    page = 1

    log.info("Harvesting DOAJ...")

    while total < max_records:
        url = f"{DOAJ_API_BASE}/search/articles/*?page={page}&pageSize=100"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Atlas-AI/1.0 (research)"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            log.warning("DOAJ request failed: %s", e)
            time.sleep(5)
            break

        results = data.get("results", [])
        if not results:
            break

        batch = []
        for article in results:
            bibjson = article.get("bibjson", {})
            title = bibjson.get("title", "").strip()
            if not title:
                continue

            abstract = bibjson.get("abstract", "").strip()
            authors_list = bibjson.get("author", [])
            authors = "; ".join(a.get("name", "") for a in authors_list)
            year = bibjson.get("year", "")
            journal_info = bibjson.get("journal", {})
            journal = journal_info.get("title", "")

            identifiers = bibjson.get("identifier", [])
            doi = ""
            for ident in identifiers:
                if ident.get("type") == "doi":
                    doi = ident.get("id", "")

            doaj_id = article.get("id", "")
            paper_id = hashlib.md5(f"doaj:{doaj_id}".encode()).hexdigest()
            subjects = ", ".join(
                s.get("term", "") for s in bibjson.get("subject", [])
            )

            batch.append((
                paper_id, "doaj", doaj_id, title, abstract[:5000],
                authors[:1000], year, journal, doi,
                subjects[:500], "en",
                f"{title} {abstract}",
            ))

        if batch:
            execute_batch(
                cur,
                "INSERT INTO academic_papers "
                "(id, source, external_id, title, abstract, authors, year, "
                "journal, doi, categories, language, tsv) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "to_tsvector('english', %s)) "
                "ON CONFLICT (id) DO NOTHING",
                batch,
                page_size=100,
            )
            conn.commit()
            total += len(batch)

        page += 1
        time.sleep(0.5)

        if total % 5000 < 100:
            log.info("  DOAJ: %d records harvested", total)

    log.info("DOAJ harvest complete: %d records", total)
    return total


# ------------------------------------------------------------------ #
# Embedding pass (separate from harvesting)                           #
# ------------------------------------------------------------------ #

def embed_unharvested(conn):
    """Embed rows in academic_papers that don't have embeddings yet."""
    cur = conn.cursor()

    cur.execute(
        "SELECT count(*) FROM academic_papers WHERE embedding IS NULL"
    )
    remaining = cur.fetchone()[0]
    if remaining == 0:
        log.info("All academic papers already embedded")
        return

    log.info("Embedding %d academic papers...", remaining)

    # Load model
    from sentence_transformers import SentenceTransformer
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    log.info("BGE-M3 loaded on %s", device)

    pca_components = None
    pca_mean = None
    if os.path.exists(PCA_PATH):
        with open(PCA_PATH, "rb") as f:
            pca = pickle.load(f)
        pca_components = pca["components"].T.astype(np.float32)
        pca_mean = pca["mean"].astype(np.float32)

    embedded = 0
    t0 = time.time()

    while True:
        cur.execute(
            "SELECT id, title, abstract FROM academic_papers "
            "WHERE embedding IS NULL ORDER BY id LIMIT %s",
            (BATCH_SIZE,),
        )
        rows = cur.fetchall()
        if not rows:
            break

        ids = [r[0] for r in rows]
        texts = [
            f"{r[1]}. {r[2] or ''}"[:1500] for r in rows
        ]

        embs = model.encode(texts, normalize_embeddings=True, batch_size=BATCH_SIZE)
        embs = np.array(embs, dtype=np.float32)

        pca_embs = None
        if pca_components is not None:
            centered = embs - pca_mean
            projected = centered @ pca_components
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            pca_embs = projected / (norms + 1e-10)

        updates = []
        for j in range(len(ids)):
            emb_str = str(embs[j].tolist())
            pca_str = str(pca_embs[j].tolist()) if pca_embs is not None else None
            updates.append((emb_str, pca_str, ids[j]))

        execute_batch(
            cur,
            "UPDATE academic_papers SET embedding = %s::vector, "
            "embedding_pca384 = %s::vector WHERE id = %s",
            updates,
            page_size=50,
        )
        conn.commit()

        embedded += len(rows)
        elapsed = time.time() - t0
        rate = embedded / elapsed if elapsed > 0 else 0
        eta = (remaining - embedded) / rate if rate > 0 else 0

        if embedded % 10000 < BATCH_SIZE:
            log.info(
                "  Embedded %d/%d (%.1f%%) @ %.1f vec/s, ETA %.0f min",
                embedded, remaining, embedded / remaining * 100,
                rate, eta / 60,
            )

    log.info("Embedding complete: %d papers in %.1f min", embedded, (time.time() - t0) / 60)

    # Create indexes if enough data
    cur.execute("SELECT count(*) FROM academic_papers WHERE embedding_pca384 IS NOT NULL")
    pca_count = cur.fetchone()[0]
    if pca_count > 1000:
        n_lists = min(2000, max(50, pca_count // 2000))
        log.info("Creating IVFFlat index (%d lists)...", n_lists)
        cur.execute("SET maintenance_work_mem = '512MB'")
        cur.execute("DROP INDEX IF EXISTS idx_academic_pca384")
        cur.execute(
            "CREATE INDEX idx_academic_pca384 ON academic_papers "
            "USING ivfflat (embedding_pca384 vector_cosine_ops) "
            "WITH (lists = %d)" % n_lists
        )
        cur.execute("DROP INDEX IF EXISTS idx_academic_tsv")
        cur.execute(
            "CREATE INDEX idx_academic_tsv ON academic_papers USING gin(tsv)"
        )
        conn.commit()
        log.info("Indexes created")


# ------------------------------------------------------------------ #
# Status                                                              #
# ------------------------------------------------------------------ #

def show_status():
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    try:
        cur.execute("SELECT count(*) FROM academic_papers")
        total = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM academic_papers WHERE embedding IS NOT NULL")
        embedded = cur.fetchone()[0]
        cur.execute(
            "SELECT source, count(*) FROM academic_papers GROUP BY source ORDER BY count(*) DESC"
        )
        by_source = cur.fetchall()

        print("\n=== Academic Papers Harvest Status ===")
        print("Total harvested: %d" % total)
        print("Embedded:        %d (%.0f%%)" % (embedded, embedded / total * 100 if total else 0))
        print("\nBy source:")
        for source, count in by_source:
            print("  %-15s %d" % (source, count))
    except psycopg2.errors.UndefinedTable:
        print("academic_papers table not created yet")
        conn.rollback()

    conn.close()


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Harvest academic sources")
    parser.add_argument(
        "--source",
        choices=["pubmed", "core", "doaj", "all"],
    )
    parser.add_argument("--embed", action="store_true", help="Embed un-embedded rows")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    conn = psycopg2.connect(DB_DSN)
    init_db(conn)

    if args.embed:
        embed_unharvested(conn)
        conn.close()
        return

    if args.source in ("pubmed", "all"):
        harvest_pubmed(conn, max_records=args.max_records)

    if args.source in ("core", "all"):
        harvest_core(conn, max_records=args.max_records)

    if args.source in ("doaj", "all"):
        harvest_doaj(conn, max_records=args.max_records)

    show_status()
    conn.close()


if __name__ == "__main__":
    main()
