#!/usr/bin/env python3
"""Index all available corpora into Atlas unified search.

Handles four source types:
  1. Git repos (code chunking + embedding)
  2. arXiv OAI-PMH XML (title + abstract embedding)
  3. Wikipedia XML dump (article chunking + embedding)
  4. Gutenberg plain text (book chunking + embedding)

Each corpus gets its own table with embedding + embedding_pca384 + tsv
columns, PCA-384 IVFFlat index, and GIN FTS index.

Run with --source to index a specific corpus:
    python index_all_corpora.py --source repos      # quick, ~10 min
    python index_all_corpora.py --source arxiv       # overnight, ~17h
    python index_all_corpora.py --source wikipedia   # ~7 days
    python index_all_corpora.py --source gutenberg   # ~3-4 days
    python index_all_corpora.py --source all         # everything
    python index_all_corpora.py --status             # show progress

Best run in tmux:
    tmux new-session -d -s index
    tmux send-keys -t index 'CUDA_VISIBLE_DEVICES=1 taskset -c 36-43 \\
        python index_all_corpora.py --source arxiv \\
        2>&1 | tee /tmp/index_arxiv.log' Enter
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("index-all")

DB_DSN = "dbname=atlas user=claude"
PCA_PATH = "/home/claude/agi-hpc/data/pca_rotation_384.pkl"
BATCH_SIZE = 128

# Repos not in the original 27
EXTRA_REPOS = [
    "/home/claude/tensor-3body",
    "/home/claude/turboquant-pro",
    "/home/claude/turboquant-experiments",
    "/home/claude/arc-agi-2",
    "/home/claude/source/aimo3-kaggle",
    "/home/claude/source/legal-retrieval",
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

INDEXABLE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".rb", ".sh", ".bash",
    ".md", ".txt", ".rst", ".yaml", ".yml", ".toml", ".json", ".ini",
    ".html", ".css", ".sql", ".r", ".jl", ".tex", ".bib", ".ipynb",
}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".eggs", "dist", "build", ".tox", ".mypy_cache", "site-packages",
}


class Embedder:
    """Shared embedding + PCA projection."""

    def __init__(self):
        log.info("Loading BGE-M3...")
        from sentence_transformers import SentenceTransformer
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        self.model = SentenceTransformer("BAAI/bge-m3", device=device)
        log.info("BGE-M3 loaded on %s", device)

        self.pca_components = None
        self.pca_mean = None
        if os.path.exists(PCA_PATH):
            with open(PCA_PATH, "rb") as f:
                pca = pickle.load(f)
            self.pca_components = pca["components"].T.astype(np.float32)
            self.pca_mean = pca["mean"].astype(np.float32)
            log.info("PCA-384 loaded (%.1f%% variance)", pca["variance_captured"] * 100)

    def embed_batch(self, texts):
        """Embed texts and return (embeddings_1024, embeddings_pca384)."""
        embs = self.model.encode(texts, normalize_embeddings=True, batch_size=BATCH_SIZE)
        embs = np.array(embs, dtype=np.float32)

        pca_embs = None
        if self.pca_components is not None:
            centered = embs - self.pca_mean
            projected = centered @ self.pca_components
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            pca_embs = projected / (norms + 1e-10)

        return embs, pca_embs


# ------------------------------------------------------------------ #
# Source 1: Extra repos                                               #
# ------------------------------------------------------------------ #

def index_repos(conn, embedder):
    """Index extra research repos into chunks table."""
    cur = conn.cursor()

    for repo_path in EXTRA_REPOS:
        repo_path = Path(repo_path)
        if not repo_path.exists():
            log.warning("Repo not found: %s", repo_path)
            continue

        repo_name = repo_path.name
        cur.execute("SELECT count(*) FROM chunks WHERE repo = %s", (repo_name,))
        existing = cur.fetchone()[0]
        if existing > 0:
            log.info("  %s: already has %d chunks, skipping", repo_name, existing)
            continue

        log.info("Indexing repo: %s", repo_name)
        chunks = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() not in INDEXABLE_EXTS:
                    continue
                if fpath.stat().st_size > 500_000:
                    continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if not text.strip():
                    continue

                rel_path = str(fpath.relative_to(repo_path))
                file_hash = hashlib.md5(text.encode()).hexdigest()

                for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk = text[i:i + CHUNK_SIZE]
                    if len(chunk.strip()) < 50:
                        continue
                    chunk_id = hashlib.md5(
                        f"{repo_name}/{rel_path}:{i}".encode()
                    ).hexdigest()
                    chunks.append({
                        "id": chunk_id,
                        "repo": repo_name,
                        "file_path": rel_path,
                        "offset": i,
                        "content": chunk,
                        "file_hash": file_hash,
                    })

        if not chunks:
            log.info("  No chunks found in %s", repo_name)
            continue

        log.info("  %d chunks, embedding...", len(chunks))
        texts = [c["content"] for c in chunks]

        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[batch_start:batch_start + BATCH_SIZE]
            batch_chunks = chunks[batch_start:batch_start + BATCH_SIZE]
            embs, pca_embs = embedder.embed_batch(batch_texts)

            values = []
            for j, c in enumerate(batch_chunks):
                emb_str = str(embs[j].tolist())
                pca_str = str(pca_embs[j].tolist()) if pca_embs is not None else None
                tsv_text = c["content"]
                values.append((
                    c["id"], c["repo"], c["file_path"], c["offset"],
                    c["content"], emb_str, c["file_hash"], pca_str, tsv_text,
                ))

            execute_batch(
                cur,
                "INSERT INTO chunks (id, repo, file_path, chunk_offset, content, "
                "embedding, file_hash, embedding_pca384, tsv) "
                "VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s::vector, "
                "to_tsvector('english', %s)) "
                "ON CONFLICT (id) DO NOTHING",
                values,
                page_size=50,
            )
            conn.commit()

            done = min(batch_start + BATCH_SIZE, len(texts))
            log.info("  %s: %d/%d chunks embedded", repo_name, done, len(texts))

        log.info("  %s: %d chunks indexed", repo_name, len(chunks))


# ------------------------------------------------------------------ #
# Source 2: arXiv                                                     #
# ------------------------------------------------------------------ #

def index_arxiv(conn, embedder):
    """Index arXiv CS metadata (title + abstract) from OAI-PMH XML."""
    cur = conn.cursor()

    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS arxiv_papers (
            id TEXT PRIMARY KEY,
            arxiv_id TEXT,
            title TEXT,
            abstract TEXT,
            categories TEXT,
            datestamp TEXT,
            embedding vector(1024),
            embedding_pca384 vector(384),
            tsv tsvector
        )
    """)
    conn.commit()

    cur.execute("SELECT count(*) FROM arxiv_papers")
    existing = cur.fetchone()[0]
    log.info("arXiv papers already indexed: %d", existing)

    xml_dir = Path("/archive/knowledge/arxiv/metadata/cs")
    xml_files = sorted(xml_dir.glob("*.xml"))
    log.info("Found %d OAI-PMH XML files", len(xml_files))

    ns = {"oai": "http://www.openarchives.org/OAI/2.0/",
          "dc": "http://purl.org/dc/elements/1.1/",
          "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/"}

    total_indexed = 0
    batch_texts = []
    batch_records = []

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except ET.ParseError:
            log.warning("Failed to parse %s", xml_file)
            continue

        for record in root.iter("{http://www.openarchives.org/OAI/2.0/}record"):
            header = record.find("{http://www.openarchives.org/OAI/2.0/}header")
            if header is None:
                continue

            identifier = header.findtext(
                "{http://www.openarchives.org/OAI/2.0/}identifier", ""
            )
            datestamp = header.findtext(
                "{http://www.openarchives.org/OAI/2.0/}datestamp", ""
            )
            categories = " ".join(
                s.text or "" for s in header.findall(
                    "{http://www.openarchives.org/OAI/2.0/}setSpec"
                )
            )

            metadata = record.find("{http://www.openarchives.org/OAI/2.0/}metadata")
            if metadata is None:
                continue

            dc = metadata.find("{http://www.openarchives.org/OAI/2.0/oai_dc/}dc")
            if dc is None:
                continue

            title = dc.findtext("{http://purl.org/dc/elements/1.1/}title", "")
            descriptions = dc.findall("{http://purl.org/dc/elements/1.1/}description")
            abstract = " ".join(d.text or "" for d in descriptions).strip()

            if not title:
                continue

            arxiv_id = identifier.replace("oai:arXiv.org:", "")
            paper_id = hashlib.md5(arxiv_id.encode()).hexdigest()

            embed_text = f"{title}. {abstract}"[:1500]
            batch_texts.append(embed_text)
            batch_records.append({
                "id": paper_id,
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract[:3000],
                "categories": categories,
                "datestamp": datestamp,
            })

            if len(batch_texts) >= BATCH_SIZE:
                _flush_arxiv_batch(cur, conn, embedder, batch_texts, batch_records)
                total_indexed += len(batch_texts)
                batch_texts = []
                batch_records = []

                if total_indexed % 10000 < BATCH_SIZE:
                    log.info("  arXiv: %d papers indexed", total_indexed)

        # Flush at end of file
        if batch_texts:
            _flush_arxiv_batch(cur, conn, embedder, batch_texts, batch_records)
            total_indexed += len(batch_texts)
            batch_texts = []
            batch_records = []

    log.info("arXiv indexing complete: %d papers", total_indexed)

    # Create indexes
    cur.execute("SELECT count(*) FROM arxiv_papers WHERE embedding_pca384 IS NOT NULL")
    pca_count = cur.fetchone()[0]
    if pca_count > 0:
        n_lists = min(1000, max(50, pca_count // 2000))
        log.info("Creating IVFFlat index on arxiv_papers (%d lists)...", n_lists)
        cur.execute("SET maintenance_work_mem = '512MB'")
        cur.execute("DROP INDEX IF EXISTS idx_arxiv_pca384")
        cur.execute(
            "CREATE INDEX idx_arxiv_pca384 ON arxiv_papers "
            "USING ivfflat (embedding_pca384 vector_cosine_ops) "
            "WITH (lists = %d)" % n_lists
        )
        cur.execute("DROP INDEX IF EXISTS idx_arxiv_tsv")
        cur.execute("CREATE INDEX idx_arxiv_tsv ON arxiv_papers USING gin(tsv)")
        conn.commit()
        log.info("Indexes created")


def _flush_arxiv_batch(cur, conn, embedder, texts, records):
    embs, pca_embs = embedder.embed_batch(texts)
    values = []
    for j, rec in enumerate(records):
        emb_str = str(embs[j].tolist())
        pca_str = str(pca_embs[j].tolist()) if pca_embs is not None else None
        tsv_text = f"{rec['title']} {rec['abstract']}"
        values.append((
            rec["id"], rec["arxiv_id"], rec["title"], rec["abstract"],
            rec["categories"], rec["datestamp"],
            emb_str, pca_str, tsv_text,
        ))
    execute_batch(
        cur,
        "INSERT INTO arxiv_papers (id, arxiv_id, title, abstract, categories, "
        "datestamp, embedding, embedding_pca384, tsv) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s::vector, "
        "to_tsvector('english', %s)) "
        "ON CONFLICT (id) DO NOTHING",
        values,
        page_size=50,
    )
    conn.commit()


# ------------------------------------------------------------------ #
# Source 3: Wikipedia                                                 #
# ------------------------------------------------------------------ #

def index_wikipedia(conn, embedder):
    """Index Wikipedia from bz2-compressed XML dump."""
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS wikipedia_chunks (
            id TEXT PRIMARY KEY,
            article_title TEXT,
            chunk_offset INT,
            content TEXT,
            embedding vector(1024),
            embedding_pca384 vector(384),
            tsv tsvector
        )
    """)
    conn.commit()

    cur.execute("SELECT count(*) FROM wikipedia_chunks")
    existing = cur.fetchone()[0]
    log.info("Wikipedia chunks already indexed: %d", existing)

    dump_path = "/archive/knowledge/wikipedia/enwiki-latest-pages-articles.xml.bz2"
    if not os.path.exists(dump_path):
        log.error("Wikipedia dump not found: %s", dump_path)
        return

    import bz2

    log.info("Streaming Wikipedia dump (24 GB compressed)...")
    total_articles = 0
    total_chunks = 0
    batch_texts = []
    batch_records = []

    # Stream parse the bz2 XML
    with bz2.open(dump_path, "rt", encoding="utf-8", errors="replace") as f:
        title = ""
        text_buf = []
        in_text = False
        in_title = False

        for line in f:
            if "<title>" in line:
                title = line.strip().replace("<title>", "").replace("</title>", "")
                in_title = False
            elif "<text" in line:
                in_text = True
                text_buf = [line.split(">", 1)[-1] if ">" in line else ""]
            elif "</text>" in line:
                in_text = False
                text_buf.append(line.split("</text>")[0])
                article_text = "".join(text_buf)

                # Skip redirects, stubs, and meta pages
                if (article_text.startswith("#REDIRECT") or
                    len(article_text) < 500 or
                    ":" in title):
                    continue

                # Clean wikitext (basic)
                clean = re.sub(r"\{\{[^}]*\}\}", "", article_text)
                clean = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", clean)
                clean = re.sub(r"<[^>]+>", "", clean)
                clean = re.sub(r"'{2,}", "", clean)
                clean = clean.strip()

                if len(clean) < 200:
                    continue

                # Chunk the article
                for i in range(0, len(clean), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk = clean[i:i + CHUNK_SIZE]
                    if len(chunk.strip()) < 100:
                        continue
                    chunk_id = hashlib.md5(
                        f"wiki:{title}:{i}".encode()
                    ).hexdigest()

                    batch_texts.append(chunk)
                    batch_records.append({
                        "id": chunk_id,
                        "title": title,
                        "offset": i,
                        "content": chunk,
                    })
                    total_chunks += 1

                total_articles += 1

                if len(batch_texts) >= BATCH_SIZE:
                    _flush_wiki_batch(cur, conn, embedder, batch_texts, batch_records)
                    batch_texts = []
                    batch_records = []

                    if total_articles % 10000 == 0:
                        log.info(
                            "  Wikipedia: %d articles, %d chunks",
                            total_articles, total_chunks,
                        )

            elif in_text:
                text_buf.append(line)

    # Final flush
    if batch_texts:
        _flush_wiki_batch(cur, conn, embedder, batch_texts, batch_records)

    log.info("Wikipedia complete: %d articles, %d chunks", total_articles, total_chunks)

    # Index
    cur.execute("SELECT count(*) FROM wikipedia_chunks WHERE embedding_pca384 IS NOT NULL")
    pca_count = cur.fetchone()[0]
    if pca_count > 1000:
        n_lists = min(2000, max(100, pca_count // 2000))
        log.info("Creating IVFFlat index (%d lists)...", n_lists)
        cur.execute("SET maintenance_work_mem = '1GB'")
        cur.execute("DROP INDEX IF EXISTS idx_wikipedia_pca384")
        cur.execute(
            "CREATE INDEX idx_wikipedia_pca384 ON wikipedia_chunks "
            "USING ivfflat (embedding_pca384 vector_cosine_ops) "
            "WITH (lists = %d)" % n_lists
        )
        cur.execute("DROP INDEX IF EXISTS idx_wikipedia_tsv")
        cur.execute("CREATE INDEX idx_wikipedia_tsv ON wikipedia_chunks USING gin(tsv)")
        conn.commit()


def _flush_wiki_batch(cur, conn, embedder, texts, records):
    embs, pca_embs = embedder.embed_batch(texts)
    values = []
    for j, rec in enumerate(records):
        emb_str = str(embs[j].tolist())
        pca_str = str(pca_embs[j].tolist()) if pca_embs is not None else None
        values.append((
            rec["id"], rec["title"], rec["offset"], rec["content"],
            emb_str, pca_str, rec["content"],
        ))
    execute_batch(
        cur,
        "INSERT INTO wikipedia_chunks (id, article_title, chunk_offset, content, "
        "embedding, embedding_pca384, tsv) "
        "VALUES (%s, %s, %s, %s, %s::vector, %s::vector, "
        "to_tsvector('english', %s)) "
        "ON CONFLICT (id) DO NOTHING",
        values,
        page_size=50,
    )
    conn.commit()


# ------------------------------------------------------------------ #
# Source 4: Gutenberg                                                 #
# ------------------------------------------------------------------ #

def index_gutenberg(conn, embedder):
    """Index Project Gutenberg plain text books."""
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS gutenberg_chunks (
            id TEXT PRIMARY KEY,
            book_id TEXT,
            book_title TEXT,
            chunk_offset INT,
            content TEXT,
            embedding vector(1024),
            embedding_pca384 vector(384),
            tsv tsvector
        )
    """)
    conn.commit()

    cur.execute("SELECT count(*) FROM gutenberg_chunks")
    existing = cur.fetchone()[0]
    log.info("Gutenberg chunks already indexed: %d", existing)

    gutenberg_dir = Path("/archive/knowledge/gutenberg")
    txt_files = sorted(gutenberg_dir.rglob("*.txt"))
    log.info("Found %d text files", len(txt_files))

    total_books = 0
    total_chunks = 0
    batch_texts = []
    batch_records = []

    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if len(text) < 1000:
            continue

        # Extract book ID from path
        book_id = txt_file.stem

        # Try to get title from first few lines
        lines = text[:2000].split("\n")
        title = book_id
        for line in lines:
            if line.strip().startswith("Title:"):
                title = line.strip().replace("Title:", "").strip()
                break

        # Chunk the book
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i:i + CHUNK_SIZE]
            if len(chunk.strip()) < 100:
                continue
            chunk_id = hashlib.md5(
                f"gut:{book_id}:{i}".encode()
            ).hexdigest()

            batch_texts.append(chunk)
            batch_records.append({
                "id": chunk_id,
                "book_id": book_id,
                "title": title[:500],
                "offset": i,
                "content": chunk,
            })
            total_chunks += 1

        total_books += 1

        if len(batch_texts) >= BATCH_SIZE:
            _flush_gutenberg_batch(cur, conn, embedder, batch_texts, batch_records)
            batch_texts = []
            batch_records = []

            if total_books % 1000 == 0:
                log.info(
                    "  Gutenberg: %d books, %d chunks",
                    total_books, total_chunks,
                )

    if batch_texts:
        _flush_gutenberg_batch(cur, conn, embedder, batch_texts, batch_records)

    log.info("Gutenberg complete: %d books, %d chunks", total_books, total_chunks)

    # Index
    cur.execute("SELECT count(*) FROM gutenberg_chunks WHERE embedding_pca384 IS NOT NULL")
    pca_count = cur.fetchone()[0]
    if pca_count > 1000:
        n_lists = min(2000, max(100, pca_count // 2000))
        log.info("Creating IVFFlat index (%d lists)...", n_lists)
        cur.execute("SET maintenance_work_mem = '1GB'")
        cur.execute("DROP INDEX IF EXISTS idx_gutenberg_pca384")
        cur.execute(
            "CREATE INDEX idx_gutenberg_pca384 ON gutenberg_chunks "
            "USING ivfflat (embedding_pca384 vector_cosine_ops) "
            "WITH (lists = %d)" % n_lists
        )
        cur.execute("DROP INDEX IF EXISTS idx_gutenberg_tsv")
        cur.execute("CREATE INDEX idx_gutenberg_tsv ON gutenberg_chunks USING gin(tsv)")
        conn.commit()


def _flush_gutenberg_batch(cur, conn, embedder, texts, records):
    embs, pca_embs = embedder.embed_batch(texts)
    values = []
    for j, rec in enumerate(records):
        emb_str = str(embs[j].tolist())
        pca_str = str(pca_embs[j].tolist()) if pca_embs is not None else None
        values.append((
            rec["id"], rec["book_id"], rec["title"], rec["offset"],
            rec["content"], emb_str, pca_str, rec["content"],
        ))
    execute_batch(
        cur,
        "INSERT INTO gutenberg_chunks (id, book_id, book_title, chunk_offset, "
        "content, embedding, embedding_pca384, tsv) "
        "VALUES (%s, %s, %s, %s, %s, %s::vector, %s::vector, "
        "to_tsvector('english', %s)) "
        "ON CONFLICT (id) DO NOTHING",
        values,
        page_size=50,
    )
    conn.commit()


# ------------------------------------------------------------------ #
# Status                                                              #
# ------------------------------------------------------------------ #

def show_status():
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    print("\n" + "=" * 60)
    print("  CORPUS INDEXING STATUS")
    print("=" * 60)

    tables = [
        ("chunks", "Code repos"),
        ("ethics_chunks", "Ethics corpus"),
        ("publications", "Publications"),
        ("arxiv_papers", "arXiv CS papers"),
        ("wikipedia_chunks", "Wikipedia"),
        ("gutenberg_chunks", "Gutenberg books"),
    ]

    total = 0
    for table, desc in tables:
        try:
            cur.execute("SELECT count(*) FROM %s" % table)
            n = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM %s WHERE embedding_pca384 IS NOT NULL" % table
            )
            pca = cur.fetchone()[0]
            total += pca
            print("  %-20s %10d rows  %10d searchable" % (desc, n, pca))
        except Exception:
            print("  %-20s (not created yet)" % desc)
            conn.rollback()

    print("  " + "-" * 50)
    print("  %-20s %10s      %10d searchable" % ("TOTAL", "", total))
    conn.close()


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Index all corpora")
    parser.add_argument(
        "--source",
        choices=["repos", "arxiv", "wikipedia", "gutenberg", "all"],
        default="all",
    )
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    conn = psycopg2.connect(DB_DSN)
    embedder = Embedder()

    if args.source in ("repos", "all"):
        log.info("=== Indexing extra repos ===")
        index_repos(conn, embedder)

    if args.source in ("arxiv", "all"):
        log.info("=== Indexing arXiv ===")
        index_arxiv(conn, embedder)

    if args.source in ("wikipedia", "all"):
        log.info("=== Indexing Wikipedia ===")
        index_wikipedia(conn, embedder)

    if args.source in ("gutenberg", "all"):
        log.info("=== Indexing Gutenberg ===")
        index_gutenberg(conn, embedder)

    show_status()
    conn.close()


if __name__ == "__main__":
    main()
