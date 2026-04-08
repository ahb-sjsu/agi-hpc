#!/usr/bin/env python3
"""Atlas RAG Indexer — indexes all ahb-sjsu repos into PostgreSQL + pgvector.

Walks all repos in /archive/ahb-sjsu/, chunks source files,
generates embeddings with BGE-M3, and stores in PostgreSQL with pgvector.

Usage:
    python3 atlas-rag-indexer.py          # full index
    python3 atlas-rag-indexer.py --update  # only new/changed files
"""

import os
import sys
import json
import hashlib
import argparse
import pickle
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

REPOS_DIR = Path("/archive/ahb-sjsu")
PCA_MODEL_PATH = "/home/claude/agi-hpc/data/pca_rotation_384.pkl"
DB_DSN = "dbname=atlas user=claude"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 64

INDEXABLE = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".rb", ".php", ".sh", ".bash", ".zsh",
    ".md", ".txt", ".rst", ".yaml", ".yml", ".toml", ".json", ".ini", ".cfg",
    ".html", ".css", ".sql", ".r", ".jl", ".lua", ".swift", ".kt",
    ".tex", ".bib", ".ipynb",
}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".eggs", "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    "site-packages",
}

MAX_FILE_SIZE = 500_000


def init_db(conn):
    """Create the chunks table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                repo TEXT NOT NULL,
                file_path TEXT NOT NULL,
                chunk_offset INT NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1024),
                file_hash TEXT
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        conn.commit()


def find_files(repos_dir):
    for repo_dir in sorted(repos_dir.iterdir()):
        if not repo_dir.is_dir() or repo_dir.name.startswith("."):
            continue
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in INDEXABLE:
                    try:
                        if fpath.stat().st_size < MAX_FILE_SIZE:
                            yield fpath
                    except OSError:
                        pass


def chunk_file(fpath, repos_dir):
    try:
        text = fpath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    repo = fpath.relative_to(repos_dir).parts[0]
    rel_path = str(fpath.relative_to(repos_dir / repo))

    if not text.strip():
        return []

    if fpath.suffix == ".ipynb":
        try:
            nb = json.loads(text)
            cells = nb.get("cells", [])
            text = "\n\n".join(
                "".join(c.get("source", []))
                for c in cells
                if c.get("cell_type") in ("code", "markdown")
            )
        except json.JSONDecodeError:
            return []

    file_hash = hashlib.md5(text.encode()).hexdigest()
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = text[i:i + CHUNK_SIZE]
        if len(chunk.strip()) < 50:
            continue
        chunk_id = hashlib.md5(f"{repo}/{rel_path}:{i}".encode()).hexdigest()
        chunks.append({
            "id": chunk_id,
            "repo": repo,
            "file_path": rel_path,
            "chunk_offset": i,
            "content": chunk,
            "file_hash": file_hash,
        })

    return chunks


def build_index(repos_dir, update_only=False):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    conn = psycopg2.connect(DB_DSN)
    init_db(conn)

    # Get existing file hashes if updating
    existing_hashes = {}
    if update_only:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT file_path, file_hash FROM chunks")
            existing_hashes = {row[0]: row[1] for row in cur.fetchall()}

    print("Scanning files...")
    all_files = list(find_files(repos_dir))
    print(f"  Found {len(all_files)} indexable files")

    # Chunk all files
    all_chunks = []
    skipped = 0
    for fpath in all_files:
        chunks = chunk_file(fpath, repos_dir)
        if not chunks:
            continue

        rel = chunks[0]["repo"] + "/" + chunks[0]["file_path"]
        file_hash = chunks[0]["file_hash"]

        if update_only and existing_hashes.get(rel) == file_hash:
            skipped += 1
            continue

        # Delete old chunks for this file if updating
        if update_only:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunks WHERE repo = %s AND file_path = %s",
                    (chunks[0]["repo"], chunks[0]["file_path"])
                )

        all_chunks.extend(chunks)

    if update_only:
        print(f"  Skipped {skipped} unchanged files")

    if not all_chunks:
        print("  No new chunks to index!")
        conn.close()
        return

    print(f"  {len(all_chunks)} chunks to embed")

    # Generate embeddings
    print("Loading embedding model (BGE-M3) on CPU...")
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")

    print(f"Generating embeddings...")
    texts = [c["content"] for c in all_chunks]

    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embs = model.encode(batch, normalize_embeddings=True)
        embeddings.extend(embs)
        done = min(i + BATCH_SIZE, len(texts))
        print(f"  {done}/{len(texts)} chunks embedded", end="\r")

    print()

    # PCA-384 projection
    pca_components = None
    pca_mean = None
    if os.path.exists(PCA_MODEL_PATH):
        with open(PCA_MODEL_PATH, "rb") as f:
            pca_data = pickle.load(f)
        pca_components = pca_data["components"].T.astype(np.float32)  # (1024, 384)
        pca_mean = pca_data["mean"].astype(np.float32)
        print(f"PCA-384 loaded ({pca_data['variance_captured']:.1%} variance)")
    else:
        print("WARNING: No PCA model found, skipping embedding_pca384")

    # Insert into PostgreSQL
    print("Inserting into PostgreSQL...")
    embeddings_arr = np.array(embeddings, dtype=np.float32)

    # Compute PCA-384 projections in one batch
    pca_embeddings = None
    if pca_components is not None:
        centered = embeddings_arr - pca_mean
        projected = centered @ pca_components
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        pca_embeddings = projected / (norms + 1e-10)
        print(f"  PCA-384 projections computed for {len(pca_embeddings)} chunks")

    with conn.cursor() as cur:
        values = []
        for i, (chunk, emb) in enumerate(zip(all_chunks, embeddings)):
            emb_list = emb.tolist()
            pca_str = str(pca_embeddings[i].tolist()) if pca_embeddings is not None else None
            values.append((
                chunk["id"],
                chunk["repo"],
                chunk["file_path"],
                chunk["chunk_offset"],
                chunk["content"],
                str(emb_list),
                chunk["file_hash"],
                pca_str,
            ))

        if pca_embeddings is not None:
            execute_values(
                cur,
                """INSERT INTO chunks (id, repo, file_path, chunk_offset, content, embedding, file_hash, embedding_pca384)
                   VALUES %s
                   ON CONFLICT (id) DO UPDATE SET
                       content = EXCLUDED.content,
                       embedding = EXCLUDED.embedding,
                       file_hash = EXCLUDED.file_hash,
                       embedding_pca384 = EXCLUDED.embedding_pca384""",
                values,
                template="(%s, %s, %s, %s, %s, %s::vector, %s, %s::vector)",
            )
        else:
            # Fallback: no PCA column
            values_no_pca = [v[:7] for v in values]
            execute_values(
                cur,
                """INSERT INTO chunks (id, repo, file_path, chunk_offset, content, embedding, file_hash)
                   VALUES %s
                   ON CONFLICT (id) DO UPDATE SET
                       content = EXCLUDED.content,
                       embedding = EXCLUDED.embedding,
                       file_hash = EXCLUDED.file_hash""",
                values_no_pca,
                template="(%s, %s, %s, %s, %s, %s::vector, %s)",
            )
        conn.commit()

    # Stats
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT repo) FROM chunks")
        repos = cur.fetchone()[0]

    print(f"\nIndex complete:")
    print(f"  Total chunks: {total}")
    print(f"  Repos indexed: {repos}")
    print(f"  New chunks added: {len(all_chunks)}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    build_index(REPOS_DIR, update_only=args.update)
