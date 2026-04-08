#!/usr/bin/env python3
"""Patch atlas-rag-server.py to use PCA-384 embeddings for search.

This switches the RAG server from searching the full 1024-dim embedding
column to the PCA-projected 384-dim column, which:
  - Uses an existing IVFFlat index (176 MB vs 1.67 GB)
  - Searches 2.7x fewer dimensions per vector
  - Maintains >0.96 cosine similarity vs full-dim

The patch:
  1. Loads the PCA rotation matrix at startup
  2. Projects query embeddings from 1024 -> 384 dims
  3. Searches embedding_pca384 instead of embedding

Deploy:
    scp patch_rag_pca384.py atlas:/home/claude/agi-hpc/scripts/
    ssh atlas 'python3 /home/claude/agi-hpc/scripts/patch_rag_pca384.py'
    ssh atlas 'tmux send-keys -t rag C-c; sleep 1; tmux send-keys -t rag "python3 /home/claude/atlas-rag-server.py 2>&1 | tee /tmp/rag_server.log" Enter'
"""

from __future__ import annotations

import os
import pickle
import re
import shutil
import sys

RAG_SERVER = "/home/claude/atlas-rag-server.py"
PCA_MODEL = "/home/claude/agi-hpc/data/pca_rotation_384.pkl"
BACKUP_SUFFIX = ".bak.pre_pca384"


def patch():
    if not os.path.exists(RAG_SERVER):
        print(f"ERROR: {RAG_SERVER} not found")
        sys.exit(1)

    if not os.path.exists(PCA_MODEL):
        print(f"ERROR: PCA model not found at {PCA_MODEL}")
        print("Run: python3 /home/claude/agi-hpc/scripts/tqpro_migrate.py --fit-only")
        sys.exit(1)

    # Verify PCA model is valid
    with open(PCA_MODEL, "rb") as f:
        pca_data = pickle.load(f)
    print(
        f"PCA model: {pca_data['original_dim']}d -> {pca_data['n_components']}d, "
        f"{pca_data['variance_captured']:.1%} variance, "
        f"cosine={pca_data['mean_cosine']:.4f}"
    )

    with open(RAG_SERVER) as f:
        content = f.read()

    # Check if already patched
    if "embedding_pca384" in content:
        print("Already patched (embedding_pca384 found in server). Skipping.")
        return

    # Backup
    backup = RAG_SERVER + BACKUP_SUFFIX
    if not os.path.exists(backup):
        shutil.copy2(RAG_SERVER, backup)
        print(f"Backup: {backup}")

    # Patch 1: Add PCA imports and loading after the embed_model line
    pca_init = '''
# --- PCA-384 projection (TurboQuant Pro) ---
import pickle as _pickle
import numpy as _np
_PCA_PATH = "/home/claude/agi-hpc/data/pca_rotation_384.pkl"
with open(_PCA_PATH, "rb") as _f:
    _pca_data = _pickle.load(_f)
_pca_components = _pca_data["components"].T.astype(_np.float32)  # (1024, 384)
_pca_mean = _pca_data["mean"].astype(_np.float32)               # (1024,)
print(f"PCA-384 loaded: {_pca_data['variance_captured']:.1%} variance captured")

def pca_project(embedding):
    """Project 1024-dim embedding to PCA-384 space, L2-normalized."""
    centered = embedding.astype(_np.float32) - _pca_mean
    projected = centered @ _pca_components
    norm = _np.linalg.norm(projected)
    if norm > 1e-10:
        projected = projected / norm
    return projected
# --- end PCA-384 ---
'''

    # Insert after embed_model loading
    marker = 'embed_model = SentenceTransformer("BAAI/bge-m3", device="cpu")'
    if marker not in content:
        # Try alternate
        marker = "embed_model = SentenceTransformer("
        idx = content.find(marker)
        if idx < 0:
            print("ERROR: Could not find SentenceTransformer initialization")
            sys.exit(1)
        # Find end of that line
        end = content.index("\n", idx)
        marker = content[idx : end + 1]

    content = content.replace(marker, marker + "\n" + pca_init)

    # Patch 2: Update the search function to project queries and use pca384
    # Replace: q_emb = embed_model.encode(...)[0]
    #          emb_str = str(q_emb.tolist())
    # With:    q_emb = embed_model.encode(...)[0]
    #          q_emb_pca = pca_project(q_emb)
    #          emb_str = str(q_emb_pca.tolist())
    content = content.replace(
        "q_emb = embed_model.encode([query], normalize_embeddings=True)[0]\n"
        "    emb_str = str(q_emb.tolist())",
        "q_emb = embed_model.encode([query], normalize_embeddings=True)[0]\n"
        "    q_emb_pca = pca_project(q_emb)\n"
        "    emb_str = str(q_emb_pca.tolist())",
    )

    # Patch 3: Replace 'embedding <=>' with 'embedding_pca384 <=>'
    content = content.replace(
        "embedding <=> %s::vector",
        "embedding_pca384 <=> %s::vector",
    )
    content = content.replace(
        "(embedding <=>",
        "(embedding_pca384 <=>",
    )

    with open(RAG_SERVER, "w") as f:
        f.write(content)

    print(f"Patched {RAG_SERVER}:")
    print("  - Added PCA-384 projection at startup")
    print("  - Query embedding projected 1024 -> 384 dims")
    print("  - Search uses embedding_pca384 column + IVFFlat index")
    print(f"\nRestart RAG server to apply changes.")


if __name__ == "__main__":
    patch()
