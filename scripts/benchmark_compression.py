#!/usr/bin/env python3
"""
Benchmark 6 vector compression methods on real BGE-M3 1024-dim embeddings.

Loads embeddings from PostgreSQL (chunks + ethics_chunks tables),
benchmarks compression ratio, quality, search accuracy, throughput,
and memory projections for each method and key combinations.

Results saved to /home/claude/agi-hpc/benchmarks/compression_results.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import logging
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import psycopg2

# Add TurboQuant to path
sys.path.insert(0, "/home/claude/agi-hpc/src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Thread limits -- set BEFORE any numpy/faiss import uses them
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"


# ------------------------------------------------------------------ #
# Temperature monitoring                                              #
# ------------------------------------------------------------------ #

def get_max_cpu_temp() -> float:
    """Read max CPU package temp from sensors."""
    try:
        out = subprocess.check_output(["sensors"], text=True, timeout=5)
        temps: list[float] = []
        for line in out.splitlines():
            if "Package" in line or "Core" in line:
                parts = line.split("+")
                if len(parts) >= 2:
                    t = parts[1].split("\u00b0")[0].split("C")[0].strip()
                    try:
                        temps.append(float(t))
                    except ValueError:
                        pass
        return max(temps) if temps else 0.0
    except Exception:
        return 0.0


def thermal_check(label: str = "") -> None:
    """Pause if CPU temp exceeds 78C."""
    t = get_max_cpu_temp()
    if t > 78:
        logger.warning(
            "CPU temp %.1fC > 78C after %s -- cooling 30s", t, label
        )
        time.sleep(30)
    elif t > 70:
        logger.info("CPU temp %.1fC after %s", t, label)


# ------------------------------------------------------------------ #
# Data loading from PostgreSQL                                        #
# ------------------------------------------------------------------ #

def load_embeddings(
    table: str,
    n: int,
    conn_str: str = "dbname=atlas user=claude",
) -> np.ndarray:
    """Load n embeddings from the given table as float32 array."""
    logger.info("Loading %d embeddings from %s ...", n, table)
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()

    if table == "ethics_chunks":
        cur.execute(
            "SELECT count(*) FROM ethics_chunks WHERE embedding IS NOT NULL"
        )
        total = cur.fetchone()[0]
        if total == 0:
            raise RuntimeError("No embeddings in ethics_chunks yet")
        pct = min(100.0, (n * 1.5 / total) * 100.0)
        cur.execute(
            f"SELECT embedding::text FROM ethics_chunks "
            f"TABLESAMPLE SYSTEM ({pct}) "
            f"WHERE embedding IS NOT NULL "
            f"LIMIT {n}"
        )
    else:
        cur.execute(
            f"SELECT embedding::text FROM {table} "
            f"WHERE embedding IS NOT NULL "
            f"ORDER BY random() LIMIT {n}"
        )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        raise RuntimeError(f"No embeddings found in {table}")

    vecs = []
    for (emb_str,) in rows:
        vals = emb_str.strip("[]").split(",")
        vecs.append([float(v) for v in vals])

    arr = np.array(vecs, dtype=np.float32)
    logger.info("Loaded %d embeddings, shape %s", len(arr), arr.shape)
    return arr


# ------------------------------------------------------------------ #
# Compression methods                                                 #
# ------------------------------------------------------------------ #

class ScalarQuantInt8:
    """Min-max scalar quantization to uint8 per dimension."""

    def __init__(self) -> None:
        self.mins: np.ndarray | None = None
        self.scales: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> ScalarQuantInt8:
        self.mins = X.min(axis=0).astype(np.float32)
        maxs = X.max(axis=0).astype(np.float32)
        self.scales = maxs - self.mins
        self.scales[self.scales == 0] = 1.0  # type: ignore[index]
        return self

    def compress(self, X: np.ndarray) -> np.ndarray:
        normalized = (X - self.mins) / self.scales
        return np.clip(normalized * 255, 0, 255).astype(np.uint8)

    def decompress(self, X_q: np.ndarray) -> np.ndarray:
        return (X_q.astype(np.float32) / 255.0) * self.scales + self.mins

    def compressed_bytes(self, n: int, d: int) -> int:
        return n * d + 2 * d * 4  # codes + mins + scales


class BinaryQuant:
    """Sign-bit binary quantization."""

    def compress(self, X: np.ndarray) -> np.ndarray:
        return np.packbits((X >= 0).astype(np.uint8), axis=1)

    def decompress(self, X_b: np.ndarray, d: int = 1024) -> np.ndarray:
        bits = np.unpackbits(X_b, axis=1)[:, :d]
        return bits.astype(np.float32) * 2 - 1

    def compressed_bytes(self, n: int, d: int) -> int:
        return n * (d // 8)


class ProductQuantizer:
    """Product Quantization using faiss."""

    def __init__(self, d: int = 1024, M: int = 16, nbits: int = 8) -> None:
        self.d = d
        self.M = M
        self.nbits = nbits
        self.K = 2**nbits
        self.dsub = d // M
        self.pq: Any = None
        self._use_faiss = False

    def fit(self, X: np.ndarray) -> ProductQuantizer:
        try:
            import faiss

            self.pq = faiss.ProductQuantizer(self.d, self.M, self.nbits)
            self.pq.train(np.ascontiguousarray(X.astype(np.float32)))
            self._use_faiss = True
        except ImportError:
            from sklearn.cluster import MiniBatchKMeans

            self._use_faiss = False
            self.centroids = np.zeros(
                (self.M, self.K, self.dsub), dtype=np.float32
            )
            for m in range(self.M):
                sub = X[:, m * self.dsub : (m + 1) * self.dsub].astype(
                    np.float32
                )
                km = MiniBatchKMeans(
                    n_clusters=self.K, max_iter=20, batch_size=1000, n_init=1
                )
                km.fit(sub)
                self.centroids[m] = km.cluster_centers_
        return self

    def compress(self, X: np.ndarray) -> np.ndarray:
        if self._use_faiss:
            return self.pq.compute_codes(
                np.ascontiguousarray(X.astype(np.float32))
            )
        n = len(X)
        codes = np.zeros((n, self.M), dtype=np.uint8)
        for m in range(self.M):
            sub = X[:, m * self.dsub : (m + 1) * self.dsub].astype(
                np.float32
            )
            batch_sz = 2000
            for i in range(0, n, batch_sz):
                batch = sub[i : i + batch_sz]
                dists = np.sum(
                    (batch[:, None, :] - self.centroids[m][None, :, :]) ** 2,
                    axis=2,
                )
                codes[i : i + batch_sz, m] = np.argmin(dists, axis=1).astype(
                    np.uint8
                )
        return codes

    def decompress(self, codes: np.ndarray) -> np.ndarray:
        if self._use_faiss:
            return self.pq.decode(codes)
        n = len(codes)
        X_hat = np.zeros((n, self.d), dtype=np.float32)
        for m in range(self.M):
            X_hat[:, m * self.dsub : (m + 1) * self.dsub] = self.centroids[
                m
            ][codes[:, m]]
        return X_hat

    def compressed_bytes(self, n: int, d: int) -> int:
        return n * self.M + self.M * self.K * self.dsub * 4


class TurboQuantWrapper:
    """Wrapper around TurboQuantKV for embedding compression."""

    def __init__(self, dim: int = 1024, bits: int = 3) -> None:
        from agi.meta.llm.turboquant_kv import TurboQuantKV

        self.dim = dim
        self.bits = bits
        self.tq = TurboQuantKV(
            head_dim=dim,
            n_heads=1,
            bits=bits,
            use_gpu=False,
            seed=42,
        )

    def compress(self, X: np.ndarray) -> Any:
        tensor = X.reshape(1, 1, X.shape[0], X.shape[1])
        return self.tq.compress(tensor, packed=True)

    def decompress(self, compressed: Any) -> np.ndarray:
        result = self.tq.decompress(compressed)
        return result.reshape(-1, self.dim)

    def compressed_bytes(self, n: int, d: int) -> int:
        return (n * d * self.bits + 7) // 8 + n * 4


class MatryoshkaTruncation:
    """Truncate embedding to lower dimension (Matryoshka property)."""

    def __init__(self, target_dim: int = 256) -> None:
        self.target_dim = target_dim

    def compress(self, X: np.ndarray) -> np.ndarray:
        return X[:, : self.target_dim].copy()

    def decompress(
        self, X_t: np.ndarray, original_dim: int = 1024
    ) -> np.ndarray:
        out = np.zeros((len(X_t), original_dim), dtype=np.float32)
        out[:, : self.target_dim] = X_t
        return out

    def compressed_bytes(self, n: int, d: int) -> int:
        return n * self.target_dim * 4


class CombinedMethod:
    """Apply Matryoshka truncation then another compression method."""

    def __init__(
        self, truncate_dim: int, compressor: Any, name: str = ""
    ) -> None:
        self.truncate_dim = truncate_dim
        self.compressor = compressor
        self.name = name

    def compress(self, X: np.ndarray) -> Tuple[Any, int]:
        X_t = X[:, : self.truncate_dim].copy()
        compressed = self.compressor.compress(X_t)
        return (compressed, self.truncate_dim)

    def decompress(
        self, data: Tuple[Any, int], original_dim: int = 1024
    ) -> np.ndarray:
        compressed, tdim = data
        X_t = self.compressor.decompress(compressed)
        if hasattr(X_t, "shape") and len(X_t.shape) == 2:
            if X_t.shape[1] < original_dim:
                out = np.zeros(
                    (len(X_t), original_dim), dtype=np.float32
                )
                out[:, : X_t.shape[1]] = X_t
                return out
            return X_t
        return X_t

    def compressed_bytes(self, n: int, d: int) -> int:
        return self.compressor.compressed_bytes(n, self.truncate_dim)


# ------------------------------------------------------------------ #
# Metrics                                                             #
# ------------------------------------------------------------------ #

def cosine_similarity_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between corresponding rows."""
    dot = np.sum(A * B, axis=1)
    normA = np.linalg.norm(A, axis=1)
    normB = np.linalg.norm(B, axis=1)
    return dot / (normA * normB + 1e-30)


def recall_at_k(
    X_orig: np.ndarray,
    X_comp: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
) -> float:
    """Recall@k: fraction of true top-k neighbors found via compressed."""
    n_queries = len(queries)
    total_recall = 0.0
    for i in range(n_queries):
        q = queries[i : i + 1]
        sims_orig = (X_orig @ q.T).ravel()
        true_topk = set(np.argsort(sims_orig)[-k:])
        sims_comp = (X_comp @ q.T).ravel()
        comp_topk = set(np.argsort(sims_comp)[-k:])
        total_recall += len(true_topk & comp_topk) / k
    return total_recall / n_queries


def recall_at_k_binary(
    X_orig: np.ndarray,
    X_binary: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
) -> float:
    """Recall@k using hamming distance for binary codes."""
    n_queries = len(queries)
    total_recall = 0.0
    for i in range(n_queries):
        q = queries[i : i + 1]
        sims_orig = (X_orig @ q.T).ravel()
        true_topk = set(np.argsort(sims_orig)[-k:])

        q_bin = np.packbits((q >= 0).astype(np.uint8), axis=1)
        xor = np.bitwise_xor(X_binary, q_bin)
        # Vectorized popcount via lookup table
        lut = np.array([bin(x).count("1") for x in range(256)], dtype=np.int32)
        hamming = lut[xor].sum(axis=1)

        comp_topk = set(np.argsort(hamming)[:k])
        total_recall += len(true_topk & comp_topk) / k
    return total_recall / n_queries


def recall_at_k_matryoshka(
    X_orig: np.ndarray,
    X_trunc: np.ndarray,
    queries_full: np.ndarray,
    k: int = 10,
) -> float:
    """Recall@k using truncated vectors for search."""
    n_queries = len(queries_full)
    total_recall = 0.0
    tdim = X_trunc.shape[1]
    for i in range(n_queries):
        q = queries_full[i : i + 1]
        sims_orig = (X_orig @ q.T).ravel()
        true_topk = set(np.argsort(sims_orig)[-k:])

        q_trunc = q[:, :tdim]
        sims_trunc = (X_trunc @ q_trunc.T).ravel()
        trunc_topk = set(np.argsort(sims_trunc)[-k:])
        total_recall += len(true_topk & trunc_topk) / k
    return total_recall / n_queries


# ------------------------------------------------------------------ #
# Benchmark runner                                                    #
# ------------------------------------------------------------------ #

def benchmark_method(
    name: str,
    X: np.ndarray,
    compressor: Any,
    queries: np.ndarray,
    is_binary: bool = False,
    is_matryoshka: bool = False,
    is_combined: bool = False,
) -> Dict[str, Any]:
    """Benchmark a single compression method."""
    logger.info("=== Benchmarking: %s ===", name)
    thermal_check(f"before {name}")

    n, d = X.shape
    original_bytes = n * d * 4

    # --- Compression ---
    t0 = time.perf_counter()
    compressed = compressor.compress(X)
    t_compress = time.perf_counter() - t0

    # --- Decompression ---
    t0 = time.perf_counter()
    if is_binary:
        X_hat = compressor.decompress(compressed, d)
    elif is_combined or is_matryoshka:
        X_hat = compressor.decompress(compressed, d)
    else:
        X_hat = compressor.decompress(compressed)
    t_decompress = time.perf_counter() - t0

    # --- Compressed size ---
    if isinstance(compressed, tuple):
        comp_bytes = compressor.compressed_bytes(n, d)
    elif hasattr(compressed, "nbytes"):
        if callable(compressed.nbytes):
            comp_bytes = compressed.nbytes()
        else:
            comp_bytes = int(compressed.nbytes)
    elif hasattr(compressor, "compressed_bytes"):
        comp_bytes = compressor.compressed_bytes(n, d)
    else:
        comp_bytes = original_bytes

    ratio = original_bytes / max(comp_bytes, 1)

    # --- Quality: cosine similarity ---
    # For methods that reconstruct to full dim, measure full cosine.
    # For truncation-based methods, also measure "retained dims cosine"
    # (which shows how well the kept dimensions are preserved).
    if X_hat.shape == X.shape:
        cosines = cosine_similarity_batch(X, X_hat)
    else:
        tdim = min(X_hat.shape[1], X.shape[1])
        cosines = cosine_similarity_batch(X[:, :tdim], X_hat[:, :tdim])
    mean_cosine = float(np.mean(cosines))
    min_cosine = float(np.min(cosines))
    p5_cosine = float(np.percentile(cosines, 5))

    # --- Search accuracy: recall@10 ---
    n_recall_queries = min(50, len(queries))
    q_subset = queries[:n_recall_queries]
    logger.info("  Computing recall@10 (%d queries) ...", n_recall_queries)

    if is_binary:
        recall = recall_at_k_binary(X, compressed, q_subset, k=10)
    elif is_matryoshka and not is_combined:
        recall = recall_at_k_matryoshka(X, compressed, q_subset, k=10)
    else:
        recall = recall_at_k(X, X_hat, q_subset, k=10)

    # --- Throughput ---
    batch_size = min(1000, n)
    X_batch = X[:batch_size]
    t0 = time.perf_counter()
    n_iters = 3
    for _ in range(n_iters):
        compressor.compress(X_batch)
    t_batch = (time.perf_counter() - t0) / n_iters
    compress_ops_per_sec = batch_size / max(t_batch, 1e-9)

    # Search throughput
    if is_binary:
        q_bin = np.packbits(
            (q_subset[:1] >= 0).astype(np.uint8), axis=1
        )
        t0 = time.perf_counter()
        for _ in range(10):
            np.bitwise_xor(compressed, q_bin)
        search_qps = 10 / max(time.perf_counter() - t0, 1e-9)
    elif is_matryoshka and not is_combined:
        q_t = q_subset[:1, : compressed.shape[1]]
        t0 = time.perf_counter()
        for _ in range(10):
            (compressed @ q_t.T).ravel()
        search_qps = 10 / max(time.perf_counter() - t0, 1e-9)
    else:
        t0 = time.perf_counter()
        for _ in range(10):
            (X_hat @ q_subset[:1].T).ravel()
        search_qps = 10 / max(time.perf_counter() - t0, 1e-9)

    # --- Memory projections ---
    bytes_per_vec = comp_bytes / max(n, 1)
    memory_proj = {
        "10k_vectors_mb": round(bytes_per_vec * 10_000 / 1e6, 2),
        "100k_vectors_mb": round(bytes_per_vec * 100_000 / 1e6, 2),
        "1m_vectors_mb": round(bytes_per_vec * 1_000_000 / 1e6, 2),
        "2_4m_vectors_mb": round(bytes_per_vec * 2_400_000 / 1e6, 2),
    }

    thermal_check(f"after {name}")

    result = {
        "method": name,
        "compression_ratio": round(ratio, 2),
        "original_bytes": original_bytes,
        "compressed_bytes": comp_bytes,
        "bytes_per_vector": round(bytes_per_vec, 1),
        "quality": {
            "mean_cosine": round(mean_cosine, 6),
            "min_cosine": round(min_cosine, 6),
            "p5_cosine": round(p5_cosine, 6),
        },
        "search": {
            "recall_at_10": round(recall, 4),
            "n_queries": n_recall_queries,
        },
        "throughput": {
            "compress_ops_per_sec": round(compress_ops_per_sec, 0),
            "decompress_time_sec": round(t_decompress, 4),
            "search_qps": round(search_qps, 0),
        },
        "memory_projection": memory_proj,
        "timing": {
            "compress_sec": round(t_compress, 4),
            "decompress_sec": round(t_decompress, 4),
        },
    }

    logger.info(
        "  Ratio: %.2fx | Cosine: %.6f | Recall@10: %.4f | Comp: %.0f v/s",
        ratio,
        mean_cosine,
        recall,
        compress_ops_per_sec,
    )
    return result


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    logger.info("=" * 70)
    logger.info("VECTOR COMPRESSION BENCHMARK")
    logger.info("=" * 70)

    thermal_check("startup")

    # --- Load data ---
    X_chunks = load_embeddings("chunks", 10_000)
    thermal_check("after loading chunks")

    try:
        X_ethics: np.ndarray | None = load_embeddings(
            "ethics_chunks", 100_000
        )
    except Exception as e:
        logger.warning("Could not load ethics_chunks: %s", e)
        X_ethics = None

    thermal_check("after loading ethics")

    X = X_chunks
    n, d = X.shape
    logger.info("Primary dataset: %d vectors, %d dimensions", n, d)

    rng = np.random.default_rng(42)
    n_queries = min(200, n)
    query_idx = rng.choice(n, n_queries, replace=False)
    queries = X[query_idx].copy()

    results: List[Dict[str, Any]] = []

    # ============================================================ #
    # 1. TurboQuant (2-bit, 3-bit, 4-bit)                         #
    # ============================================================ #
    for bits in [2, 3, 4]:
        name = f"TurboQuant {bits}-bit"
        try:
            tq = TurboQuantWrapper(dim=d, bits=bits)
            r = benchmark_method(name, X, tq, queries)
            results.append(r)
        except Exception as e:
            logger.error("FAILED %s: %s", name, e)
            import traceback

            traceback.print_exc()

    # ============================================================ #
    # 2. Product Quantization (M=8, 16, 32)                        #
    # ============================================================ #
    for M in [8, 16, 32]:
        name = f"PQ M={M} K=256"
        try:
            pq = ProductQuantizer(d=d, M=M, nbits=8)
            logger.info("Training PQ M=%d ...", M)
            train_n = min(5000, n)
            pq.fit(X[:train_n])
            r = benchmark_method(name, X, pq, queries)
            results.append(r)
        except Exception as e:
            logger.error("FAILED %s: %s", name, e)
            import traceback

            traceback.print_exc()
        thermal_check(f"after PQ M={M}")

    # ============================================================ #
    # 3. Binary Quantization                                        #
    # ============================================================ #
    try:
        bq = BinaryQuant()
        r = benchmark_method(
            "Binary Quantization", X, bq, queries, is_binary=True
        )
        results.append(r)
    except Exception as e:
        logger.error("FAILED Binary: %s", e)
        import traceback

        traceback.print_exc()

    # ============================================================ #
    # 4. Scalar Quantization (int8)                                 #
    # ============================================================ #
    try:
        sq = ScalarQuantInt8()
        sq.fit(X)
        r = benchmark_method("Scalar int8", X, sq, queries)
        results.append(r)
    except Exception as e:
        logger.error("FAILED Scalar: %s", e)
        import traceback

        traceback.print_exc()

    # ============================================================ #
    # 5. Matryoshka Truncation (512, 256, 128)                      #
    # ============================================================ #
    for tdim in [512, 256, 128]:
        name = f"Matryoshka {tdim}d"
        try:
            mt = MatryoshkaTruncation(target_dim=tdim)
            r = benchmark_method(name, X, mt, queries, is_matryoshka=True)
            results.append(r)
        except Exception as e:
            logger.error("FAILED %s: %s", name, e)
            import traceback

            traceback.print_exc()

    # ============================================================ #
    # 6. Combinations                                               #
    # ============================================================ #

    # Matryoshka 256 + TurboQuant 3-bit
    try:
        tq3 = TurboQuantWrapper(dim=256, bits=3)
        combo = CombinedMethod(256, tq3, "Mat256+TQ3")
        r = benchmark_method(
            "Matryoshka 256 + TurboQuant 3-bit",
            X,
            combo,
            queries,
            is_combined=True,
        )
        results.append(r)
    except Exception as e:
        logger.error("FAILED Mat256+TQ3: %s", e)
        import traceback

        traceback.print_exc()

    # Matryoshka 256 + PQ (M=8)
    try:
        pq_combo = ProductQuantizer(d=256, M=8, nbits=8)
        pq_combo.fit(X[:5000, :256])

        class PQComboWrap:
            def __init__(self, pq: Any) -> None:
                self.pq = pq

            def compress(self, X: np.ndarray) -> np.ndarray:
                return self.pq.compress(X)

            def decompress(self, codes: np.ndarray) -> np.ndarray:
                return self.pq.decompress(codes)

            def compressed_bytes(self, n: int, d: int) -> int:
                return self.pq.compressed_bytes(n, 256)

        combo2 = CombinedMethod(256, PQComboWrap(pq_combo), "Mat256+PQ8")
        r = benchmark_method(
            "Matryoshka 256 + PQ M=8",
            X,
            combo2,
            queries,
            is_combined=True,
        )
        results.append(r)
    except Exception as e:
        logger.error("FAILED Mat256+PQ8: %s", e)
        import traceback

        traceback.print_exc()

    # Matryoshka 512 + Binary
    try:

        class BinaryTruncWrap:
            def __init__(self, tdim: int) -> None:
                self.tdim = tdim
                self.bq = BinaryQuant()

            def compress(self, X: np.ndarray) -> np.ndarray:
                return self.bq.compress(X)

            def decompress(self, X_b: np.ndarray) -> np.ndarray:
                return self.bq.decompress(X_b, self.tdim)

            def compressed_bytes(self, n: int, d: int) -> int:
                return n * (self.tdim // 8)

        combo3 = CombinedMethod(
            512, BinaryTruncWrap(512), "Mat512+Bin"
        )
        r = benchmark_method(
            "Matryoshka 512 + Binary",
            X,
            combo3,
            queries,
            is_combined=True,
        )
        results.append(r)
    except Exception as e:
        logger.error("FAILED Mat512+Bin: %s", e)
        import traceback

        traceback.print_exc()

    # Matryoshka 256 + Scalar int8
    try:
        sq2 = ScalarQuantInt8()
        sq2.fit(X[:, :256])

        class SQTruncWrap:
            def __init__(self, sq: ScalarQuantInt8) -> None:
                self.sq = sq

            def compress(self, X: np.ndarray) -> np.ndarray:
                return self.sq.compress(X)

            def decompress(self, X_q: np.ndarray) -> np.ndarray:
                return self.sq.decompress(X_q)

            def compressed_bytes(self, n: int, d: int) -> int:
                return self.sq.compressed_bytes(n, 256)

        combo4 = CombinedMethod(
            256, SQTruncWrap(sq2), "Mat256+SQ8"
        )
        r = benchmark_method(
            "Matryoshka 256 + Scalar int8",
            X,
            combo4,
            queries,
            is_combined=True,
        )
        results.append(r)
    except Exception as e:
        logger.error("FAILED Mat256+SQ8: %s", e)
        import traceback

        traceback.print_exc()

    # ============================================================ #
    # Ethics validation                                             #
    # ============================================================ #
    ethics_results: Dict[str, Any] = {}
    if X_ethics is not None:
        logger.info("")
        logger.info("=" * 70)
        logger.info("ETHICS_CHUNKS VALIDATION (%d vectors)", len(X_ethics))
        logger.info("=" * 70)

        ne = len(X_ethics)
        eq_idx = rng.choice(ne, min(100, ne), replace=False)
        eq = X_ethics[eq_idx].copy()

        for label, builder in [
            (
                "turboquant_3bit",
                lambda: TurboQuantWrapper(dim=d, bits=3),
            ),
            (
                "scalar_int8",
                lambda: ScalarQuantInt8().fit(X_ethics),
            ),
        ]:
            try:
                comp = builder()
                kw = {}
                if "binary" in label:
                    kw["is_binary"] = True
                r = benchmark_method(
                    f"[Ethics] {label}", X_ethics, comp, eq, **kw
                )
                ethics_results[label] = r
            except Exception as e:
                logger.error("FAILED ethics %s: %s", label, e)

        try:
            bq = BinaryQuant()
            r = benchmark_method(
                "[Ethics] binary", X_ethics, bq, eq, is_binary=True
            )
            ethics_results["binary"] = r
        except Exception as e:
            logger.error("FAILED ethics binary: %s", e)

        try:
            pq16 = ProductQuantizer(d=d, M=16, nbits=8)
            pq16.fit(X_ethics[:10000])
            r = benchmark_method("[Ethics] PQ M=16", X_ethics, pq16, eq)
            ethics_results["pq_m16"] = r
        except Exception as e:
            logger.error("FAILED ethics PQ: %s", e)

        try:
            tq3e = TurboQuantWrapper(dim=256, bits=3)
            comboe = CombinedMethod(256, tq3e, "Mat256+TQ3")
            r = benchmark_method(
                "[Ethics] Matryoshka 256+TQ3",
                X_ethics,
                comboe,
                eq,
                is_combined=True,
            )
            ethics_results["mat256_tq3"] = r
        except Exception as e:
            logger.error("FAILED ethics combo: %s", e)

    # ============================================================ #
    # Recommendations                                               #
    # ============================================================ #
    recommendations = generate_recommendations(results)

    # ============================================================ #
    # Output                                                        #
    # ============================================================ #
    output = {
        "benchmark_info": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": "Atlas HP Z840 (2x Xeon E5-2690 v4, 48 cores)",
            "dataset": f"{n} vectors from chunks table, {d}-dim BGE-M3",
            "ethics_dataset": (
                f"{len(X_ethics)} vectors"
                if X_ethics is not None
                else "N/A"
            ),
        },
        "results": results,
        "ethics_results": ethics_results,
        "recommendations": recommendations,
    }

    out_path = "/home/claude/agi-hpc/benchmarks/compression_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)

    print_summary_table(results, recommendations)


def generate_recommendations(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate per-use-case recommendations from benchmark results."""
    if not results:
        return {}

    recs: Dict[str, Any] = {}

    max_ops = max(
        r["throughput"]["compress_ops_per_sec"] for r in results
    ) or 1
    max_decomp = max(
        1.0 / max(r["throughput"]["decompress_time_sec"], 1e-9)
        for r in results
    ) or 1

    use_cases = {
        "rag_search": {
            "weights": lambda r: (
                r["search"]["recall_at_10"] * 0.6
                + r["quality"]["mean_cosine"] * 0.3
                + min(r["compression_ratio"] / 50, 1.0) * 0.1
            ),
            "rationale": "High recall and cosine fidelity with decent compression",
        },
        "nats_transport": {
            "weights": lambda r: (
                min(r["compression_ratio"] / 50, 1.0) * 0.5
                + (r["throughput"]["compress_ops_per_sec"] / max_ops) * 0.3
                + r["quality"]["mean_cosine"] * 0.2
            ),
            "rationale": "Maximum compression with fast encode/decode for message bus",
        },
        "l2_ram_cache": {
            "weights": lambda r: (
                min(r["compression_ratio"] / 50, 1.0) * 0.3
                + (
                    (
                        1.0
                        / max(r["throughput"]["decompress_time_sec"], 1e-9)
                    )
                    / max_decomp
                )
                * 0.4
                + r["quality"]["mean_cosine"] * 0.3
            ),
            "rationale": "Fast decompression with good quality for hot cache",
        },
        "cold_storage": {
            "weights": lambda r: (
                min(r["compression_ratio"] / 50, 1.0) * 0.7
                + r["quality"]["mean_cosine"] * 0.2
                + r["search"]["recall_at_10"] * 0.1
            ),
            "rationale": "Maximum compression ratio for archival storage",
        },
        "realtime_kv_cache": {
            "weights": lambda r: (
                (r["throughput"]["compress_ops_per_sec"] / max_ops) * 0.3
                + r["quality"]["mean_cosine"] * 0.5
                + r["search"]["recall_at_10"] * 0.2
            ),
            "rationale": "High cosine fidelity with fast compression for attention",
        },
    }

    for uc_name, uc_cfg in use_cases.items():
        scored = [
            (uc_cfg["weights"](r), r["method"]) for r in results
        ]
        scored.sort(reverse=True)
        recs[uc_name] = {
            "recommended": scored[0][1],
            "score": round(scored[0][0], 4),
            "top_3": [
                (m, round(s, 4)) for s, m in scored[:3]
            ],
            "rationale": uc_cfg["rationale"],
        }

    return recs


def print_summary_table(
    results: List[Dict[str, Any]],
    recommendations: Dict[str, Any],
) -> None:
    """Print a formatted summary table."""
    print()
    hdr = (
        f"{'Method':<40} {'Ratio':>6} {'Cosine':>8} {'MinCos':>8} "
        f"{'R@10':>6} {'Comp/s':>10} {'B/Vec':>8} {'2.4M MB':>10}"
    )
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['method']:<40} "
            f"{r['compression_ratio']:>5.1f}x "
            f"{r['quality']['mean_cosine']:>8.6f} "
            f"{r['quality']['min_cosine']:>8.6f} "
            f"{r['search']['recall_at_10']:>6.4f} "
            f"{r['throughput']['compress_ops_per_sec']:>10.0f} "
            f"{r['bytes_per_vector']:>8.1f} "
            f"{r['memory_projection']['2_4m_vectors_mb']:>10.1f}"
        )
    print("=" * len(hdr))

    print()
    print("RECOMMENDATIONS BY USE CASE:")
    print("-" * 80)
    for uc, rec in recommendations.items():
        print(f"  {uc:<25} -> {rec['recommended']}")
        print(
            f"    {'':25}    Rationale: {rec['rationale']}"
        )
        print(
            f"    {'':25}    Top 3: "
            f"{', '.join(m for m, s in rec['top_3'])}"
        )
        print()

    orig_gb = 1024 * 4 * 2_400_000 / 1e9
    print(
        f"Original float32: {1024 * 4} bytes/vec = "
        f"{orig_gb:.2f} GB for 2.4M vectors"
    )


if __name__ == "__main__":
    main()
