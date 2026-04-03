#!/usr/bin/env python3
# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without WARRANTIES or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""
Benchmark TurboQuant KV cache compression.

Measures compression ratio, reconstruction error, and throughput for
2-bit, 3-bit, and 4-bit quantisation on simulated KV cache tensors
matching Gemma 4 27B dimensions.

Benchmarks include:
  - Packed vs unpacked compression ratios
  - GPU kernel throughput (if CuPy available)
  - Streaming cache simulation (append + query pattern)
  - Baseline comparison (no compression) end-to-end

Adapted from Theory Radar's TurboBeam for the Gemma 4 Good Hackathon.

Usage (CPU only -- safe to run alongside GPU workloads):
    python scripts/benchmark_turboquant_kv.py

    # With CuPy on a specific GPU:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_turboquant_kv.py --gpu

    # Quick mode (smaller tensors):
    python scripts/benchmark_turboquant_kv.py --quick
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Dict, List

import numpy as np

# Ensure the source tree is importable when running from repo root
sys.path.insert(0, "src")

from agi.meta.llm.turboquant_kv import (  # noqa: E402
    TurboQuantKV,
    TurboQuantKVCache,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Gemma 4 27B model dimensions                                       #
# ------------------------------------------------------------------ #

GEMMA4_CONFIG = {
    "n_layers": 36,
    "n_kv_heads": 16,
    "head_dim": 256,
    "hidden_dim": 4096,
}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity between corresponding vectors."""
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    dot = np.sum(a_flat * b_flat, axis=-1)
    norm_a = np.linalg.norm(a_flat, axis=-1)
    norm_b = np.linalg.norm(b_flat, axis=-1)
    denom = np.maximum(norm_a * norm_b, 1e-30)
    return float(np.mean(dot / denom))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Mean relative L2 error per vector."""
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    err = np.linalg.norm(a_flat - b_flat, axis=-1)
    orig = np.maximum(np.linalg.norm(a_flat, axis=-1), 1e-30)
    return float(np.mean(err / orig))


# ------------------------------------------------------------------ #
# Single-config benchmark (packed and unpacked)                       #
# ------------------------------------------------------------------ #


def benchmark_single(
    bits: int,
    seq_len: int,
    use_gpu: bool,
    packed: bool = False,
    n_warmup: int = 1,
    n_trials: int = 3,
) -> Dict[str, float]:
    """Run a single benchmark configuration.

    Args:
        bits: Quantisation width (2, 3, or 4).
        seq_len: Sequence length to simulate.
        use_gpu: Whether to use CuPy.
        packed: Whether to use bit-packing.
        n_warmup: Number of warm-up iterations (not timed).
        n_trials: Number of timed iterations.

    Returns:
        Dict with metrics.
    """
    head_dim = GEMMA4_CONFIG["head_dim"]
    n_kv_heads = GEMMA4_CONFIG["n_kv_heads"]
    batch = 1

    rng = np.random.default_rng(42)
    tensor = rng.standard_normal((batch, n_kv_heads, seq_len, head_dim)).astype(
        np.float16
    )

    tq = TurboQuantKV(
        head_dim=head_dim,
        n_heads=n_kv_heads,
        bits=bits,
        use_gpu=use_gpu,
        seed=0,
    )

    # Warm-up
    for _ in range(n_warmup):
        c = tq.compress(tensor, packed=packed)
        _ = tq.decompress(c)

    # Timed compression
    t0 = time.perf_counter()
    for _ in range(n_trials):
        compressed = tq.compress(tensor, packed=packed)
    compress_time = (time.perf_counter() - t0) / n_trials

    # Timed decompression
    t0 = time.perf_counter()
    for _ in range(n_trials):
        reconstructed = tq.decompress(compressed)
    decompress_time = (time.perf_counter() - t0) / n_trials

    # Quality metrics (on float32 for precision)
    tensor_f32 = tensor.astype(np.float32)
    cos_sim = _cosine_similarity(tensor_f32, reconstructed)
    mse = _mse(tensor_f32, reconstructed)
    rel_err = _relative_error(tensor_f32, reconstructed)

    # Throughput
    tensor_mb = tensor.nbytes / (1024**2)
    compress_throughput = tensor_mb / max(compress_time, 1e-9)
    decompress_throughput = tensor_mb / max(decompress_time, 1e-9)

    # Compression ratio
    ratio = compressed.compression_ratio(head_dim)

    return {
        "bits": bits,
        "seq_len": seq_len,
        "packed": 1.0 if packed else 0.0,
        "tensor_mb": tensor_mb,
        "compressed_mb": compressed.nbytes() / (1024**2),
        "ratio": ratio,
        "mse": mse,
        "cosine_sim": cos_sim,
        "relative_err": rel_err,
        "compress_ms": compress_time * 1000,
        "decompress_ms": decompress_time * 1000,
        "compress_mb_s": compress_throughput,
        "decompress_mb_s": decompress_throughput,
    }


# ------------------------------------------------------------------ #
# Streaming cache benchmark                                           #
# ------------------------------------------------------------------ #


def benchmark_streaming(
    n_tokens: int,
    use_gpu: bool,
    hot_window: int = 512,
    bits: int = 3,
) -> Dict[str, float]:
    """Benchmark the streaming TurboQuantKVCache.

    Simulates an autoregressive inference scenario: append tokens one
    at a time, periodically query the full cache.

    Args:
        n_tokens: Total number of tokens to simulate.
        use_gpu: Whether to use CuPy.
        hot_window: Hot window size.
        bits: Quantisation bits.

    Returns:
        Dict with timing and memory stats.
    """
    head_dim = GEMMA4_CONFIG["head_dim"]
    n_kv_heads = GEMMA4_CONFIG["n_kv_heads"]

    rng = np.random.default_rng(42)

    cache = TurboQuantKVCache(
        head_dim=head_dim,
        n_heads=n_kv_heads,
        bits=bits,
        hot_window=hot_window,
        use_gpu=use_gpu,
        seed=0,
    )

    # Timed append
    t0 = time.perf_counter()
    for i in range(n_tokens):
        k = rng.standard_normal((1, n_kv_heads, 1, head_dim)).astype(np.float32)
        v = rng.standard_normal((1, n_kv_heads, 1, head_dim)).astype(np.float32)
        cache.append(k, v)
    append_time = time.perf_counter() - t0

    # Timed full-range query
    t0 = time.perf_counter()
    keys = cache.get_keys(0, cache.length)
    values = cache.get_values(0, cache.length)
    query_time = time.perf_counter() - t0

    # Timed hot-only query
    t0 = time.perf_counter()
    hot_start = max(0, cache.length - hot_window)
    hot_keys = cache.get_keys(hot_start, cache.length)
    hot_query_time = time.perf_counter() - t0

    stats = cache.memory_stats()

    return {
        "n_tokens": n_tokens,
        "hot_window": hot_window,
        "bits": bits,
        "cold_tokens": cache.cold_length,
        "hot_tokens": cache.hot_length,
        "append_total_ms": append_time * 1000,
        "append_per_token_us": append_time / n_tokens * 1e6,
        "full_query_ms": query_time * 1000,
        "hot_query_ms": hot_query_time * 1000,
        "cold_mb": stats["cold_bytes"] / (1024**2),
        "hot_mb": stats["hot_bytes"] / (1024**2),
        "total_mb": stats["total_bytes"] / (1024**2),
        "uncompressed_mb": stats["uncompressed_equivalent_bytes"] / (1024**2),
        "effective_ratio": stats["effective_ratio"],
    }


# ------------------------------------------------------------------ #
# Bit-packing throughput benchmark                                    #
# ------------------------------------------------------------------ #


def benchmark_packing_throughput(
    bits: int,
    n_elements: int,
    use_gpu: bool,
    n_trials: int = 5,
) -> Dict[str, float]:
    """Benchmark raw bit-packing throughput.

    Args:
        bits: Bit width (2, 3, or 4).
        n_elements: Number of index values to pack/unpack.
        use_gpu: Whether to use CuPy GPU kernels.
        n_trials: Number of timed iterations.

    Returns:
        Dict with throughput metrics.
    """
    tq = TurboQuantKV(head_dim=128, n_heads=1, bits=bits, use_gpu=use_gpu, seed=0)

    rng = np.random.default_rng(42)
    indices = rng.integers(0, 2**bits, size=n_elements, dtype=np.uint8)

    if use_gpu:
        try:
            import cupy as _cp

            indices_dev = _cp.asarray(indices)
        except ImportError:
            return {"error": 1.0}
    else:
        indices_dev = indices

    # Warm-up
    packed = tq._pack_bits(indices_dev)
    _ = tq._unpack_bits(packed, n_elements)

    # Timed pack
    t0 = time.perf_counter()
    for _ in range(n_trials):
        packed = tq._pack_bits(indices_dev)
    pack_time = (time.perf_counter() - t0) / n_trials

    # Timed unpack
    t0 = time.perf_counter()
    for _ in range(n_trials):
        unpacked = tq._unpack_bits(packed, n_elements)
    unpack_time = (time.perf_counter() - t0) / n_trials

    input_mb = n_elements / (1024**2)
    packed_mb = len(packed.ravel()) / (1024**2)

    return {
        "bits": bits,
        "n_elements_m": n_elements / 1e6,
        "pack_ms": pack_time * 1000,
        "unpack_ms": unpack_time * 1000,
        "pack_throughput_mb_s": input_mb / max(pack_time, 1e-9),
        "unpack_throughput_mb_s": packed_mb / max(unpack_time, 1e-9),
        "packing_ratio": n_elements / max(len(packed.ravel()), 1),
    }


# ------------------------------------------------------------------ #
# Baseline comparison (no compression)                                #
# ------------------------------------------------------------------ #


def benchmark_baseline_comparison(
    seq_len: int,
    bits: int = 3,
    use_gpu: bool = False,
) -> Dict[str, float]:
    """Compare end-to-end: no compression vs packed compression.

    Args:
        seq_len: Sequence length to simulate.
        bits: Quantisation bits.
        use_gpu: Whether to use CuPy.

    Returns:
        Dict with comparison metrics.
    """
    head_dim = GEMMA4_CONFIG["head_dim"]
    n_kv_heads = GEMMA4_CONFIG["n_kv_heads"]

    rng = np.random.default_rng(42)
    tensor_fp16 = rng.standard_normal((1, n_kv_heads, seq_len, head_dim)).astype(
        np.float16
    )

    # Baseline: no compression, just fp16 storage
    baseline_bytes = tensor_fp16.nbytes * 2  # K + V

    # Packed compression
    tq = TurboQuantKV(
        head_dim=head_dim,
        n_heads=n_kv_heads,
        bits=bits,
        use_gpu=use_gpu,
        seed=0,
    )

    t0 = time.perf_counter()
    compressed = tq.compress(tensor_fp16, packed=True)
    compress_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    reconstructed = tq.decompress(compressed)
    decompress_time = time.perf_counter() - t0

    compressed_bytes = compressed.nbytes() * 2  # K + V
    cos_sim = _cosine_similarity(tensor_fp16.astype(np.float32), reconstructed)

    return {
        "seq_len": seq_len,
        "bits": bits,
        "baseline_mb": baseline_bytes / (1024**2),
        "compressed_mb": compressed_bytes / (1024**2),
        "ratio": baseline_bytes / max(compressed_bytes, 1),
        "cosine_sim": cos_sim,
        "compress_ms": compress_time * 1000,
        "decompress_ms": decompress_time * 1000,
        "savings_mb": (baseline_bytes - compressed_bytes) / (1024**2),
    }


# ------------------------------------------------------------------ #
# Print helpers                                                       #
# ------------------------------------------------------------------ #


def print_results(results: List[Dict[str, float]]) -> None:
    """Pretty-print benchmark results."""
    print()
    print("=" * 90)
    print("TurboQuant KV Cache Compression Benchmark")
    print(
        f"Model: Gemma 4 27B  (n_kv_heads={GEMMA4_CONFIG['n_kv_heads']}, "
        f"head_dim={GEMMA4_CONFIG['head_dim']})"
    )
    print("=" * 90)

    # Separate packed vs unpacked results
    unpacked = [r for r in results if r.get("packed", 0.0) < 0.5]
    packed = [r for r in results if r.get("packed", 0.0) >= 0.5]

    for label, group in [
        ("UNPACKED (uint8)", unpacked),
        ("PACKED (bit-packed)", packed),
    ]:
        if not group:
            continue
        print()
        print(f"--- {label} ---")
        seq_lens = sorted({int(r["seq_len"]) for r in group})
        for seq_len in seq_lens:
            sub = [r for r in group if int(r["seq_len"]) == seq_len]
            print(f"  Context: {seq_len} tokens")
            print(
                f"  {'Bits':>6} {'Ratio':>8} {'MSE':>12} {'Cos Sim':>10} "
                f"{'Rel Err':>10} {'Comp ms':>10} {'Decomp ms':>10} "
                f"{'Comp MB/s':>10} {'Decomp MB/s':>11}"
            )
            for r in sorted(sub, key=lambda x: x["bits"]):
                print(
                    f"  {int(r['bits']):>6d} {r['ratio']:>8.2f}x "
                    f"{r['mse']:>12.6f} {r['cosine_sim']:>10.6f} "
                    f"{r['relative_err']:>10.6f} "
                    f"{r['compress_ms']:>10.1f} {r['decompress_ms']:>10.1f} "
                    f"{r['compress_mb_s']:>10.1f} {r['decompress_mb_s']:>11.1f}"
                )
            print()


def print_packing_throughput(results: List[Dict[str, float]]) -> None:
    """Print bit-packing throughput results."""
    print()
    print("=" * 90)
    print("Bit-Packing Throughput")
    print("=" * 90)
    print(
        f"  {'Bits':>6} {'Elements':>10} {'Pack ms':>10} {'Unpack ms':>10} "
        f"{'Pack MB/s':>12} {'Unpack MB/s':>12} {'Pack Ratio':>12}"
    )
    for r in results:
        if "error" in r:
            continue
        print(
            f"  {int(r['bits']):>6d} {r['n_elements_m']:>8.1f}M "
            f"{r['pack_ms']:>10.2f} {r['unpack_ms']:>10.2f} "
            f"{r['pack_throughput_mb_s']:>12.1f} "
            f"{r['unpack_throughput_mb_s']:>12.1f} "
            f"{r['packing_ratio']:>12.2f}"
        )
    print()


def print_streaming_results(results: List[Dict[str, float]]) -> None:
    """Print streaming cache benchmark results."""
    print()
    print("=" * 90)
    print("Streaming KV Cache (TurboQuantKVCache)")
    print("=" * 90)
    for r in results:
        print(
            f"  Tokens={int(r['n_tokens']):>6d}  "
            f"hot_window={int(r['hot_window']):>4d}  "
            f"bits={int(r['bits'])}  "
            f"cold={int(r['cold_tokens'])}  hot={int(r['hot_tokens'])}"
        )
        print(
            f"    Append: {r['append_total_ms']:.1f} ms total, "
            f"{r['append_per_token_us']:.1f} us/token"
        )
        print(
            f"    Query:  full={r['full_query_ms']:.1f} ms, "
            f"hot-only={r['hot_query_ms']:.1f} ms"
        )
        print(
            f"    Memory: {r['total_mb']:.2f} MB compressed "
            f"({r['uncompressed_mb']:.2f} MB uncompressed), "
            f"ratio={r['effective_ratio']:.2f}x"
        )
        print()


def print_baseline_comparison(results: List[Dict[str, float]]) -> None:
    """Print baseline vs compressed comparison."""
    print()
    print("=" * 90)
    print("End-to-End: Baseline (fp16) vs Packed Compression")
    print("=" * 90)
    print(
        f"  {'Context':>10} {'Bits':>6} {'Baseline MB':>12} "
        f"{'Compressed MB':>14} {'Ratio':>8} {'Saved MB':>10} "
        f"{'Cos Sim':>10}"
    )
    for r in results:
        print(
            f"  {int(r['seq_len']):>10d} {int(r['bits']):>6d} "
            f"{r['baseline_mb']:>12.2f} {r['compressed_mb']:>14.2f} "
            f"{r['ratio']:>7.2f}x {r['savings_mb']:>10.2f} "
            f"{r['cosine_sim']:>10.6f}"
        )
    print()


def print_memory_estimates() -> None:
    """Print full-model memory estimates."""
    print()
    print("=" * 90)
    print("Full-model KV cache memory estimates (Gemma 4 27B, fp16 baseline)")
    print("=" * 90)

    for label, bit_packed in [
        ("Current: uint8 storage (1 byte per index)", False),
        ("With bit-packing (b bits per index)", True),
    ]:
        print()
        print(f"  {label}")
        print(
            f"  {'Context':>10} {'Bits':>6} {'Original':>12} "
            f"{'Compressed':>12} {'Ratio':>8} {'Saved':>12}"
        )
        for ctx in [2048, 4096, 8192, 16384, 32768]:
            for bits in [2, 3, 4]:
                est = TurboQuantKV.estimate_memory(
                    n_layers=GEMMA4_CONFIG["n_layers"],
                    n_kv_heads=GEMMA4_CONFIG["n_kv_heads"],
                    head_dim=GEMMA4_CONFIG["head_dim"],
                    seq_len=ctx,
                    bits=bits,
                    original_dtype="float16",
                    bit_packed=bit_packed,
                )
                print(
                    f"  {ctx:>10d} {bits:>6d} "
                    f"{est['original_gb']:>10.3f} GB "
                    f"{est['compressed_gb']:>10.3f} GB "
                    f"{est['ratio']:>7.2f}x "
                    f"{est['saved_gb']:>10.3f} GB"
                )
        print()


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TurboQuant KV cache compression"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use CuPy GPU backend (default: CPU/NumPy)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with smaller tensors",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of timed trials (default: 3)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # ---- 1. Packed vs unpacked compression ----------------------------
    if args.quick:
        seq_lens = [256, 1024]
    else:
        seq_lens = [512, 2048, 8192]

    bits_list = [2, 3, 4]
    results: List[Dict[str, float]] = []

    for seq_len in seq_lens:
        for bits in bits_list:
            for packed in [False, True]:
                logger.info(
                    "Benchmarking: bits=%d, seq_len=%d, packed=%s, gpu=%s",
                    bits,
                    seq_len,
                    packed,
                    args.gpu,
                )
                r = benchmark_single(
                    bits=bits,
                    seq_len=seq_len,
                    use_gpu=args.gpu,
                    packed=packed,
                    n_trials=args.trials,
                )
                results.append(r)

    print_results(results)

    # ---- 2. Bit-packing throughput ------------------------------------
    packing_results: List[Dict[str, float]] = []
    for bits in bits_list:
        n_elements = 1_000_000 if args.quick else 10_000_000
        logger.info(
            "Packing throughput: bits=%d, n_elements=%d, gpu=%s",
            bits,
            n_elements,
            args.gpu,
        )
        r = benchmark_packing_throughput(
            bits=bits,
            n_elements=n_elements,
            use_gpu=args.gpu,
            n_trials=args.trials,
        )
        packing_results.append(r)

    print_packing_throughput(packing_results)

    # ---- 3. Streaming cache simulation --------------------------------
    streaming_results: List[Dict[str, float]] = []
    n_tokens_list = [256, 1024] if args.quick else [512, 2048]
    for n_tokens in n_tokens_list:
        logger.info("Streaming cache: n_tokens=%d, gpu=%s", n_tokens, args.gpu)
        r = benchmark_streaming(
            n_tokens=n_tokens,
            use_gpu=args.gpu,
            hot_window=256 if args.quick else 512,
            bits=3,
        )
        streaming_results.append(r)

    print_streaming_results(streaming_results)

    # ---- 4. Baseline comparison ---------------------------------------
    baseline_results: List[Dict[str, float]] = []
    baseline_seqs = [1024] if args.quick else [2048, 8192]
    for seq_len in baseline_seqs:
        for bits in bits_list:
            logger.info("Baseline comparison: seq_len=%d, bits=%d", seq_len, bits)
            r = benchmark_baseline_comparison(
                seq_len=seq_len, bits=bits, use_gpu=args.gpu
            )
            baseline_results.append(r)

    print_baseline_comparison(baseline_results)

    # ---- 5. Memory estimates ------------------------------------------
    print_memory_estimates()


if __name__ == "__main__":
    main()
