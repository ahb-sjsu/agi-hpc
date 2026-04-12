#!/usr/bin/env python3
# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Comprehensive TurboQuant weight compression benchmark on Atlas.

Designed for large models (7B-72B) on Atlas hardware (251GB RAM,
2x Quadro GV100 32GB). Uses layer-by-layer compression to avoid
2x memory overhead, fp16 loading, and thermal-safe thread caps.

Usage (run ON Atlas, not remotely):
    python scripts/benchmark_atlas_compression.py --model Qwen/Qwen2.5-7B-Instruct
    python scripts/benchmark_atlas_compression.py --model Qwen/Qwen2.5-7B-Instruct \\
        --methods beam beam_mixed
    python scripts/benchmark_atlas_compression.py --all  # run full suite

Credit: Mixed-precision method inspired by Reddit u/FabulousExample4605
(r/LocalLLaMA per-weight precision selection, Apr 2026).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

# Thermal safety: start with conservative cap, ThermalController adjusts dynamically
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from agi.meta.llm.turboquant_weights import (  # noqa: E402
    CompressedModel,
    CompressedWeight,
    TurboQuantWeights,
    WeightCompressionConfig,
)

# Dynamic thermal control via batch-probe
# Target 75C (not 80) with max 12 threads (not 20) — sustained
# compression + eval at 20 threads hits 100C on the dual Xeons.
_thermal = None
try:
    from batch_probe import ThermalController

    _thermal = ThermalController(
        target_temp=75.0,
        max_threads=12,
        min_threads=2,
        auto_apply=True,
        verbose=True,
    )
    _thermal.start()
    print("[thermal] ThermalController: target=75C, max=12, auto_apply=True")
except ImportError:
    torch.set_num_threads(8)
    os.environ["OMP_NUM_THREADS"] = "8"
    print("[thermal] batch-probe not available, static 8 threads")


def _sync_thermal() -> None:
    """Sync torch thread count with ThermalController."""
    if _thermal is not None:
        n = _thermal.get_threads()
        torch.set_num_threads(n)
        os.environ["OMP_NUM_THREADS"] = str(n)


# ------------------------------------------------------------------ #
# Perplexity evaluation                                               #
# ------------------------------------------------------------------ #


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: list,
    max_length: int = 1024,
    stride: int = 512,
    device: str = "cpu",
) -> dict:
    """Sliding-window perplexity on a list of texts."""
    model.eval()

    _sync_thermal()

    # Resolve actual device from model (handles device_map="auto")
    if device == "auto":
        try:
            actual_device = next(model.parameters()).device
        except StopIteration:
            actual_device = torch.device("cpu")
    else:
        actual_device = torch.device(device)

    total_nll = 0.0
    total_tokens = 0
    t0 = time.perf_counter()

    for text in texts:
        if not text.strip():
            continue
        encodings = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        input_ids = encodings["input_ids"].to(actual_device)
        seq_len = input_ids.size(1)
        if seq_len < 2:
            continue

        for begin in range(0, seq_len - 1, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - begin - 1
            if target_len <= 0:
                continue
            ids = input_ids[:, begin:end]
            with torch.no_grad():
                outputs = model(ids, labels=ids)
                nll = outputs.loss.item() * target_len
            total_nll += nll
            total_tokens += target_len
            if end >= seq_len:
                break

    elapsed = time.perf_counter() - t0
    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
    return {
        "perplexity": round(ppl, 2),
        "total_tokens": total_tokens,
        "time_s": round(elapsed, 2),
    }


# ------------------------------------------------------------------ #
# Layer-by-layer compression (memory-safe for large models)           #
# ------------------------------------------------------------------ #


def compress_model_streaming(
    model: torch.nn.Module,
    config: WeightCompressionConfig,
) -> CompressedModel:
    """Compress model weights layer-by-layer to minimize peak RAM.

    Unlike compress_state_dict (which copies the full state_dict),
    this iterates over named_parameters and compresses in-place.
    """
    engine = TurboQuantWeights(config)
    weights: dict[str, CompressedWeight] = {}
    uncompressed: dict[str, np.ndarray] = {}
    n_compressed = 0
    n_skipped = 0

    params = list(model.named_parameters())
    total = len(params)

    for i, (name, param) in enumerate(params):
        arr = param.detach().cpu().float().numpy()

        if engine._should_skip(name, arr):
            uncompressed[name] = arr
            n_skipped += 1
        else:
            weights[name] = engine.compress_weight(arr, name=name)
            n_compressed += 1

        del arr
        if (i + 1) % 20 == 0:
            gc.collect()
            print(f"    [{i+1}/{total}] {n_compressed} compressed, {n_skipped} skipped")

    cm = CompressedModel(weights=weights, uncompressed=uncompressed, config=config)
    print(
        f"    Done: {n_compressed} compressed, {n_skipped} skipped, "
        f"{cm.compression_ratio():.1f}x overall"
    )
    return cm


# ------------------------------------------------------------------ #
# Single model benchmark                                              #
# ------------------------------------------------------------------ #


def benchmark_model(
    model_name: str,
    methods: list[str],
    bits_list: list[int],
    max_samples: int = 50,
    max_length: int = 1024,
    device: str = "cpu",
    output_path: str | None = None,
) -> list[dict]:
    """Benchmark all configs for a single model."""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")

    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("  Loading WikiText-2...")
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t for t in ds["text"] if t.strip()][:max_samples]
    print(f"  {len(texts)} text samples")

    # Load model
    print(f"  Loading model (fp16, device={device})...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    load_time = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    model_mb = sum(p.nbytes for p in model.parameters()) / (1024 * 1024)
    print(f"  Loaded: {n_params/1e9:.1f}B params, {model_mb:.0f}MB, {load_time:.0f}s")

    # Baseline PPL
    print("  Computing baseline perplexity...")
    baseline = compute_perplexity(
        model, tokenizer, texts, max_length=max_length, device=device
    )
    print(f"  Baseline PPL: {baseline['perplexity']:.2f} ({baseline['time_s']:.0f}s)")

    results = []

    for method in methods:
        for bits in bits_list:
            label = f"{method}/{bits}-bit"
            print(f"\n  --- {label} ---")

            config = WeightCompressionConfig(method=method, bits=bits, use_gpu=True)
            engine = TurboQuantWeights(config)

            # Compress (sync thermal before heavy CPU work)
            _sync_thermal()
            print(f"  Compressing ({method}, {bits}-bit)...")
            t0 = time.perf_counter()
            compressed = compress_model_streaming(model, config)
            compress_time = time.perf_counter() - t0
            summary = compressed.summary()

            print(
                f"  {summary['compressed_layers']} layers, "
                f"{summary['original_mb']:.0f}MB -> {summary['compressed_mb']:.0f}MB "
                f"({summary['ratio']:.1f}x), {compress_time:.0f}s"
            )

            # Reload fresh model for patching
            print("  Reloading model for eval...")
            model2 = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            engine.patch_torch_model(model2, compressed)

            # Eval
            print("  Computing compressed perplexity...")
            comp_ppl = compute_perplexity(
                model2, tokenizer, texts, max_length=max_length, device=device
            )
            print(f"  Compressed PPL: {comp_ppl['perplexity']:.2f}")

            ppl_ratio = comp_ppl["perplexity"] / max(baseline["perplexity"], 1e-6)

            result = {
                "model": model_name,
                "params_b": round(n_params / 1e9, 1),
                "method": method,
                "bits": bits,
                "baseline_ppl": baseline["perplexity"],
                "compressed_ppl": comp_ppl["perplexity"],
                "ppl_ratio": round(ppl_ratio, 4),
                "compression_ratio": round(summary["ratio"], 2),
                "original_mb": round(summary["original_mb"], 0),
                "compressed_mb": round(summary["compressed_mb"], 0),
                "savings_pct": round(summary["savings_pct"], 1),
                "compress_time_s": round(compress_time, 0),
                "eval_time_s": comp_ppl["time_s"],
            }
            results.append(result)

            # Save incrementally (crash-safe)
            if output_path:
                _save_incremental(results, model_name, output_path)

            # Free memory
            del model2, compressed, engine
            gc.collect()

    # Print summary table
    _print_summary(results)
    return results


def _save_incremental(results: list, model_name: str, path: str) -> None:
    """Save results incrementally to JSON."""
    output = {
        "benchmark_info": {
            "description": "TurboQuant weight compression on Atlas (HP Z840)",
            "hardware": "2x Quadro GV100 32GB, 251GB RAM, 2x Xeon E5-2690 v3",
            "credit": "Mixed-precision inspired by Reddit u/FabulousExample4605",
        },
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def _print_summary(results: list) -> None:
    """Print results table."""
    if not results:
        return
    print(f"\n{'='*95}")
    print(
        f"{'Model':>25} {'Method':>10} {'Bits':>4} "
        f"{'Base PPL':>9} {'Comp PPL':>9} {'Ratio':>7} "
        f"{'Comp':>6} {'Save%':>6}"
    )
    print("-" * 95)
    for r in results:
        model_short = r["model"].split("/")[-1][:25]
        print(
            f"{model_short:>25} {r['method']:>10} {r['bits']:>4} "
            f"{r['baseline_ppl']:>9.2f} {r['compressed_ppl']:>9.2f} "
            f"{r['ppl_ratio']:>7.2f} "
            f"{r['compression_ratio']:>5.1f}x {r['savings_pct']:>5.1f}%"
        )
    print("=" * 95)


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TurboQuant weight compression benchmark (Atlas)"
    )
    parser.add_argument("--model", type=str, help="Single HF model to benchmark")
    parser.add_argument("--all", action="store_true", help="Run all default models")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["beam", "beam_mixed"],
        help="Compression methods",
    )
    parser.add_argument(
        "--bits", nargs="+", type=int, default=[3, 4], help="Bit widths"
    )
    parser.add_argument(
        "--max-samples", type=int, default=50, help="WikiText-2 samples"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for eval (cpu or cuda:N)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/atlas_weight_compression.json",
        help="Output JSON",
    )
    args = parser.parse_args()

    if args.all:
        models = DEFAULT_MODELS
    elif args.model:
        models = [args.model]
    else:
        parser.error("Specify --model MODEL or --all")

    print("TurboQuant Weight Compression Benchmark (Atlas)")
    print(f"Models: {len(models)}")
    print(f"Methods: {args.methods}")
    print(f"Bits: {args.bits}")
    print(f"Samples: {args.max_samples}")
    print(f"Device: {args.device}")
    print(f"Threads: {os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"RAM: {os.popen('free -h | head -2').read().strip()}")

    all_results = []
    for model_name in models:
        try:
            results = benchmark_model(
                model_name=model_name,
                methods=args.methods,
                bits_list=args.bits,
                max_samples=args.max_samples,
                device=args.device,
                output_path=args.output,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"\n  FAILED {model_name}: {e}")
            import traceback

            traceback.print_exc()

        # Aggressive cleanup between models
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Final save
    _save_incremental(all_results, "all", args.output)

    print(f"\n\nFINAL RESULTS ({len(all_results)} configs)")
    _print_summary(all_results)
    print(f"\nSaved to {args.output}")

    if _thermal is not None:
        _thermal.stop()
        summary = _thermal.summary()
        if summary:
            print(
                f"\n[thermal] Summary: avg={summary['temp_mean']:.0f}C "
                f"max={summary['temp_max']:.0f}C "
                f"threads={summary['threads_mean']:.0f} "
                f"({summary['threads_min']}-{summary['threads_max']})"
            )


if __name__ == "__main__":
    main()
