#!/usr/bin/env python3
# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Perplexity evaluation for TurboQuant SVD weight compression.

Loads a HuggingFace model, compresses its weights via SVD + TurboQuant,
patches the model with CompressedLinear modules, and evaluates
perplexity on WikiText-2. Compares baseline vs compressed at multiple
settings.

Usage:
    python scripts/eval_weight_compression.py
    python scripts/eval_weight_compression.py --model gpt2 --quick
    python scripts/eval_weight_compression.py --model gpt2-medium --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def compute_perplexity(
    model,
    tokenizer,
    texts: list,
    max_length: int = 1024,
    stride: int = 512,
) -> dict:
    """Compute perplexity on a list of texts using sliding window.

    Returns dict with ppl, total_nll, total_tokens, time_s.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device

    total_nll = 0.0
    total_tokens = 0
    t0 = time.perf_counter()

    for text in texts:
        if not text.strip():
            continue
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encodings["input_ids"].to(device)
        seq_len = input_ids.size(1)
        if seq_len < 2:
            continue

        # Sliding window
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
    ppl = float("inf")
    if total_tokens > 0:
        import math

        ppl = math.exp(total_nll / total_tokens)

    return {
        "perplexity": round(ppl, 2),
        "total_nll": round(total_nll, 4),
        "total_tokens": total_tokens,
        "time_s": round(elapsed, 2),
    }


def load_wikitext(max_samples: int = 0) -> list:
    """Load WikiText-2 validation split."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t for t in ds["text"] if t.strip()]
    if max_samples > 0:
        texts = texts[:max_samples]
    return texts


def evaluate_config(
    model_name: str,
    energy: float,
    bits: int,
    texts: list,
    max_length: int = 1024,
    method: str = "beam",
) -> dict:
    """Evaluate one compression configuration."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from agi.meta.llm.turboquant_weights import (
        TurboQuantWeights,
        WeightCompressionConfig,
    )

    print(f"\n  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    # Baseline perplexity
    print("  Computing baseline perplexity...")
    baseline = compute_perplexity(model, tokenizer, texts, max_length=max_length)
    print(f"  Baseline PPL: {baseline['perplexity']:.2f}")

    # Compress
    print(f"  Compressing (method={method}, energy={energy}, bits={bits})...")
    config = WeightCompressionConfig(
        method=method,
        energy_threshold=energy,
        bits=bits,
    )
    engine = TurboQuantWeights(config)

    t0 = time.perf_counter()
    state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    compressed = engine.compress_state_dict(state)
    compress_time = time.perf_counter() - t0

    summary = compressed.summary()
    print(
        f"  Compressed: {summary['compressed_layers']} layers, "
        f"{summary['original_mb']:.1f}MB -> {summary['compressed_mb']:.1f}MB "
        f"({summary['ratio']:.1f}x)"
    )

    # Patch model
    print("  Patching model...")
    model2 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    engine.patch_torch_model(model2, compressed)

    # Compressed perplexity
    print("  Computing compressed perplexity...")
    compressed_ppl = compute_perplexity(model2, tokenizer, texts, max_length=max_length)
    print(f"  Compressed PPL: {compressed_ppl['perplexity']:.2f}")

    ppl_increase = compressed_ppl["perplexity"] / max(baseline["perplexity"], 1e-6)

    return {
        "model": model_name,
        "method": method,
        "energy_threshold": energy,
        "bits": bits,
        "baseline_ppl": baseline["perplexity"],
        "compressed_ppl": compressed_ppl["perplexity"],
        "ppl_ratio": round(ppl_increase, 4),
        "compression_ratio": round(summary["ratio"], 2),
        "original_mb": round(summary["original_mb"], 2),
        "compressed_mb": round(summary["compressed_mb"], 2),
        "savings_pct": round(summary["savings_pct"], 1),
        "compress_time_s": round(compress_time, 2),
        "baseline_eval_time_s": baseline["time_s"],
        "compressed_eval_time_s": compressed_ppl["time_s"],
        "total_tokens": baseline["total_tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TurboQuant weight compression perplexity evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace model name (default: gpt2)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick eval: single config, fewer samples",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep energy and bit settings",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of text samples (0=all)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max sequence length for eval",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="beam",
        choices=["beam", "beam_mixed", "svd", "both"],
        help="Compression method (default: beam)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/weight_compression_ppl.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    # Load evaluation data
    max_samples = args.max_samples or (50 if args.quick else 0)
    print(f"Loading WikiText-2 validation (max_samples={max_samples or 'all'})...")
    texts = load_wikitext(max_samples=max_samples)
    print(f"  {len(texts)} text samples loaded")

    # Configure sweep: (method, energy, bits)
    methods = ["beam", "beam_mixed", "svd"] if args.method == "both" else [args.method]
    if args.sweep:
        configs = [
            (m, e, b)
            for m in methods
            for e, b in [(0.85, 3), (0.90, 3), (0.95, 3), (0.99, 3), (0.95, 4)]
        ]
    elif args.quick:
        configs = [(m, 0.95, 3) for m in methods]
    else:
        configs = [(m, 0.95, 3) for m in methods] + [(m, 0.95, 4) for m in methods]

    results = []
    print(f"\nEvaluating {len(configs)} configurations on {args.model}...")

    for method, energy, bits in configs:
        try:
            r = evaluate_config(
                args.model,
                energy,
                bits,
                texts,
                max_length=args.max_length,
                method=method,
            )
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")

    # Print summary table
    print("\n" + "=" * 90)
    print(
        f"{'Method':>6}  {'Energy':>6}  {'Bits':>4}  {'Base PPL':>9}  {'Comp PPL':>9}  "
        f"{'PPL Ratio':>9}  {'Comp Ratio':>10}  {'Savings':>8}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['method']:>6}  {r['energy_threshold']:>6.2f}  {r['bits']:>4}  "
            f"{r['baseline_ppl']:>9.2f}  {r['compressed_ppl']:>9.2f}  "
            f"{r['ppl_ratio']:>9.4f}  {r['compression_ratio']:>10.1f}x  "
            f"{r['savings_pct']:>7.1f}%"
        )
    print("=" * 90)

    # Save
    output = {
        "benchmark_info": {
            "description": "TurboQuant weight compression perplexity evaluation",
            "model": args.model,
            "dataset": "wikitext-2-raw-v1 (validation)",
            "max_length": args.max_length,
            "num_samples": len(texts),
        },
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
