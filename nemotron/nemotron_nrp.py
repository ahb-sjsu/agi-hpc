"""Nemotron 3 Reasoning Challenge — NRP Adaptation

Adapted from nemotron_v3_kaggle.py for NRP Nautilus cluster.
Changes from Kaggle version:
  - Model from HuggingFace (not kagglehub)
  - Data from local mount or env var path
  - mamba_ssm pre-installed in Docker image
  - Output adapter weights to /output/ (mount or Ceph)
  - Supports multi-GPU via device_map="auto"

Run as a Kubernetes Job via nats-bursting:
  NEUROGOLF_WORKER=nemotron python3 /app/nemotron_nrp.py

Or directly:
  python3 nemotron_nrp.py --data-dir /data --output-dir /output --lr 1.5e-4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import polars as pl
import torch

# ═══════════════════════════════════════════════════════════════
# CLI args
# ═══════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("NEMOTRON_DATA", "/data"))
    ap.add_argument("--output-dir", default=os.environ.get("NEMOTRON_OUTPUT", "/output"))
    ap.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16")
    ap.add_argument("--lr", type=float, default=None, help="Learning rate (skip proxy sweep if set)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--n-augments", type=int, default=5, help="Augmentation multiplier per task type")
    ap.add_argument("--proxy-sweep", action="store_true", help="Run 3-config LR sweep on 20% data first")
    ap.add_argument("--validate", action="store_true", help="Run self-consistency validation after training")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ═══════════════════════════════════════════════════════════════
# Import the full v3 pipeline components
# (task classification, augmentation, CoT templates)
# ═══════════════════════════════════════════════════════════════

# These are imported from the v3 Kaggle script, adapted inline below.
# The group-theoretic augmentation is the core competitive edge.

def classify_task(prompt):
    p = prompt.lower()
    if "bit manipulation" in p or ("01" in prompt and "->" in prompt):
        return "bit"
    elif "encryption" in p or "decrypt" in p or "cipher" in p:
        return "encrypt"
    elif "gravitational" in p or "g =" in p or "free-fall" in p:
        return "physics"
    elif "unit conversion" in p or "convert" in p:
        return "unit"
    elif "numeral system" in p or "roman" in p:
        return "numeral"
    elif "transformation rules" in p or "symbol" in p:
        return "symbol"
    return "unknown"


# ═══════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[nemotron_nrp] GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[nemotron_nrp] VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else "")
    print(f"[nemotron_nrp] model: {args.model_name}")
    print(f"[nemotron_nrp] data: {args.data_dir}")
    print(f"[nemotron_nrp] output: {args.output_dir}")

    # ── Step 1: Load data ──────────────────────────────────────
    data_dir = Path(args.data_dir)
    train_df = pl.read_csv(str(data_dir / "train.csv"))
    test_df = pl.read_csv(str(data_dir / "test.csv"))
    print(f"[data] train={len(train_df)} test={len(test_df)}")

    # ── Step 2: Augment (import full v3 augmentation) ──────────
    # TODO: import augmentation from nemotron_v3_kaggle or inline here
    # For now, placeholder — the full augmentation is ~300 lines
    print(f"[augment] placeholder — wire in v3 group-theoretic augmentation")
    train_data = [
        {"prompt": row["prompt"], "answer": str(row["answer"]), "type": classify_task(row["prompt"])}
        for row in train_df.iter_rows(named=True)
    ]
    type_counts = Counter(d["type"] for d in train_data)
    print(f"[augment] task types: {dict(type_counts)}")

    # ── Step 3: Load model ─────────────────────────────────────
    print("[model] importing mamba_ssm...")
    import mamba_ssm  # Must import before model load
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    print(f"[model] loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print(f"[model] loaded. params={sum(p.numel() for p in model.parameters())/1e9:.1f}B")

    # ── Step 4: Apply LoRA ─────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=16,
        target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] r={args.lora_r} trainable={trainable/1e6:.1f}M")

    # ── Step 5: Format training data ───────────────────────────
    # TODO: apply full CoT template formatting per task type
    def format_example(d):
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{d['prompt']}\n\n"
            f"Think step by step. Put your final answer in \\boxed{{}}.\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"\\boxed{{{d['answer']}}}\n"
            f"<|eot_id|>"
        )

    formatted = [{"text": format_example(d)} for d in train_data]
    print(f"[format] {len(formatted)} training examples")

    # ── Step 6: Train ──────────────────────────────────────────
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    ds = Dataset.from_list(formatted)
    split = ds.train_test_split(test_size=0.1, seed=args.seed)

    lr = args.lr or 1.5e-4  # Default; use proxy sweep result if available
    print(f"[train] lr={lr} epochs={args.epochs} batch={args.batch_size}×{args.grad_accum}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=args.max_seq_len,
        logging_steps=10,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed/60:.1f} min")

    # ── Step 7: Save adapter ───────────────────────────────────
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[save] adapter saved to {adapter_dir}")

    # Create submission.zip
    subprocess.run(
        f"cd {adapter_dir} && zip -r {output_dir}/submission.zip *",
        shell=True, check=True)
    print(f"[save] submission.zip at {output_dir}/submission.zip")
    print(f"[done] total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
