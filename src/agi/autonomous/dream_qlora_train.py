"""QLoRA fine-tuning for Erebus — runs during the dream window.

Reads verified (task_examples, python_transform) pairs from
/archive/neurogolf/training_data/*.jsonl, fine-tunes a rank-16 LoRA
adapter on top of Qwen2.5-32B-Instruct (4-bit NF4 base), saves the
adapter to /archive/neurogolf/models/erebus/lora_YYYYMMDD/.

The base model matches Kirk (the cortex on Atlas GPU 1), so the trained
adapter can be loaded into Kirk to produce an Erebus-tuned cortex.

Usage:
  python dream_qlora_train.py                       # train on all JSONL in training_data/
  python dream_qlora_train.py --day 2026-04-18      # only one day's JSONL
  python dream_qlora_train.py --base Qwen/Qwen2.5-14B-Instruct --rank 8 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("qlora")

DEFAULT_BASE = "Qwen/Qwen2.5-32B-Instruct"
TRAIN_DIR = Path("/archive/neurogolf/training_data")
OUT_DIR = Path("/archive/neurogolf/models/erebus")


def load_training_pairs(day: str | None) -> list[dict]:
    """Load JSONL training pairs. If `day` is given (YYYY-MM-DD), only that
    day's file; otherwise all daily files."""
    if day:
        files = [TRAIN_DIR / f"solves_{day}.jsonl"]
    else:
        files = sorted(TRAIN_DIR.glob("solves_*.jsonl"))
    pairs = []
    for fp in files:
        if not fp.exists():
            continue
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pairs


def format_example(pair: dict) -> dict:
    """Format one training pair as a chat-style example.

    Returns {"messages": [{"role": ..., "content": ...}, ...]} —
    Qwen's chat template will render this to the expected prompt format.
    """
    # Render the training examples compactly
    grids = []
    for ex in pair["task_examples"][:3]:   # cap to avoid runaway length
        inp = ex["input"]
        out = ex["output"]
        grids.append(f"Input:\n{inp}\nOutput:\n{out}")
    examples_str = "\n\n".join(grids)

    user = (
        "You are solving an ARC-AGI task. Given the input/output examples below, "
        "write a Python `transform(grid)` function that maps input grids to "
        "output grids. Return only the function.\n\n"
        f"{examples_str}"
    )
    assistant = f"```python\n{pair['python_transform'].strip()}\n```"
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def build_dataset(pairs: list[dict], tokenizer, max_len: int = 2048):
    """Tokenize chat examples with Qwen chat template. Mask labels on the user
    prompt so the loss only applies to the assistant's Python code."""
    import torch
    from torch.utils.data import Dataset

    class SFTDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            msgs = self.examples[idx]["messages"]
            # Render the full prompt
            full = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False)
            # Render just the user portion to know where to start masking
            user_only = tokenizer.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True)

            full_ids = tokenizer(full, truncation=True, max_length=max_len,
                                 return_tensors=None)["input_ids"]
            user_ids = tokenizer(user_only, truncation=True, max_length=max_len,
                                 return_tensors=None)["input_ids"]

            labels = list(full_ids)
            # Mask the user portion: only learn to generate the assistant response
            for i in range(min(len(user_ids), len(labels))):
                labels[i] = -100

            return {
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            }

    return SFTDataset([format_example(p) for p in pairs])


def collate(batch, pad_id: int):
    import torch
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids, labels, attn = [], [], []
    for x in batch:
        pad = max_len - x["input_ids"].size(0)
        input_ids.append(torch.cat([x["input_ids"],
                                    torch.full((pad,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([x["labels"],
                                 torch.full((pad,), -100, dtype=torch.long)]))
        attn.append(torch.cat([x["attention_mask"],
                               torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn),
    }


def train(base: str, pairs: list[dict], rank: int, epochs: int,
          out_path: Path, dry_run: bool = False, quant: str = "nf4"):
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              BitsAndBytesConfig, TrainingArguments, Trainer)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    log.info(f"Loading tokenizer: {base}")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log.info(f"Building dataset from {len(pairs)} pairs")
    ds = build_dataset(pairs, tok)
    if dry_run:
        sample = ds[0]
        log.info(f"Sample input_ids shape: {sample['input_ids'].shape}, "
                 f"first 20: {sample['input_ids'][:20].tolist()}")
        log.info("Dry run — not loading model.")
        return

    if quant == "none":
        log.info(f"Loading base model: {base} (bf16, no quantization)")
        model = AutoModelForCausalLM.from_pretrained(
            base, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
    else:
        log.info(f"Loading base model: {base} (4-bit {quant})")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    out_path.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    log.info("Starting training")
    trainer.train()

    log.info(f"Saving adapter to {out_path}")
    model.save_pretrained(str(out_path))
    tok.save_pretrained(str(out_path))
    # Write a metadata JSON so the loader can reconstruct things
    (out_path / "erebus_meta.json").write_text(json.dumps({
        "base_model": base,
        "rank": rank,
        "epochs": epochs,
        "n_pairs": len(pairs),
        "trained_at": datetime.now().isoformat(),
    }, indent=2))
    log.info("Training complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--day", help="YYYY-MM-DD; default: all days")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--min-pairs", type=int, default=5,
                    help="Skip training if fewer pairs than this")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quant", default="nf4", choices=["nf4", "fp4", "none"],
                    help="Quantization: nf4/fp4 (4-bit, needs Ampere+) or none (bf16, works on Volta)")
    ap.add_argument("--out")
    args = ap.parse_args()

    pairs = load_training_pairs(args.day)
    log.info(f"Loaded {len(pairs)} training pairs")
    if len(pairs) < args.min_pairs:
        log.warning(f"Only {len(pairs)} pairs (< {args.min_pairs}). Skipping.")
        sys.exit(0)

    tag = datetime.now().strftime("%Y%m%d")
    out_path = Path(args.out) if args.out else OUT_DIR / f"lora_{tag}"
    train(args.base, pairs, args.rank, args.epochs, out_path,
          dry_run=args.dry_run, quant=args.quant)


if __name__ == "__main__":
    main()
