#!/usr/bin/env python3
"""
Fine-tune Gemma 4 E4B on ethics corpora using Unsloth QLoRA.

Targets the Gemma 4 Good competition (Kaggle) / Unsloth special track.
Trains on GPU 1 to avoid disrupting inference on GPU 0.

Hardware: Quadro GV100 32GB (Volta, compute 7.0)
  - No bf16 support -- uses fp16
  - 32GB VRAM sufficient for 4-bit E4B (~5GB) + LoRA overhead

Copyright (c) 2024 AGI-HPC Project
Author: Andrew H. Bond (agi.hpc@gmail.com)
"""
from __future__ import annotations

import json
import logging
import os
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration -----------------------------------------------------------
MODEL_NAME = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"  # Pre-quantized 4-bit
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

DATA_PATH = "/home/claude/ethics_finetune_data.jsonl"
OUTPUT_DIR = "/home/claude/models/gemma4-ethics-lora"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Force GPU 1 (GPU 0 runs Spock / Gemma 4 31B inference)
DEVICE_INDEX = 1


def format_prompt(example: dict) -> str:
    """Format an example into the Gemma chat template."""
    instruction = example["instruction"]
    output = example["output"]

    # Use Gemma chat format
    return (
        "<start_of_turn>user\n"
        f"{instruction}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{output}"
        "<end_of_turn>"
    )


def load_dataset_from_jsonl(path: str) -> list[dict]:
    """Load and format the ethics JSONL dataset."""
    examples = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line.strip())
            examples.append({"text": format_prompt(row)})
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return examples


def main() -> None:
    logger.info("=" * 60)
    logger.info("Gemma 4 E4B Ethics Fine-tuning (Unsloth QLoRA)")
    logger.info("=" * 60)

    # --- Validate environment -------------------------------------------------
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Aborting.")
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    logger.info(f"CUDA devices: {gpu_count}")
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem // (1024 ** 3)
        logger.info(f"  GPU {i}: {name} ({mem}GB)")

    # Check GPU 1 memory
    torch.cuda.set_device(DEVICE_INDEX)
    free_mem = torch.cuda.mem_get_info(DEVICE_INDEX)[0] / (1024 ** 3)
    logger.info(f"GPU {DEVICE_INDEX} free memory: {free_mem:.1f} GB")
    if free_mem < 8:
        logger.error(
            f"GPU {DEVICE_INDEX} has only {free_mem:.1f}GB free. Need at least 8GB. "
            "Is Qwen still running? Switch to train mode first."
        )
        sys.exit(1)

    # --- Load model with Unsloth ----------------------------------------------
    logger.info(f"Loading model: {MODEL_NAME}")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,  # Volta does not support bf16
        load_in_4bit=True,
        device_map={"": DEVICE_INDEX},  # Force to GPU 1
    )

    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimized checkpointing
        random_state=42,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    # --- Load dataset ---------------------------------------------------------
    logger.info(f"Loading dataset from {DATA_PATH}")
    raw_examples = load_dataset_from_jsonl(DATA_PATH)

    from datasets import Dataset

    dataset = Dataset.from_list(raw_examples)
    logger.info(f"Dataset: {len(dataset)} examples")

    # --- Training -------------------------------------------------------------
    logger.info("Setting up SFTTrainer...")
    from trl import SFTTrainer, SFTConfig

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    training_args = SFTConfig(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        fp16=True,   # Volta: use fp16, not bf16
        bf16=False,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        seed=42,
        report_to="none",  # No wandb/tensorboard for now
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # --- Train ----------------------------------------------------------------
    logger.info("Starting training...")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
    logger.info(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")

    total_steps = (len(dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)) * NUM_EPOCHS
    logger.info(f"  Total training steps: ~{total_steps}")

    stats = trainer.train()

    logger.info("Training complete!")
    logger.info(f"  Total steps: {stats.global_step}")
    logger.info(f"  Training loss: {stats.training_loss:.4f}")

    # --- Save -----------------------------------------------------------------
    logger.info(f"Saving LoRA adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Also save merged model for easy loading (optional, uses more disk)
    # Uncomment if you want a merged checkpoint:
    # logger.info("Saving merged model (16-bit)...")
    # model.save_pretrained_merged(
    #     os.path.join(OUTPUT_DIR, "merged_16bit"),
    #     tokenizer,
    #     save_method="merged_16bit",
    # )

    # Save GGUF for llama.cpp / Ollama deployment (optional)
    # Uncomment if needed:
    # logger.info("Saving GGUF Q4_K_M...")
    # model.save_pretrained_gguf(
    #     os.path.join(OUTPUT_DIR, "gguf"),
    #     tokenizer,
    #     quantization_method="q4_k_m",
    # )

    logger.info("=" * 60)
    logger.info("Fine-tuning complete! Adapter saved to:")
    logger.info(f"  {OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
