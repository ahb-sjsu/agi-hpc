"""Nemotron 3 Reasoning Challenge — Baseline LoRA Fine-tuning
NVIDIA Competition | $106K + DGX Sparks

Run on Kaggle notebook with GPU enabled.
Add competition data and model as inputs before running.
"""

# ═══════════════════════════════════════════════════════════════
# CELL 1: Setup & Discover Inputs
# ═══════════════════════════════════════════════════════════════

!pip install -q transformers peft accelerate bitsandbytes datasets trl

import os, json, time, random
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch: {torch.__version__}")
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
else:
    print("WARNING: No GPU detected. Enable GPU in notebook settings.")

print("\nAvailable inputs:")
for root, dirs, files in os.walk("/kaggle/input"):
    for d in dirs:
        print(f"  DIR:  {os.path.join(root, d)}")
    for f in files[:5]:
        print(f"  FILE: {os.path.join(root, f)}")

# ═══════════════════════════════════════════════════════════════
# CELL 2: Load Data
# ═══════════════════════════════════════════════════════════════

DATA_DIR = "/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge"
if not os.path.exists(DATA_DIR):
    print(f"Path not found: {DATA_DIR}")
    print("Searching for train.csv...")
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f == "train.csv":
                DATA_DIR = root
                print(f"Found at: {DATA_DIR}")
                break

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print(f"Columns: {list(train_df.columns)}")

# ═══════════════════════════════════════════════════════════════
# CELL 3: Format Training Data
# ═══════════════════════════════════════════════════════════════

def format_prompt(prompt, answer=None):
    text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}\n\n"
        f"Think step by step. Put your final answer in \\boxed{{}}.\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if answer is not None:
        text += f"After analyzing the pattern, the answer is:\n\n\\boxed{{{answer}}}\n<|eot_id|>"
    return text

formatted = []
for _, row in train_df.iterrows():
    formatted.append({
        "text": format_prompt(row["prompt"], row["answer"]),
        "prompt": row["prompt"],
        "answer": str(row["answer"]),
    })

random.seed(42)
random.shuffle(formatted)
split_idx = int(len(formatted) * 0.9)
train_dataset = Dataset.from_list(formatted[:split_idx])
val_dataset = Dataset.from_list(formatted[split_idx:])
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ═══════════════════════════════════════════════════════════════
# CELL 4: Load Model
# ═══════════════════════════════════════════════════════════════

# Check if model is available as Kaggle input first
MODEL_NAME = None
for root, dirs, files in os.walk("/kaggle/input"):
    for d in dirs:
        if "nemotron" in d.lower() and "model" in root.lower():
            candidate = os.path.join(root, d)
            if os.path.exists(os.path.join(candidate, "config.json")):
                MODEL_NAME = candidate
                print(f"Found model as Kaggle input: {MODEL_NAME}")
                break
    if MODEL_NAME:
        break

if not MODEL_NAME:
    MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
    print(f"Using HuggingFace model: {MODEL_NAME}")
    print("If this fails with 401, add the model as a Kaggle input or login to HuggingFace")

print(f"\nLoading {MODEL_NAME} with 4-bit quantization...")
t0 = time.time()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)
print(f"Model loaded in {time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════════════
# CELL 5: Configure LoRA
# ═══════════════════════════════════════════════════════════════

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable/1e6:.1f}M / {total/1e9:.1f}B ({100*trainable/total:.2f}%)")

# ═══════════════════════════════════════════════════════════════
# CELL 6: Train
# ═══════════════════════════════════════════════════════════════

OUTPUT_DIR = "/kaggle/working/nemotron_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=True,
    max_seq_length=1024,
    dataset_text_field="text",
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

print("Starting training...")
t0 = time.time()
result = trainer.train()
print(f"Training complete in {(time.time()-t0)/60:.1f} min, loss: {result.training_loss:.4f}")

ADAPTER_DIR = f"{OUTPUT_DIR}/final_adapter"
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Adapter saved to {ADAPTER_DIR}")

# ═══════════════════════════════════════════════════════════════
# CELL 7: Evaluate
# ═══════════════════════════════════════════════════════════════

import re

def extract_boxed_answer(text):
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""

print("Evaluating on validation set...")
model.eval()
correct = 0
total = 0

for item in formatted[split_idx:split_idx+50]:
    prompt_text = format_prompt(item["prompt"])
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0, top_p=1.0, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    predicted = extract_boxed_answer(response)
    actual = item["answer"]
    total += 1
    match = predicted.strip() == actual.strip()
    if not match:
        try:
            match = abs(float(predicted) - float(actual)) / max(abs(float(actual)), 1e-10) < 0.01
        except (ValueError, ZeroDivisionError):
            pass
    if match:
        correct += 1
    elif total <= 10:
        print(f"  MISS: predicted='{predicted[:30]}' actual='{actual[:30]}'")
    if total % 10 == 0:
        print(f"  [{total}/50] accuracy: {correct/total:.0%}")

print(f"\nValidation accuracy: {correct}/{total} ({correct/total:.1%})")

# ═══════════════════════════════════════════════════════════════
# CELL 8: Package Submission
# ═══════════════════════════════════════════════════════════════

import shutil

SUBMISSION_DIR = "/kaggle/working/submission_adapter"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
for f in os.listdir(ADAPTER_DIR):
    shutil.copy2(f"{ADAPTER_DIR}/{f}", f"{SUBMISSION_DIR}/{f}")

assert os.path.exists(f"{SUBMISSION_DIR}/adapter_config.json"), "Missing adapter_config.json!"

shutil.make_archive("/kaggle/working/submission", "zip", SUBMISSION_DIR)
print("submission.zip created")
for f in os.listdir(SUBMISSION_DIR):
    print(f"  {f} ({os.path.getsize(f'{SUBMISSION_DIR}/{f}')/1e6:.1f} MB)")
