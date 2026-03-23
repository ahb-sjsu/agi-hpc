"""BirdCLEF 2026 — Phase 1: Baseline Classifier Training."""

import argparse
import yaml
import torch
import random
import numpy as np
from pathlib import Path

from src.data.dataset import (
    BirdCLEFTrainDataset, BirdCLEFSoundscapeDataset,
    build_species_list, make_balanced_sampler,
)
from src.models.classifier import BirdClassifier
from src.training.baseline_trainer import train_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=== BirdCLEF 2026 — Phase 1: Baseline ===")

    # Seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    data_cfg = config["data"]
    data_dir = Path(data_cfg.get("data_dir", "data"))

    # Build species list from sample_submission (authoritative column order)
    species_list = build_species_list(
        sample_submission_csv=str(data_dir / "sample_submission.csv"),
        taxonomy_csv=str(data_dir / "taxonomy.csv"),
    )
    num_classes = len(species_list)
    print(f"Species: {num_classes}")

    # Build training dataset from individual recordings
    full_dataset = BirdCLEFTrainDataset(
        train_csv=str(data_dir / "train.csv"),
        audio_dir=str(data_dir / "train_audio"),
        species_list=species_list,
        precomputed_dir=data_cfg.get("precomputed_dir"),
        augment=True,
    )
    print(f"Individual recordings: {len(full_dataset)}")

    # 90/10 split (deterministic with seed)
    n = len(full_dataset)
    indices = list(range(n))
    random.shuffle(indices)
    n_val = int(n * 0.1)

    train_samples = [full_dataset.samples[i] for i in indices[n_val:]]
    val_samples = [full_dataset.samples[i] for i in indices[:n_val]]

    # Create separate train/val datasets
    train_dataset = BirdCLEFTrainDataset(
        train_csv=str(data_dir / "train.csv"),
        audio_dir=str(data_dir / "train_audio"),
        species_list=species_list,
        precomputed_dir=data_cfg.get("precomputed_dir"),
        augment=True,
    )
    train_dataset.samples = train_samples

    val_dataset = BirdCLEFTrainDataset(
        train_csv=str(data_dir / "train.csv"),
        audio_dir=str(data_dir / "train_audio"),
        species_list=species_list,
        precomputed_dir=data_cfg.get("precomputed_dir"),
        augment=False,
    )
    val_dataset.samples = val_samples

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model_cfg = config["model"]
    classifier = BirdClassifier(
        num_classes=num_classes,
        backbone=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
    )
    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"Model params: {total_params/1e6:.1f}M")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    best = train_baseline(classifier, train_dataset, val_dataset,
                          config["training"], device=device)
    print(f"\nDone. Best validation AUC: {best:.4f}")


if __name__ == "__main__":
    main()
