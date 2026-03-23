"""BirdCLEF 2026 — Phase 2: Adversarial Augmentation Training.

Uses two GPUs:
  GPU 0: Classifier (EfficientNet-B3)
  GPU 1: Adversarial Generator (U-Net)
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.data.dataset import BirdCLEFTrainDataset, build_species_list
from src.models.classifier import BirdClassifier
from src.models.generator import AdversarialGenerator
from src.training.adversarial_trainer import train_adversarial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adversarial.yaml")
    parser.add_argument("--classifier-checkpoint", required=True,
                        help="Path to Phase 1 best_model.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=== BirdCLEF 2026 — Phase 2: Adversarial Training ===")
    print(f"Config: {args.config}")
    print(f"Classifier checkpoint: {args.classifier_checkpoint}")

    assert torch.cuda.device_count() >= 2, \
        f"Need 2 GPUs, found {torch.cuda.device_count()}"
    print(f"GPU 0: {torch.cuda.get_device_name(0)} (Classifier)")
    print(f"GPU 1: {torch.cuda.get_device_name(1)} (Generator)")

    # Build species list
    species_list = build_species_list(config["data"]["taxonomy_csv"])
    num_classes = len(species_list)
    print(f"Species: {num_classes}")

    # Datasets
    train_dataset = BirdCLEFTrainDataset(
        train_csv=config["data"]["train_csv"],
        audio_dir=config["data"]["audio_dir"],
        species_list=species_list,
        precomputed_dir=config["data"].get("precomputed_dir"),
        augment=True,
    )
    val_dataset = BirdCLEFTrainDataset(
        train_csv=config["data"]["train_csv"],
        audio_dir=config["data"]["audio_dir"],
        species_list=species_list,
        precomputed_dir=config["data"].get("precomputed_dir"),
        augment=False,
    )

    n = len(train_dataset)
    n_val = int(n * 0.1)
    n_train = n - n_val
    train_dataset.samples = train_dataset.samples[:n_train]
    val_dataset.samples = val_dataset.samples[n_train:]
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Load pre-trained classifier from Phase 1
    classifier = BirdClassifier(
        num_classes=num_classes,
        backbone="tf_efficientnet_b3_ns",
        pretrained=False,
        dropout=0.3,
    )
    ckpt = torch.load(args.classifier_checkpoint, map_location="cpu")
    classifier.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded classifier from epoch {ckpt['epoch']} "
          f"(val_auc={ckpt['metric']:.4f})")

    # Create generator
    gen_config = config.get("generator", {})
    generator = AdversarialGenerator(
        channels=gen_config.get("channels", [32, 64, 128, 256]),
        epsilon=gen_config.get("epsilon", 0.15),
    )

    best = train_adversarial(classifier, generator, train_dataset,
                             val_dataset, config["training"])
    print(f"Done. Best validation AUC: {best:.4f}")


if __name__ == "__main__":
    main()
