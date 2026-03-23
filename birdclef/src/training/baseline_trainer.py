"""Phase 1: Standard baseline classifier training."""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from ..data.augmentations import mixup
from ..evaluation.metrics import competition_metric
from .losses import FocalBCELoss


def train_baseline(classifier, train_dataset, val_dataset, config, device="cuda:0"):
    """Train baseline classifier with standard augmentations.

    Args:
        classifier: BirdClassifier model
        train_dataset: BirdCLEFTrainDataset
        val_dataset: BirdCLEFTrainDataset (no augmentation)
        config: dict with training hyperparameters
        device: CUDA device
    """
    classifier = classifier.to(device)

    # Data loaders
    from ..data.dataset import make_balanced_sampler
    sampler = make_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        sampler=sampler, num_workers=config.get("num_workers", 8),
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config.get("num_workers", 8),
        pin_memory=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.get("T_0", 5), T_mult=config.get("T_mult", 2),
    )
    criterion = FocalBCELoss(gamma=config.get("focal_gamma", 2.0))
    scaler = GradScaler()

    output_dir = config.get("output_dir", "output/baseline")
    os.makedirs(output_dir, exist_ok=True)

    best_metric = 0.0
    patience_counter = 0
    patience = config.get("early_stopping_patience", 5)
    mixup_alpha = config.get("mixup_alpha", 0.4)
    mixup_prob = config.get("mixup_prob", 0.5)

    for epoch in range(config["epochs"]):
        # === Train ===
        classifier.train()
        train_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for specs, labels in train_loader:
            specs = specs.to(device)
            labels = labels.to(device)

            # Mixup augmentation
            if torch.rand(1).item() < mixup_prob:
                specs, labels = mixup(specs, labels, alpha=mixup_alpha)

            optimizer.zero_grad()
            with autocast():
                logits = classifier(specs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1

            if n_batches % config.get("log_every", 50) == 0:
                print(f"  [{n_batches}/{len(train_loader)}] loss={loss.item():.4f}")

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # === Validate ===
        val_metric = validate(classifier, val_loader, device)

        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"loss={avg_train_loss:.4f} | "
              f"val_auc={val_metric:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"{elapsed:.0f}s")

        # Checkpointing
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metric": best_metric,
            }, os.path.join(output_dir, "best_model.pt"))
            print(f"  -> New best: {best_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric": val_metric,
        }, os.path.join(output_dir, "latest_model.pt"))

    print(f"Training complete. Best val AUC: {best_metric:.4f}")
    return best_metric


@torch.no_grad()
def validate(classifier, val_loader, device):
    """Run validation and return competition metric."""
    classifier.eval()
    all_preds = []
    all_labels = []

    for specs, labels in val_loader:
        specs = specs.to(device)
        with autocast():
            logits = classifier(specs)
        probs = torch.sigmoid(logits).cpu()
        all_preds.append(probs)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return competition_metric(all_labels, all_preds)
