"""Phase 2: Adversarial augmentation training with two GPUs.

GPU 0: Classifier (EfficientNet-B3) — learns to identify species
GPU 1: Generator (U-Net) — learns to create perturbations that fool classifier

The generator creates realistic audio degradation (noise, overlap, distance)
while the classifier becomes robust to these perturbations.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from ..data.augmentations import mixup
from ..evaluation.metrics import competition_metric
from .losses import FocalBCELoss


def train_adversarial(classifier, generator, train_dataset, val_dataset, config):
    """Adversarial training loop across two GPUs.

    Args:
        classifier: BirdClassifier (pre-trained from Phase 1)
        generator: AdversarialGenerator
        train_dataset: BirdCLEFTrainDataset
        val_dataset: BirdCLEFTrainDataset (no augmentation)
        config: dict with training hyperparameters
    """
    dev_c = torch.device("cuda:0")  # Classifier
    dev_g = torch.device("cuda:1")  # Generator

    classifier = classifier.to(dev_c)
    generator = generator.to(dev_g)

    # Data loader
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

    # Optimizers (generator lr higher than classifier)
    opt_c = torch.optim.AdamW(
        classifier.parameters(),
        lr=config.get("classifier_lr", 2e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    opt_g = torch.optim.AdamW(
        generator.parameters(),
        lr=config.get("generator_lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_c, T_max=config["epochs"],
    )
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=config["epochs"],
    )

    criterion = FocalBCELoss(gamma=config.get("focal_gamma", 2.0))
    scaler_c = GradScaler()
    scaler_g = GradScaler()

    output_dir = config.get("output_dir", "output/adversarial")
    os.makedirs(output_dir, exist_ok=True)

    lambda_recon = config.get("lambda_recon", 10.0)
    epsilon_start = config.get("epsilon", 0.15)
    epsilon_end = config.get("epsilon_final", 0.05)
    clean_ratio = config.get("clean_adv_ratio", 0.5)

    best_metric = 0.0

    for epoch in range(config["epochs"]):
        # Anneal epsilon
        progress = epoch / max(config["epochs"] - 1, 1)
        current_epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        generator.epsilon = current_epsilon

        classifier.train()
        generator.train()

        stats = {"c_loss": 0, "g_loss": 0, "g_fool": 0, "g_recon": 0,
                 "delta_norm": 0, "n": 0}
        t0 = time.time()

        for specs, labels in train_loader:
            specs_c = specs.to(dev_c)
            specs_g = specs.to(dev_g)
            labels_c = labels.to(dev_c)
            labels_g = labels.to(dev_g)

            # ============================================
            # Step 1: Update Generator (fool classifier)
            # ============================================
            opt_g.zero_grad()

            with autocast():
                delta = generator(specs_g)
                perturbed_g = specs_g + delta

            # Transfer perturbed to classifier GPU
            perturbed_c = perturbed_g.detach().to(dev_c)
            perturbed_c.requires_grad_(True)  # Need grad for generator update

            with autocast():
                adv_logits = classifier(perturbed_c)
                # Generator wants to MAXIMIZE classifier error
                g_fool_loss = -criterion(adv_logits, labels_c)
                # But keep perturbations small (realistic)
                g_recon_loss = (delta ** 2).mean().to(dev_c)
                g_loss = g_fool_loss + lambda_recon * g_recon_loss

            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            # ============================================
            # Step 2: Update Classifier (resist adversary)
            # ============================================
            opt_c.zero_grad()

            # Regenerate perturbations (detached from generator)
            with torch.no_grad():
                delta = generator(specs_g)
                perturbed_c = (specs_g + delta).to(dev_c)

            with autocast():
                # Loss on clean data
                clean_logits = classifier(specs_c)
                clean_loss = criterion(clean_logits, labels_c)
                # Loss on adversarial data
                adv_logits = classifier(perturbed_c)
                adv_loss = criterion(adv_logits, labels_c)
                # Combined: classifier must handle both
                c_loss = clean_ratio * clean_loss + (1 - clean_ratio) * adv_loss

            scaler_c.scale(c_loss).backward()
            scaler_c.step(opt_c)
            scaler_c.update()

            # Stats
            stats["c_loss"] += c_loss.item()
            stats["g_loss"] += g_loss.item()
            stats["g_fool"] += g_fool_loss.item()
            stats["g_recon"] += g_recon_loss.item()
            stats["delta_norm"] += delta.norm(p=2).item() / delta.numel() ** 0.5
            stats["n"] += 1

            if stats["n"] % config.get("log_every", 50) == 0:
                n = stats["n"]
                print(f"  [{n}/{len(train_loader)}] "
                      f"c_loss={stats['c_loss']/n:.4f} "
                      f"g_fool={stats['g_fool']/n:.4f} "
                      f"delta={stats['delta_norm']/n:.4f}")

        sched_c.step()
        sched_g.step()

        n = max(stats["n"], 1)
        elapsed = time.time() - t0

        # === Validate classifier ===
        val_metric = validate_adversarial(classifier, val_loader, dev_c)

        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"c_loss={stats['c_loss']/n:.4f} | "
              f"g_fool={stats['g_fool']/n:.4f} | "
              f"g_recon={stats['g_recon']/n:.4f} | "
              f"eps={current_epsilon:.3f} | "
              f"val_auc={val_metric:.4f} | "
              f"{elapsed:.0f}s")

        # Save best
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save({
                "epoch": epoch,
                "classifier_state_dict": classifier.state_dict(),
                "generator_state_dict": generator.state_dict(),
                "metric": best_metric,
            }, os.path.join(output_dir, "best_model.pt"))
            print(f"  -> New best: {best_metric:.4f}")

        # Save latest
        torch.save({
            "epoch": epoch,
            "classifier_state_dict": classifier.state_dict(),
            "generator_state_dict": generator.state_dict(),
            "opt_c": opt_c.state_dict(),
            "opt_g": opt_g.state_dict(),
            "metric": val_metric,
        }, os.path.join(output_dir, "latest_model.pt"))

    print(f"Adversarial training complete. Best val AUC: {best_metric:.4f}")
    return best_metric


@torch.no_grad()
def validate_adversarial(classifier, val_loader, device):
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
