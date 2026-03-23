"""BirdCLEF 2026 — Train ensemble combining CNN + geometric features.

The geometric features (SPD + TDA) provide orthogonal signal to the CNN:
  - CNN: learns visual patterns from spectrograms
  - SPD: captures cross-frequency correlations (harmonic structure)
  - TDA: captures topological structure (periodicity, call dynamics)

The ensemble is a simple logistic regression — lightweight enough for
CPU-only inference within the 90-minute Kaggle constraint.
"""

import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.data.dataset import build_species_list


def load_geometric_features(feature_dir, train_df, species_list):
    """Load precomputed geometric features and align with labels."""
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    features = []
    labels = []
    indices = []

    for i, row in train_df.iterrows():
        label = str(row["primary_label"])
        if label not in species_to_idx:
            continue

        filename = row["filename"]
        stem = Path(filename).stem
        feat_path = Path(feature_dir) / label / f"{stem}.npy"

        if feat_path.exists():
            feat = np.load(str(feat_path))
            features.append(feat)
            labels.append(species_to_idx[label])
            indices.append(i)

    return np.array(features), np.array(labels), indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--feature-dir", default="precomputed/geometric")
    parser.add_argument("--output", default="output/ensemble")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== BirdCLEF 2026 — Geometric Ensemble Training ===")

    # Species list
    species_list = build_species_list(
        sample_submission_csv=str(data_dir / "sample_submission.csv"),
    )
    num_classes = len(species_list)
    print(f"Species: {num_classes}")

    # Load features
    train_df = pd.read_csv(data_dir / "train.csv")
    features, labels, indices = load_geometric_features(
        args.feature_dir, train_df, species_list,
    )
    print(f"Loaded {len(features)} samples, feature dim: {features.shape[1]}")

    # Replace NaN/Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Train/val split (deterministic)
    np.random.seed(42)
    perm = np.random.permutation(len(features))
    n_val = int(len(features) * 0.1)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Train logistic regression (one-vs-rest for multi-class)
    print("Training logistic regression...")
    clf = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        multi_class="multinomial", n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    val_probs = clf.predict_proba(X_val)

    # Compute macro AUC (skip classes with no positives)
    aucs = []
    for i in range(num_classes):
        mask = y_val == i
        if mask.sum() > 0 and mask.sum() < len(mask):
            auc = roc_auc_score((y_val == i).astype(int), val_probs[:, i])
            aucs.append(auc)
    macro_auc = np.mean(aucs) if aucs else 0.0

    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy: {val_acc:.4f}")
    print(f"Val macro AUC: {macro_auc:.4f}")
    print(f"Classes with AUC: {len(aucs)}/{num_classes}")

    # Save model and scaler
    with open(output_dir / "geo_classifier.pkl", "wb") as f:
        pickle.dump({"classifier": clf, "scaler": scaler,
                      "species_list": species_list,
                      "feature_dim": features.shape[1],
                      "val_auc": macro_auc}, f)
    print(f"Saved to {output_dir / 'geo_classifier.pkl'}")

    # Feature importance analysis
    print("\n=== Feature Group Importance ===")
    # Ablation: zero out each feature group and measure AUC drop
    groups = {
        "SPD (covariance)": slice(0, 136),
        "Trajectory": slice(136, 140),
        "TDA (topology)": slice(140, 156),
    }
    for name, sl in groups.items():
        X_ablated = X_val.copy()
        X_ablated[:, sl] = 0
        ablated_probs = clf.predict_proba(X_ablated)
        ablated_aucs = []
        for i in range(num_classes):
            mask = y_val == i
            if mask.sum() > 0 and mask.sum() < len(mask):
                auc = roc_auc_score((y_val == i).astype(int), ablated_probs[:, i])
                ablated_aucs.append(auc)
        ablated_auc = np.mean(ablated_aucs) if ablated_aucs else 0.0
        drop = macro_auc - ablated_auc
        print(f"  {name:25s}: AUC drop = {drop:+.4f} "
              f"(from {macro_auc:.4f} to {ablated_auc:.4f})")


if __name__ == "__main__":
    main()
