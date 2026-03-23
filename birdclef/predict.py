"""BirdCLEF 2026 — Inference and Submission Generation."""

import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.cuda.amp import autocast

from src.data.dataset import BirdCLEFTestDataset, build_species_list
from src.models.classifier import BirdClassifier


def predict(classifier, test_dataset, device, tta=True):
    """Run inference with optional test-time augmentation."""
    classifier.eval()
    predictions = {}

    with torch.no_grad():
        for spec, row_id in test_dataset:
            spec = spec.unsqueeze(0).to(device)

            with autocast():
                logits = classifier(spec)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            if tta:
                # Time-reversal TTA
                spec_rev = spec.flip(-1)
                with autocast():
                    logits_rev = classifier(spec_rev)
                    probs_rev = torch.sigmoid(logits_rev).cpu().numpy()[0]
                probs = 0.5 * probs + 0.5 * probs_rev

            predictions[row_id] = probs

    return predictions


def make_submission(predictions, species_list, output_path,
                    sample_submission_path=None):
    """Create submission CSV."""
    if sample_submission_path:
        sample = pd.read_csv(sample_submission_path)
        row_ids = sample["row_id"].tolist()
    else:
        row_ids = sorted(predictions.keys())

    rows = []
    for row_id in row_ids:
        probs = predictions.get(row_id, np.zeros(len(species_list)))
        row = {"row_id": row_id}
        for i, species in enumerate(species_list):
            row[species] = float(probs[i])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Submission saved: {output_path} ({len(df)} rows)")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--soundscape-dir", default="data/test_soundscapes")
    parser.add_argument("--taxonomy-csv", default="data/taxonomy.csv")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    species_list = build_species_list(args.taxonomy_csv)
    num_classes = len(species_list)
    print(f"Species: {num_classes}")

    # Load model
    classifier = BirdClassifier(
        num_classes=num_classes,
        backbone="tf_efficientnet_b3_ns",
        pretrained=False,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("classifier_state_dict")
    classifier.load_state_dict(state_dict)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    classifier = classifier.to(device)
    print(f"Model loaded on {device}")

    # Test dataset
    test_dataset = BirdCLEFTestDataset(args.soundscape_dir)
    print(f"Test windows: {len(test_dataset)}")

    # Predict
    predictions = predict(classifier, test_dataset, device, tta=not args.no_tta)

    # Write submission
    make_submission(predictions, species_list, args.output, args.sample_submission)


if __name__ == "__main__":
    main()
