"""BirdCLEF 2026 — Batch geometric feature extraction on HPC.

Uses 56 CPUs in parallel to extract SPD manifold + TDA features
from all training audio. No GPU needed.

Output: precomputed feature vectors for ensemble with CNN predictions.
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time

from src.data.audio import load_audio, extract_window, audio_to_mel, make_mel_transform
from src.data.geometric_features import extract_geometric_features


def process_one_file(args_tuple):
    """Process a single audio file — designed for multiprocessing."""
    audio_path, output_path, mel_transform_params = args_tuple

    try:
        # Load audio
        import torch
        import torchaudio
        waveform = load_audio(audio_path)
        window = extract_window(waveform, offset=0)  # deterministic first window

        # Mel spectrogram
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000, n_fft=2048, hop_length=512, n_mels=128, power=2.0)
        db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        spec = db(mel(window.unsqueeze(0)))  # (1, 128, T)
        spec_np = spec.numpy()

        # Extract geometric features
        features = extract_geometric_features(
            waveform=window.numpy(),
            spectrogram=spec_np,
            n_bands=16,
            tda_delay=10,
            tda_dim=3,
            tda_max_points=500,
        )

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, features)
        return audio_path, True, len(features)

    except Exception as e:
        return audio_path, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="precomputed/geometric")
    parser.add_argument("--workers", type=int, default=0,
                        help="0 = use all CPUs")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N files (0 = all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    n_workers = args.workers or cpu_count()

    print(f"=== Geometric Feature Extraction ===")
    print(f"CPUs: {n_workers}")

    # Build file list from train.csv
    train_df = pd.read_csv(data_dir / "train.csv")
    tasks = []
    for _, row in train_df.iterrows():
        filename = row["filename"]
        audio_path = str(data_dir / "train_audio" / filename)
        stem = Path(filename).stem
        label = str(row["primary_label"])
        out_path = str(output_dir / label / f"{stem}.npy")

        if not os.path.exists(out_path):
            tasks.append((audio_path, out_path, None))

    if args.limit > 0:
        tasks = tasks[:args.limit]

    print(f"Files to process: {len(tasks)}")
    if not tasks:
        print("All features already computed!")
        return

    t0 = time.time()
    success = 0
    fail = 0

    with Pool(n_workers) as pool:
        for i, (path, ok, info) in enumerate(pool.imap_unordered(process_one_file, tasks)):
            if ok:
                success += 1
            else:
                fail += 1
                if fail <= 10:
                    print(f"  FAIL: {path}: {info}")

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(tasks) - i - 1) / rate / 60
                print(f"  [{i+1}/{len(tasks)}] "
                      f"{success} ok, {fail} fail, "
                      f"{rate:.0f} files/sec, "
                      f"~{remaining:.0f} min remaining")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Success: {success}, Failed: {fail}")
    print(f"Feature dimension: 156 (136 SPD + 4 trajectory + 16 TDA)")


if __name__ == "__main__":
    main()
