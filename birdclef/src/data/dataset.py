"""BirdCLEF 2026 dataset classes."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from pathlib import Path

from .audio import load_audio, extract_window, audio_to_mel, make_mel_transform
from .augmentations import TrainAugmentor


def build_species_list(taxonomy_csv=None, sample_submission_csv=None):
    """Build sorted species list from submission columns (authoritative)."""
    if sample_submission_csv and os.path.exists(sample_submission_csv):
        df = pd.read_csv(sample_submission_csv, nrows=0)
        cols = [c for c in df.columns if c != "row_id"]
        return cols
    if taxonomy_csv and os.path.exists(taxonomy_csv):
        df = pd.read_csv(taxonomy_csv)
        return sorted(df["primary_label"].astype(str).unique().tolist())
    raise FileNotFoundError("Need sample_submission.csv or taxonomy.csv")


class BirdCLEFTrainDataset(Dataset):
    """Training dataset from train_audio/ individual recordings."""

    def __init__(self, train_csv, audio_dir, species_list, precomputed_dir=None,
                 augment=True):
        self.audio_dir = Path(audio_dir)
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir else None
        self.augmentor = TrainAugmentor() if augment else None
        self.mel_transform = make_mel_transform()

        df = pd.read_csv(train_csv)

        # Build sample list
        self.samples = []
        for _, row in df.iterrows():
            label = str(row["primary_label"])
            label_idx = self.species_to_idx.get(label)
            if label_idx is None:
                continue
            filename = row["filename"]  # format: "species_id/file.ogg"
            audio_path = self.audio_dir / filename
            self.samples.append({
                "audio_path": str(audio_path),
                "label": label,
                "label_idx": label_idx,
                "filename": filename,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Try precomputed spectrogram
        if self.precomputed_dir:
            stem = Path(sample["filename"]).stem
            species = sample["label"]
            npy_path = self.precomputed_dir / species / f"{stem}_0.npy"
            if npy_path.exists():
                spec = torch.from_numpy(np.load(str(npy_path)))
                if self.augmentor:
                    spec = self.augmentor(spec)
                label = torch.zeros(self.num_classes)
                label[sample["label_idx"]] = 1.0
                return spec, label

        # Load raw audio
        try:
            waveform = load_audio(sample["audio_path"])
            window = extract_window(waveform)
            spec = audio_to_mel(window, self.mel_transform)
        except Exception:
            # Return silence on error
            spec = torch.zeros(1, 128, 313)

        if self.augmentor:
            spec = self.augmentor(spec)

        label = torch.zeros(self.num_classes)
        label[sample["label_idx"]] = 1.0
        return spec, label

    def get_class_weights(self):
        """Per-sample weights for balanced sampling (sqrt balancing)."""
        class_counts = np.zeros(self.num_classes)
        for s in self.samples:
            class_counts[s["label_idx"]] += 1
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / np.sqrt(class_counts)
        return [weights[s["label_idx"]] for s in self.samples]


class BirdCLEFSoundscapeDataset(Dataset):
    """Training dataset from labeled soundscapes (multi-label, 5-sec windows)."""

    def __init__(self, soundscape_dir, labels_csv, species_list, augment=True):
        self.soundscape_dir = Path(soundscape_dir)
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.augmentor = TrainAugmentor() if augment else None
        self.mel_transform = make_mel_transform()

        df = pd.read_csv(labels_csv)
        self.samples = []
        for _, row in df.iterrows():
            labels_str = str(row["primary_label"])
            label_ids = [s.strip() for s in labels_str.split(";") if s.strip()]
            label_indices = []
            for lid in label_ids:
                idx = self.species_to_idx.get(lid)
                if idx is not None:
                    label_indices.append(idx)
            if not label_indices:
                continue

            # Parse time window
            start_parts = str(row["start"]).split(":")
            start_sec = int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + int(start_parts[2])

            self.samples.append({
                "filename": row["filename"],
                "start_sec": start_sec,
                "label_indices": label_indices,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = self.soundscape_dir / sample["filename"]

        try:
            waveform = load_audio(str(audio_path))
            from .audio import SAMPLE_RATE, WINDOW_SAMPLES
            offset = sample["start_sec"] * SAMPLE_RATE
            chunk = waveform[offset:offset + WINDOW_SAMPLES]
            if chunk.shape[0] < WINDOW_SAMPLES:
                pad = torch.zeros(WINDOW_SAMPLES)
                pad[:chunk.shape[0]] = chunk
                chunk = pad
            spec = audio_to_mel(chunk, self.mel_transform)
        except Exception:
            spec = torch.zeros(1, 128, 313)

        if self.augmentor:
            spec = self.augmentor(spec)

        label = torch.zeros(self.num_classes)
        for li in sample["label_indices"]:
            label[li] = 1.0
        return spec, label


class BirdCLEFTestDataset(Dataset):
    """Test dataset — chunks soundscapes into 5-second windows."""

    def __init__(self, soundscape_dir, sample_rate=32000, window_sec=5.0):
        self.mel_transform = make_mel_transform()
        self.window_samples = int(sample_rate * window_sec)
        self.window_sec = window_sec

        self.windows = []
        soundscape_dir = Path(soundscape_dir)
        for audio_file in sorted(soundscape_dir.glob("*.ogg")):
            file_id = audio_file.stem
            try:
                waveform = load_audio(str(audio_file), sr=sample_rate)
            except Exception:
                continue
            n_windows = max(1, len(waveform) // self.window_samples)
            for i in range(n_windows):
                start = i * self.window_samples
                end_sec = int((i + 1) * window_sec)
                row_id = f"{file_id}_{end_sec}"
                chunk = waveform[start:start + self.window_samples]
                if chunk.shape[0] < self.window_samples:
                    pad = torch.zeros(self.window_samples)
                    pad[:chunk.shape[0]] = chunk
                    chunk = pad
                self.windows.append({
                    "spec": audio_to_mel(chunk, self.mel_transform),
                    "row_id": row_id,
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        return w["spec"], w["row_id"]


def make_balanced_sampler(dataset):
    """Create a WeightedRandomSampler for class-balanced training."""
    weights = dataset.get_class_weights()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
