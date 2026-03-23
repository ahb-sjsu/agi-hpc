"""Spectrogram augmentations for BirdCLEF training."""

import torch
import torch.nn.functional as F


class SpecAugment:
    """Time and frequency masking on mel spectrograms."""

    def __init__(self, time_mask_width=40, time_mask_count=2,
                 freq_mask_width=16, freq_mask_count=2):
        self.time_mask_width = time_mask_width
        self.time_mask_count = time_mask_count
        self.freq_mask_width = freq_mask_width
        self.freq_mask_count = freq_mask_count

    def __call__(self, spec):
        """spec: (1, n_mels, T)"""
        _, n_mels, T = spec.shape
        for _ in range(self.time_mask_count):
            w = torch.randint(1, self.time_mask_width + 1, (1,)).item()
            t0 = torch.randint(0, max(T - w, 1), (1,)).item()
            spec[:, :, t0:t0 + w] = 0
        for _ in range(self.freq_mask_count):
            w = torch.randint(1, self.freq_mask_width + 1, (1,)).item()
            f0 = torch.randint(0, max(n_mels - w, 1), (1,)).item()
            spec[:, f0:f0 + w, :] = 0
        return spec


class TimeShift:
    """Random circular shift along time axis."""

    def __init__(self, max_shift=50):
        self.max_shift = max_shift

    def __call__(self, spec):
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(spec, shifts=shift, dims=-1)


class GaussianNoise:
    """Additive Gaussian noise on spectrogram."""

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, spec):
        return spec + torch.randn_like(spec) * self.std


class RandomGain:
    """Random amplitude scaling."""

    def __init__(self, low=0.8, high=1.2):
        self.low = low
        self.high = high

    def __call__(self, spec):
        gain = torch.empty(1).uniform_(self.low, self.high).item()
        return spec * gain


class Cutout:
    """Random rectangular patches zeroed out."""

    def __init__(self, n_patches=3, patch_size=16):
        self.n_patches = n_patches
        self.patch_size = patch_size

    def __call__(self, spec):
        _, h, w = spec.shape
        for _ in range(self.n_patches):
            y = torch.randint(0, max(h - self.patch_size, 1), (1,)).item()
            x = torch.randint(0, max(w - self.patch_size, 1), (1,)).item()
            spec[:, y:y + self.patch_size, x:x + self.patch_size] = 0
        return spec


class TrainAugmentor:
    """Composite augmentor for training."""

    def __init__(self):
        self.spec_augment = SpecAugment()
        self.time_shift = TimeShift()
        self.noise = GaussianNoise()
        self.gain = RandomGain()
        self.cutout = Cutout()

    def __call__(self, spec):
        spec = self.spec_augment(spec)
        if torch.rand(1).item() < 0.5:
            spec = self.time_shift(spec)
        if torch.rand(1).item() < 0.3:
            spec = self.noise(spec)
        if torch.rand(1).item() < 0.3:
            spec = self.gain(spec)
        if torch.rand(1).item() < 0.3:
            spec = self.cutout(spec)
        return spec


def mixup(specs, labels, alpha=0.4):
    """Mixup augmentation on a batch.

    Args:
        specs: (B, 1, n_mels, T)
        labels: (B, num_classes)
        alpha: Beta distribution parameter
    Returns:
        mixed_specs, mixed_labels
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(specs.size(0))
    mixed_specs = lam * specs + (1 - lam) * specs[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed_specs, mixed_labels
