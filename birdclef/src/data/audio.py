"""Audio loading and mel spectrogram extraction for BirdCLEF."""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

SAMPLE_RATE = 32000
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


def load_audio(path, sr=SAMPLE_RATE):
    """Load audio file and resample to target rate."""
    waveform, orig_sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    return waveform.squeeze(0)  # (samples,)


def make_mel_transform(sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       n_mels=N_MELS):
    """Create mel spectrogram + amplitude-to-dB transform."""
    mel = T.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0,
    )
    db = T.AmplitudeToDB(stype="power", top_db=80)
    return torch.nn.Sequential(mel, db)


_mel_transform = None


def get_mel_transform():
    global _mel_transform
    if _mel_transform is None:
        _mel_transform = make_mel_transform()
    return _mel_transform


def audio_to_mel(waveform, mel_transform=None):
    """Convert waveform to normalized mel spectrogram.

    Args:
        waveform: (samples,) tensor
    Returns:
        (1, n_mels, time_frames) tensor, normalized to ~zero mean unit var
    """
    if mel_transform is None:
        mel_transform = get_mel_transform()
    spec = mel_transform(waveform.unsqueeze(0))  # (1, n_mels, T)
    # Normalize per-spectrogram
    mean = spec.mean()
    std = spec.std() + 1e-6
    spec = (spec - mean) / std
    return spec


def extract_window(waveform, offset=None):
    """Extract a 5-second window from waveform.

    Args:
        waveform: (samples,) tensor
        offset: sample offset, or None for random
    Returns:
        (WINDOW_SAMPLES,) tensor
    """
    n = waveform.shape[0]
    if n >= WINDOW_SAMPLES:
        if offset is None:
            offset = torch.randint(0, n - WINDOW_SAMPLES + 1, (1,)).item()
        return waveform[offset:offset + WINDOW_SAMPLES]
    else:
        # Pad short clips with silence
        pad = torch.zeros(WINDOW_SAMPLES)
        if offset is None:
            offset = torch.randint(0, WINDOW_SAMPLES - n + 1, (1,)).item()
        pad[offset:offset + n] = waveform
        return pad


def precompute_spectrograms(audio_path, output_path):
    """Precompute mel spectrograms for a single audio file.

    Splits into non-overlapping 5-second windows and saves as .npy.
    Returns list of saved file paths.
    """
    mel_transform = make_mel_transform()
    waveform = load_audio(audio_path)
    n = waveform.shape[0]

    saved = []
    for i in range(0, n, WINDOW_SAMPLES):
        chunk = waveform[i:i + WINDOW_SAMPLES]
        if chunk.shape[0] < WINDOW_SAMPLES // 2:
            continue  # Skip very short trailing chunks
        if chunk.shape[0] < WINDOW_SAMPLES:
            pad = torch.zeros(WINDOW_SAMPLES)
            pad[:chunk.shape[0]] = chunk
            chunk = pad
        spec = audio_to_mel(chunk, mel_transform)
        out = f"{output_path}_{i // WINDOW_SAMPLES}.npy"
        np.save(out, spec.numpy())
        saved.append(out)
    return saved
