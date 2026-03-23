"""Geometric feature extraction for BirdCLEF — SPD manifold + TDA.

Based on Bond (2026), "Geometric Methods in Computational Modeling":
  - Ch. 4.6: SPD covariance features from mel spectrograms
  - Ch. 5: Topological data analysis via Takens embedding

These features are CPU-only, no GPU needed — perfect for the 90-min
CPU submission constraint. They capture structural properties (harmonic
correlations, periodicity topology) that are invariant to amplitude
scaling, noise, and domain shift.
"""

import numpy as np
from scipy.linalg import logm, eigvalsh
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# SPD Manifold Features (Chapter 4)
# ═══════════════════════════════════════════════════════════════

def compute_covariance(spectrogram: np.ndarray, n_bands: int = 16,
                       epsilon: float = 1e-4) -> np.ndarray:
    """Compute frequency-band covariance matrix from mel spectrogram.

    Groups mel bins into n_bands frequency bands and computes the
    covariance matrix, capturing cross-frequency correlations that
    flat spectrograms miss. (Bond 2026a, §4.6)

    Args:
        spectrogram: (n_mels, T) mel spectrogram (log-scaled)
        n_bands: number of frequency bands (default 16 → 16×16 SPD matrix)
        epsilon: regularization for positive-definiteness

    Returns:
        (n_bands, n_bands) symmetric positive-definite covariance matrix
    """
    n_mels, T = spectrogram.shape
    band_size = n_mels // n_bands

    # Average mel bins within each band
    bands = np.zeros((n_bands, T))
    for i in range(n_bands):
        start = i * band_size
        end = start + band_size if i < n_bands - 1 else n_mels
        bands[i] = spectrogram[start:end].mean(axis=0)

    # Covariance matrix + regularization
    cov = np.cov(bands)  # (n_bands, n_bands)
    cov += epsilon * np.eye(n_bands)
    return cov


def spd_features_from_spectrogram(spectrogram: np.ndarray,
                                   n_bands: int = 16,
                                   epsilon: float = 1e-4) -> np.ndarray:
    """Extract SPD manifold features from mel spectrogram.

    Computes the log-Euclidean features: the upper triangle of the
    matrix logarithm of the covariance matrix. These live in a flat
    vector space where standard ML methods apply directly.

    Args:
        spectrogram: (n_mels, T) or (1, n_mels, T) mel spectrogram
        n_bands: frequency band count
        epsilon: regularization

    Returns:
        Feature vector of shape (n_bands*(n_bands+1)//2,)
        Default: 136-dimensional for n_bands=16
    """
    if spectrogram.ndim == 3:
        spectrogram = spectrogram[0]

    cov = compute_covariance(spectrogram, n_bands, epsilon)

    # Matrix logarithm → log-Euclidean space
    log_cov = logm(cov).real

    # Extract upper triangle (including diagonal)
    indices = np.triu_indices(n_bands)
    features = log_cov[indices]

    return features.astype(np.float32)


def spd_distance(cov1: np.ndarray, cov2: np.ndarray) -> float:
    """Log-Euclidean distance between two SPD matrices.

    d_LE(S1, S2) = ||log(S1) - log(S2)||_F

    This is a proper metric on SPD(n) that respects the manifold
    geometry. (Bond 2026a, §4.2)
    """
    diff = logm(cov1).real - logm(cov2).real
    return np.linalg.norm(diff, 'fro')


def spectral_trajectory(spectrogram: np.ndarray, n_bands: int = 16,
                         window: int = 64, hop: int = 32,
                         epsilon: float = 1e-4) -> dict:
    """Compute spectral trajectory on the SPD manifold.

    Slides a window across time, computing covariance at each position,
    then measures the geodesic deviation — how "straight" the trajectory
    is on the manifold. (Bond 2026a, §4.5)

    Args:
        spectrogram: (n_mels, T) mel spectrogram
        n_bands: frequency band count
        window: window size in frames
        hop: hop size in frames

    Returns:
        dict with trajectory statistics:
          - path_length: total log-Euclidean distance traveled
          - geodesic_distance: direct distance between endpoints
          - deviation: path_length - geodesic_distance (≥ 0)
          - n_steps: number of trajectory points
    """
    if spectrogram.ndim == 3:
        spectrogram = spectrogram[0]

    n_mels, T = spectrogram.shape
    covs = []
    for t in range(0, T - window, hop):
        chunk = spectrogram[:, t:t + window]
        if chunk.shape[1] >= n_bands * 3:  # need enough frames
            covs.append(compute_covariance(chunk, n_bands, epsilon))

    if len(covs) < 2:
        return {"path_length": 0.0, "geodesic_distance": 0.0,
                "deviation": 0.0, "n_steps": len(covs)}

    # Path length: sum of consecutive log-Euclidean distances
    path_length = sum(spd_distance(covs[i], covs[i + 1])
                      for i in range(len(covs) - 1))

    # Geodesic distance: direct distance between endpoints
    geo_dist = spd_distance(covs[0], covs[-1])

    return {
        "path_length": float(path_length),
        "geodesic_distance": float(geo_dist),
        "deviation": float(path_length - geo_dist),
        "n_steps": len(covs),
    }


# ═══════════════════════════════════════════════════════════════
# Topological Data Analysis (Chapter 5)
# ═══════════════════════════════════════════════════════════════

def time_delay_embedding(signal: np.ndarray, delay: int = 10,
                          dim: int = 3) -> np.ndarray:
    """Takens' time-delay embedding: reconstruct attractor from 1D series.

    Given a signal x(t), constructs vectors:
        v(t) = [x(t), x(t+τ), x(t+2τ), ..., x(t+(d-1)τ)]

    By Takens' theorem, for generic τ and d ≥ 2m+1 (m = attractor
    dimension), this reconstructs the topology of the original
    dynamical system. (Bond 2026a, §5.2)
    """
    n = len(signal) - (dim - 1) * delay
    if n <= 0:
        return np.zeros((1, dim))
    embedded = np.empty((n, dim))
    for d in range(dim):
        embedded[:, d] = signal[d * delay: d * delay + n]
    return embedded


def subsample_cloud(cloud: np.ndarray, max_points: int = 1000,
                     seed: Optional[int] = None) -> np.ndarray:
    """Subsample point cloud for computational tractability."""
    if len(cloud) <= max_points:
        return cloud
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(cloud), max_points, replace=False)
    return cloud[idx]


def _persistence_diagrams_ripser(cloud: np.ndarray,
                                  max_dim: int = 1,
                                  thresh: float = 2.0):
    """Compute persistence diagrams using ripser."""
    try:
        from ripser import ripser
        result = ripser(cloud, maxdim=max_dim, thresh=thresh)
        return result['dgms']
    except ImportError:
        return _persistence_diagrams_naive(cloud, max_dim)


def _persistence_diagrams_naive(cloud: np.ndarray,
                                 max_dim: int = 1):
    """Naive H0 persistence (no ripser dependency).

    Computes connected components persistence using single-linkage
    clustering. H1 features are set to empty (requires ripser).
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import single, fcluster

    dists = pdist(cloud)
    Z = single(dists)

    # H0: birth=0, death=merge distance
    h0 = np.column_stack([np.zeros(len(Z)), Z[:, 2]])

    diagrams = [h0]
    for d in range(1, max_dim + 1):
        diagrams.append(np.empty((0, 2)))

    return diagrams


def _diagram_features(dgm: np.ndarray) -> np.ndarray:
    """Extract 8 summary statistics from a persistence diagram.

    Returns: [count, mean_lifetime, std_lifetime, max_lifetime,
              p75_lifetime, mean_birth, total_persistence,
              normalized_persistence]
    (Bond 2026a, §5.5)
    """
    if len(dgm) == 0:
        return np.zeros(8, dtype=np.float32)

    finite_mask = np.isfinite(dgm[:, 1])
    finite = dgm[finite_mask]

    if len(finite) == 0:
        return np.zeros(8, dtype=np.float32)

    lifetimes = finite[:, 1] - finite[:, 0]
    n = len(lifetimes)
    total = float(np.sum(lifetimes ** 2))

    return np.array([
        n,
        lifetimes.mean(),
        lifetimes.std() if n > 1 else 0.0,
        lifetimes.max(),
        np.percentile(lifetimes, 75),
        finite[:, 0].mean(),
        total,
        np.sqrt(total) / (n + 1e-10),
    ], dtype=np.float32)


def tda_features(signal: np.ndarray, delay: int = 10, dim: int = 3,
                  max_points: int = 1000, max_homology_dim: int = 1,
                  thresh: float = 2.0, seed: int = 42) -> np.ndarray:
    """Extract topological features from an audio signal.

    Full pipeline: signal → Takens embedding → subsample →
    persistent homology → feature vector. (Bond 2026a, §5.6)

    Args:
        signal: 1D audio waveform (numpy array)
        delay: time delay τ for embedding
        dim: embedding dimension d
        max_points: max points for persistence computation
        max_homology_dim: max homology dimension (0=components, 1=loops)
        thresh: Rips complex threshold
        seed: random seed for subsampling

    Returns:
        Feature vector of shape (8 * (max_homology_dim + 1),)
        Default: 16-dimensional (8 for H0 + 8 for H1)
    """
    # Normalize signal
    signal = signal.astype(np.float64)
    signal = (signal - signal.mean()) / (signal.std() + 1e-10)

    # Time-delay embedding
    cloud = time_delay_embedding(signal, delay=delay, dim=dim)

    # Subsample for tractability
    cloud = subsample_cloud(cloud, max_points=max_points, seed=seed)

    # Normalize point cloud
    cloud = (cloud - cloud.mean(axis=0)) / (cloud.std(axis=0) + 1e-10)

    # Persistent homology
    diagrams = _persistence_diagrams_ripser(cloud, max_dim=max_homology_dim,
                                             thresh=thresh)

    # Extract features per homology dimension
    features = []
    for d in range(max_homology_dim + 1):
        if d < len(diagrams):
            features.append(_diagram_features(diagrams[d]))
        else:
            features.append(np.zeros(8, dtype=np.float32))

    return np.concatenate(features)


# ═══════════════════════════════════════════════════════════════
# Combined Feature Extraction
# ═══════════════════════════════════════════════════════════════

def extract_geometric_features(waveform: np.ndarray,
                                spectrogram: np.ndarray,
                                n_bands: int = 16,
                                tda_delay: int = 10,
                                tda_dim: int = 3,
                                tda_max_points: int = 500) -> np.ndarray:
    """Extract combined SPD + TDA features from audio.

    Args:
        waveform: raw audio signal (1D numpy array)
        spectrogram: mel spectrogram (n_mels, T) or (1, n_mels, T)
        n_bands: frequency bands for SPD features
        tda_delay: time delay for Takens embedding
        tda_dim: embedding dimension
        tda_max_points: max points for TDA

    Returns:
        Combined feature vector:
          - 136 SPD features (16×16 upper triangle)
          - 4 trajectory features (path_length, geo_dist, deviation, n_steps)
          - 16 TDA features (8×H0 + 8×H1)
          Total: 156-dimensional
    """
    # SPD features from spectrogram
    spd = spd_features_from_spectrogram(spectrogram, n_bands=n_bands)

    # Spectral trajectory
    if spectrogram.ndim == 3:
        spec2d = spectrogram[0]
    else:
        spec2d = spectrogram
    traj = spectral_trajectory(spec2d, n_bands=n_bands)
    traj_vec = np.array([
        traj["path_length"], traj["geodesic_distance"],
        traj["deviation"], traj["n_steps"],
    ], dtype=np.float32)

    # TDA features from waveform
    tda = tda_features(waveform, delay=tda_delay, dim=tda_dim,
                       max_points=tda_max_points)

    return np.concatenate([spd, traj_vec, tda])
