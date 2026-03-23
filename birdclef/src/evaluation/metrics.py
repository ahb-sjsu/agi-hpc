"""Competition evaluation metric for BirdCLEF."""

import numpy as np
from sklearn.metrics import roc_auc_score


def competition_metric(y_true, y_pred):
    """Macro-averaged ROC-AUC, skipping classes with no true positives.

    This is the official BirdCLEF 2026 evaluation metric.
    """
    aucs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
            except ValueError:
                continue
    return np.mean(aucs) if aucs else 0.0


def per_class_auc(y_true, y_pred, species_list=None, top_k=50):
    """Compute per-class AUC for diagnostic purposes."""
    results = {}
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                name = species_list[i] if species_list else str(i)
                results[name] = auc
            except ValueError:
                continue
    # Sort by AUC ascending (worst first)
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]))
    if top_k:
        return dict(list(sorted_results.items())[:top_k])
    return sorted_results
