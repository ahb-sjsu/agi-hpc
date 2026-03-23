"""Loss functions for BirdCLEF training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    """Binary focal loss for multi-label classification.

    Downweights easy examples, focuses on hard ones.
    Critical for BirdCLEF's extreme class imbalance.
    """

    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label — penalizes FN more than FP.

    From: https://arxiv.org/abs/2009.14119
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # Asymmetric clipping
        probs_neg = (probs + self.clip).clamp(max=1)
        # Loss components
        pos_loss = targets * torch.log(probs.clamp(min=1e-8))
        neg_loss = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))
        # Focal weighting
        if self.gamma_pos > 0:
            pos_loss *= (1 - probs) ** self.gamma_pos
        if self.gamma_neg > 0:
            neg_loss *= probs_neg ** self.gamma_neg
        return -(pos_loss + neg_loss).mean()
