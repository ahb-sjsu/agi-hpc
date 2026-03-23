"""EfficientNet-B3 classifier for BirdCLEF species identification."""

import torch
import torch.nn as nn
import timm


class BirdClassifier(nn.Module):
    """EfficientNet-B3 with custom classification head for multi-label bird ID."""

    def __init__(self, num_classes, backbone="tf_efficientnet_b3_ns",
                 pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, in_chans=1, num_classes=0,
        )
        self.feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, n_mels, T) mel spectrogram
        Returns:
            (B, num_classes) logits (pre-sigmoid)
        """
        features = self.backbone.forward_features(x)
        return self.head(features)

    def predict_proba(self, x):
        """Return sigmoid probabilities."""
        return torch.sigmoid(self.forward(x))
