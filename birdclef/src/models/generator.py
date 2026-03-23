"""U-Net adversarial augmentor for spectrogram perturbation."""

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from non-power-of-2 dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AdversarialGenerator(nn.Module):
    """U-Net that generates adversarial perturbations on mel spectrograms."""

    def __init__(self, channels=(32, 64, 128, 256), epsilon=0.15):
        super().__init__()
        self.epsilon = epsilon
        c = channels

        # Encoder
        self.down1 = DownBlock(1, c[0])       # 1 -> 32
        self.down2 = DownBlock(c[0], c[1])     # 32 -> 64
        self.down3 = DownBlock(c[1], c[2])     # 64 -> 128
        self.down4 = DownBlock(c[2], c[3])     # 128 -> 256

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c[3], c[3], 3, padding=1),
            nn.BatchNorm2d(c[3]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder — skip_ch matches the encoder output at each level
        self.up4 = UpBlock(c[3], c[3], c[2])   # 256 + skip(256) -> 128
        self.up3 = UpBlock(c[2], c[2], c[1])   # 128 + skip(128) -> 64
        self.up2 = UpBlock(c[1], c[1], c[0])   # 64 + skip(64) -> 32
        self.up1 = UpBlock(c[0], c[0], c[0])   # 32 + skip(32) -> 32

        # Output: perturbation delta
        self.out_conv = nn.Sequential(
            nn.Conv2d(c[0], 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """Generate perturbation delta for input spectrogram.

        Args:
            x: (B, 1, n_mels, T) mel spectrogram
        Returns:
            delta: (B, 1, n_mels, T) perturbation, clamped to [-eps, eps]
        """
        # Encoder
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x4, skip4 = self.down4(x3)

        # Bottleneck
        b = self.bottleneck(x4)

        # Decoder with skip connections
        u4 = self.up4(b, skip4)
        u3 = self.up3(u4, skip3)
        u2 = self.up2(u3, skip2)
        u1 = self.up1(u2, skip1)

        # Output perturbation scaled by epsilon
        delta = self.out_conv(u1) * self.epsilon

        # Handle size mismatch with input
        if delta.shape != x.shape:
            delta = nn.functional.interpolate(delta, size=x.shape[2:])

        return delta

    def perturb(self, x):
        """Apply adversarial perturbation to spectrogram."""
        return x + self.forward(x)
