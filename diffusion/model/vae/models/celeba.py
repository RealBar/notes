from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..modules.conv_blocks import ConvBlock, DeconvBlock
from .base import VAEBase


class CelebaVAE(VAEBase):
    def __init__(
        self,
        *,
        latent_dim: int = 256,
        image_size: int = 64,
        base_channels: int = 32,
        recon_loss: str = "mse",
        beta: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels=3,
            image_size=image_size,
            latent_dim=latent_dim,
            recon_loss=recon_loss,  # type: ignore[arg-type]
            beta=beta,
        )
        if image_size < 32:
            raise ValueError("CelebaVAE expects image_size >= 32")

        self.base_channels = int(base_channels)

        c = self.base_channels
        self.encoder = nn.Sequential(
            ConvBlock(3, c, norm=False, activation="lrelu"),
            ConvBlock(c, c * 2, norm=True, activation="lrelu"),
            ConvBlock(c * 2, c * 4, norm=True, activation="lrelu"),
            ConvBlock(c * 4, c * 8, norm=True, activation="lrelu"),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        enc_out = c * 8 * 4 * 4
        self.fc_mu = nn.Linear(enc_out, self.latent_dim)
        self.fc_logvar = nn.Linear(enc_out, self.latent_dim)

        self.fc_z = nn.Linear(self.latent_dim, enc_out)
        self.decoder = nn.Sequential(
            DeconvBlock(c * 8, c * 4, norm=True, activation="relu"),
            DeconvBlock(c * 4, c * 2, norm=True, activation="relu"),
            DeconvBlock(c * 2, c, norm=True, activation="relu"),
            nn.ConvTranspose2d(c, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        h = self.pool(h)
        h = h.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_z(z)
        h = h.view(z.shape[0], self.base_channels * 8, 4, 4)
        x = self.decoder(h)
        if x.shape[-1] != self.image_size:
            x = torch.nn.functional.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return x

