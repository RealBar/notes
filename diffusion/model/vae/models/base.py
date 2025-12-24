from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ReconLoss = Literal["mse", "bce"]


@dataclass(frozen=True)
class VAELoss:
    loss: Tensor
    recon: Tensor
    kl: Tensor


class VAEBase(nn.Module, ABC):
    def __init__(
        self,
        *,
        in_channels: int,
        image_size: int,
        latent_dim: int,
        recon_loss: ReconLoss = "mse",
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.latent_dim = int(latent_dim)
        self.recon_loss = recon_loss
        self.beta = float(beta)

    @abstractmethod
    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def kl_divergence(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def reconstruction_loss(self, recon: Tensor, x: Tensor) -> Tensor:
        if self.recon_loss == "mse":
            per_elem = F.mse_loss(recon, x, reduction="none")
            return per_elem.flatten(1).sum(dim=1)
        if self.recon_loss == "bce":
            per_elem = F.binary_cross_entropy(recon, x, reduction="none")
            return per_elem.flatten(1).sum(dim=1)
        raise ValueError(f"Unknown recon_loss: {self.recon_loss}")

    def compute_loss(self, recon: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> VAELoss:
        recon_loss = self.reconstruction_loss(recon, x)
        kl = self.kl_divergence(mu, logvar)
        loss = recon_loss + self.beta * kl
        return VAELoss(loss=loss.mean(), recon=recon_loss.mean(), kl=kl.mean())

    @torch.inference_mode()
    def reconstruct(self, x: Tensor) -> Tensor:
        self.eval()
        recon, _, _ = self(x)
        return recon

    @torch.inference_mode()
    def sample(self, num_samples: int, *, device: torch.device | None = None) -> Tensor:
        self.eval()
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

