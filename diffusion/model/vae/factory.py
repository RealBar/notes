from typing import Literal

import torch

from .models.base import VAEBase
from .models.celeba import CelebaVAE
from .models.mnist import MnistVAE

ModelName = Literal["celeba", "mnist"]

def create_model(
    name: ModelName,
    *,
    latent_dim: int,
    image_size: int,
    in_channels: int | None = None,
    base_channels: int = 32,
) -> VAEBase:
    if name == "celeba":
        return CelebaVAE(
            latent_dim=latent_dim,
            image_size=image_size,
            base_channels=base_channels,
        )
    if name == "mnist":
        return MnistVAE(
            latent_dim=latent_dim,
            image_size=image_size,
            in_channels=in_channels or 1,
            base_channels=base_channels,
        )
    raise ValueError(f"Unknown model name: {name}")


def load_model_checkpoint(
    model: VAEBase,
    ckpt_path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(ckpt_path, map_location=map_location)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    return payload if isinstance(payload, dict) else {"model": payload}

