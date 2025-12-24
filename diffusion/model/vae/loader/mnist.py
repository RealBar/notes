from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from torchvision import datasets, transforms


@dataclass(frozen=True)
class MnistDataConfig:
    root: str
    image_size: int = 28


def build_mnist_datasets(cfg: MnistDataConfig) -> tuple[Dataset[torch.Tensor], Dataset[torch.Tensor]]:
    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
        ]
    )
    train = datasets.MNIST(root=cfg.root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=cfg.root, train=False, download=True, transform=tfm)
    return train, test

