from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, random_split

from torchvision import transforms

from .image_dir import ImageDirDataset


@dataclass(frozen=True)
class CelebaDataConfig:
    root: str
    image_size: int = 64
    train_ratio: float = 0.95
    seed: int = 42
    limit: int | None = None


def build_celeba_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def build_celeba_datasets(cfg: CelebaDataConfig) -> tuple[Dataset[torch.Tensor], Dataset[torch.Tensor]]:
    root = Path(cfg.root)
    img_dir = root
    if (root / "img_align_celeba").exists():
        img_dir = root / "img_align_celeba"
    transform = build_celeba_transforms(cfg.image_size)
    ds = ImageDirDataset(img_dir, transform=transform, limit=cfg.limit)
    n = len(ds)
    n_train = int(n * float(cfg.train_ratio))
    n_test = n - n_train
    g = torch.Generator().manual_seed(int(cfg.seed))
    train_ds, test_ds = random_split(ds, [n_train, n_test], generator=g)
    return train_ds, test_ds

