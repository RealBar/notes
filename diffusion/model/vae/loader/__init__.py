from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset, Subset

from .celeba import CelebaDataConfig, build_celeba_datasets
from .mnist import MnistDataConfig, build_mnist_datasets


DatasetName = Literal["celeba", "mnist"]
DataSpace = Literal["zero_one", "minus_one_one"]
ReconLoss = Literal["mse", "bce"]


@dataclass(frozen=True)
class DatasetSpec:
    name: DatasetName
    in_channels: int
    data_space: DataSpace
    default_recon_loss: ReconLoss
    default_image_size: int


def get_dataset_spec(name: DatasetName) -> DatasetSpec:
    if name == "celeba":
        return DatasetSpec(
            name="celeba",
            in_channels=3,
            data_space="minus_one_one",
            default_recon_loss="mse",
            default_image_size=64,
        )
    if name == "mnist":
        return DatasetSpec(
            name="mnist",
            in_channels=1,
            data_space="zero_one",
            default_recon_loss="bce",
            default_image_size=28,
        )
    raise ValueError(f"Unknown dataset: {name}")


def _maybe_limit(ds: Dataset[torch.Tensor], limit: int | None) -> Dataset[torch.Tensor]:
    if limit is None:
        return ds
    n = min(int(limit), len(ds))
    return Subset(ds, list(range(n)))


def build_datasets(
    name: DatasetName,
    *,
    data_root: str,
    image_size: int | None = None,
    seed: int = 42,
    train_ratio: float = 0.95,
    limit: int | None = None,
) -> tuple[Dataset[torch.Tensor], Dataset[torch.Tensor], DatasetSpec]:
    spec = get_dataset_spec(name)
    image_size = int(image_size) if image_size is not None else spec.default_image_size

    if name == "celeba":
        cfg = CelebaDataConfig(
            root=data_root,
            image_size=image_size,
            train_ratio=float(train_ratio),
            seed=int(seed),
            limit=limit,
        )
        train_ds, test_ds = build_celeba_datasets(cfg)
    elif name == "mnist":
        cfg = MnistDataConfig(root=data_root, image_size=image_size)
        train_ds, test_ds = build_mnist_datasets(cfg)
        train_ds = _maybe_limit(train_ds, limit)
        test_ds = _maybe_limit(test_ds, limit)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_ds, test_ds, spec


__all__ = [
    "CelebaDataConfig",
    "MnistDataConfig",
    "DatasetSpec",
    "build_celeba_datasets",
    "build_mnist_datasets",
    "build_datasets",
    "get_dataset_spec",
]

