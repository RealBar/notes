from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    step: int | None = None,
    extra: dict | None = None,
) -> None:
    payload: dict = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if step is not None:
        payload["step"] = int(step)
    if extra is not None:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Invalid checkpoint payload")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload

