from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        d = torch.device(prefer)
        if d.type == "cuda" and torch.cuda.is_available():
            return d
        if d.type == "mps" and torch.backends.mps.is_available():
            return d
        if d.type == "cpu":
            return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

