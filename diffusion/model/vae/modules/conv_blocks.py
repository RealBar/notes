from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: bool = True,
        activation: str = "lrelu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not norm,
            )
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation == "lrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "none":
            pass
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeconvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not norm,
            )
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "lrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "none":
            pass
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

