from __future__ import annotations

import math
from dataclasses import dataclass
from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and getattr(mps_backend, "is_available", None) is not None:
        if mps_backend.is_available():
            return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_autocast(device: str):
    if device == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return lambda **kwargs: torch.amp.autocast(device_type="cuda", **kwargs)
        return torch.cuda.amp.autocast
    return lambda **kwargs: nullcontext()


def make_grad_scaler(device: str, enabled: bool):
    if device == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda", enabled=enabled)
        return torch.cuda.amp.GradScaler(enabled=enabled)

    class _NoOpScaler:
        def scale(self, loss):
            return loss

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            return

    return _NoOpScaler()


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, down: bool = False, up: bool = False) -> None:
        super().__init__()
        self.down = bool(down)
        self.up = bool(up)
        self.norm1 = nn.GroupNorm(32 if in_ch >= 32 else 1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32 if out_ch >= 32 else 1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

        if self.down:
            self.ds = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.ds = nn.Identity()
        if self.up:
            self.us = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.us = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        h = h + self.skip(x)
        h = self.ds(h)
        h = self.us(h)
        return h


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_channels: int = 4,
        downsample_factor: int = 8,
    ) -> None:
        super().__init__()
        if downsample_factor not in (4, 8, 16):
            raise ValueError("downsample_factor must be one of 4, 8, 16")

        c = int(base_channels)
        levels = int(math.log2(downsample_factor))

        enc: list[nn.Module] = [nn.Conv2d(in_channels, c, kernel_size=3, padding=1)]
        in_c = c
        for i in range(levels):
            out_c = c * (2 ** i)
            enc.append(ResBlock(in_c, out_c, down=True))
            in_c = out_c
        enc.append(ResBlock(in_c, in_c))
        self.encoder = nn.Sequential(*enc)
        self.to_stats = nn.Conv2d(in_c, 2 * latent_channels, kernel_size=1)

        dec: list[nn.Module] = [nn.Conv2d(latent_channels, in_c, kernel_size=3, padding=1), ResBlock(in_c, in_c)]
        for i in reversed(range(levels)):
            out_c = c * (2 ** i)
            dec.append(ResBlock(in_c, out_c, up=True))
            in_c = out_c
        dec.append(nn.GroupNorm(32 if in_c >= 32 else 1, in_c))
        dec.append(nn.SiLU())
        dec.append(nn.Conv2d(in_c, in_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        stats = self.to_stats(h)
        mu, logvar = stats.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        return torch.tanh(self.decoder(z))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


@dataclass(frozen=True)
class Schedule:
    betas: Tensor
    alphas: Tensor
    alphas_cumprod: Tensor


def make_linear_schedule(
    *,
    num_train_timesteps: int,
    beta_start: float,
    beta_end: float,
    device: str,
) -> Schedule:
    betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return Schedule(betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod)


def q_sample(*, x0: Tensor, noise: Tensor, t: Tensor, schedule: Schedule) -> Tensor:
    a_bar = schedule.alphas_cumprod[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise


def predict_x0_from_eps(*, xt: Tensor, eps: Tensor, t: Tensor, schedule: Schedule) -> Tensor:
    a_bar = schedule.alphas_cumprod[t].view(-1, 1, 1, 1)
    return (xt - torch.sqrt(1 - a_bar) * eps) / torch.sqrt(a_bar)


def make_ddim_timesteps(num_train_timesteps: int, ddim_steps: int, device: str) -> Tensor:
    if ddim_steps <= 0 or ddim_steps > num_train_timesteps:
        raise ValueError("invalid ddim_steps")
    step = num_train_timesteps // ddim_steps
    ts = torch.arange(0, num_train_timesteps, step, device=device)
    ts = ts[:ddim_steps]
    return ts.flip(0)


@torch.no_grad()
def ddim_sample(
    *,
    model: nn.Module,
    shape: tuple[int, int, int, int],
    schedule: Schedule,
    ddim_steps: int,
    eta: float,
    device: str,
    dtype: torch.dtype,
    seed: int | None = None,
) -> Tensor:
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        x = torch.randn(shape, device=device, generator=gen)
    else:
        x = torch.randn(shape, device=device)

    timesteps = make_ddim_timesteps(schedule.betas.shape[0], ddim_steps, device)
    autocast_ctx = make_autocast(device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((shape[0],), int(t.item()), device=device, dtype=torch.long)

        with autocast_ctx(dtype=dtype, cache_enabled=False):
            eps = model(x, t_batch)

        x0 = predict_x0_from_eps(xt=x, eps=eps, t=t_batch, schedule=schedule)

        if i == len(timesteps) - 1:
            x = x0
            continue

        t_prev = timesteps[i + 1]
        t_prev_batch = torch.full((shape[0],), int(t_prev.item()), device=device, dtype=torch.long)

        a_bar = schedule.alphas_cumprod[t_batch].view(-1, 1, 1, 1)
        a_bar_prev = schedule.alphas_cumprod[t_prev_batch].view(-1, 1, 1, 1)

        sigma = (
            eta
            * torch.sqrt((1 - a_bar_prev) / (1 - a_bar))
            * torch.sqrt(1 - a_bar / a_bar_prev)
        )
        noise = torch.randn_like(x)

        dir_xt = torch.sqrt(torch.clamp(1 - a_bar_prev - sigma**2, min=0.0)) * eps
        x = torch.sqrt(a_bar_prev) * x0 + dir_xt + sigma * noise

    return x


class EMA:
    def __init__(self, model: nn.Module, *, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in msd.items():
                if k not in self.shadow:
                    self.shadow[k] = v.detach().clone()
                else:
                    if self.shadow[k].device != v.device:
                        self.shadow[k] = self.shadow[k].to(v.device)
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)
