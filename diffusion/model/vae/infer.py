from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from .factory import create_model, load_model_checkpoint
from .utils.device import get_device


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="sample.png")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dataset", type=str, default="celeba", choices=["celeba", "mnist"])
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = get_device(args.device)
    payload = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    model_name = ckpt_args.get("model", ckpt_args.get("model_name", args.dataset))
    latent_dim = int(ckpt_args.get("latent_dim", 256))
    image_size = int(ckpt_args.get("image_size", 64))
    base_channels = int(ckpt_args.get("base_channels", 32))

    in_channels = 3 if args.dataset == "celeba" else 1
    model = create_model(
        model_name,
        latent_dim=latent_dim,
        image_size=image_size,
        in_channels=in_channels,
        base_channels=base_channels,
    )
    load_model_checkpoint(model, args.ckpt, map_location="cpu")
    model.to(device)
    model.eval()

    samples = model.sample(args.n, device=device).detach().cpu()
    if args.dataset == "celeba":
        samples = (samples * 0.5 + 0.5).clamp(0, 1)
    grid = make_grid(samples, nrow=int(args.n**0.5), padding=2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

