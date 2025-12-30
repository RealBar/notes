from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import math

import torch
from torchvision.utils import make_grid, save_image

from config import CFG
from dit import DiT
from ldm import AutoencoderKL, ddim_sample, get_device, make_linear_schedule, seed_everything


def load_torch(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    seed_everything(CFG.train.seed)
    device = get_device()

    latent_size = CFG.train.image_size // CFG.ae.downsample_factor
    schedule = make_linear_schedule(
        num_train_timesteps=CFG.diffusion.num_train_timesteps,
        beta_start=CFG.diffusion.beta_start,
        beta_end=CFG.diffusion.beta_end,
        device=device,
    )

    ae = AutoencoderKL(
        in_channels=CFG.train.in_channels,
        base_channels=CFG.ae.base_channels,
        latent_channels=CFG.ae.latent_channels,
        downsample_factor=CFG.ae.downsample_factor,
    ).to(device)
    ae_payload = load_torch(CFG.ae.ckpt_path)
    ae_state = ae_payload.get("model", ae_payload) if isinstance(ae_payload, dict) else ae_payload
    ae.load_state_dict(ae_state, strict=True)
    ae.eval()

    model = DiT(
        input_size=latent_size,
        patch_size=CFG.dit.patch_size,
        in_channels=CFG.ae.latent_channels,
        hidden_size=CFG.dit.hidden_size,
        depth=CFG.dit.depth,
        num_heads=CFG.dit.num_heads,
        mlp_ratio=CFG.dit.mlp_ratio,
        use_checkpointing=False,
    ).to(device)
    payload = load_torch(CFG.train.diffusion_ckpt_path)
    if isinstance(payload, dict) and "ema" in payload:
        model.load_state_dict(payload["ema"], strict=True)
    elif isinstance(payload, dict) and "model" in payload:
        model.load_state_dict(payload["model"], strict=True)
    else:
        model.load_state_dict(payload, strict=True)
    model.eval()

    amp_enabled = device == "cuda"
    dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16

    latents = ddim_sample(
        model=model,
        shape=(CFG.train.sample_n, CFG.ae.latent_channels, latent_size, latent_size),
        schedule=schedule,
        ddim_steps=CFG.diffusion.ddim_steps,
        eta=CFG.diffusion.ddim_eta,
        device=device,
        dtype=dtype if amp_enabled else torch.float32,
        seed=CFG.train.seed,
    )
    latents = latents / CFG.diffusion.latent_scale
    images = ae.decode(latents).detach().cpu()
    images = (images * 0.5 + 0.5).clamp(0, 1)
    grid = make_grid(images, nrow=int(math.sqrt(CFG.train.sample_n)), padding=2)

    out_dir = Path(CFG.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inference_grid.png"
    save_image(grid, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
