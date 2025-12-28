from __future__ import annotations

import os
import math
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from config import CFG
from dit import DiT
from ldm import (
    AutoencoderKL,
    EMA,
    Schedule,
    ddim_sample,
    get_device,
    make_grad_scaler,
    make_linear_schedule,
    q_sample,
    seed_everything,
)


def load_torch(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_diffusion_checkpoint(path: Path):
    payload = load_torch(path)
    if not isinstance(payload, dict):
        return {"model": payload}, 0, 0

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    epoch = int(meta.get("epoch", payload.get("epoch", 0)))
    global_step = int(meta.get("global_step", payload.get("global_step", 0)))
    return payload, epoch, global_step


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def save_diffusion_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    ema: EMA,
    optimizer: AdamW,
    epoch: int,
    global_step: int,
    save_optimizer: bool,
) -> None:
    payload: dict = {
        "model": model.state_dict(),
        "ema": ema.shadow,
        "meta": {"epoch": int(epoch), "global_step": int(global_step)},
    }
    if save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    atomic_torch_save(payload, path)


def build_dataloader(*, batch_size: int) -> DataLoader:
    smoke_test = os.environ.get("SMOKE_TEST") == "1"
    if smoke_test:
        images = torch.rand(64, CFG.train.in_channels, CFG.train.image_size, CFG.train.image_size) * 2 - 1
        return DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=True, num_workers=0)

    if CFG.data.dataset_type != "hf":
        raise ValueError(f"unknown dataset_type: {CFG.data.dataset_type}")

    from datasets import load_dataset

    dataset = load_dataset(CFG.data.hf_name, split=CFG.data.hf_split)
    tfm = transforms.Compose(
        [
            transforms.Resize((CFG.train.image_size, CFG.train.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * CFG.train.in_channels, std=[0.5] * CFG.train.in_channels),
        ]
    )

    def transform_images(examples):
        examples["pixel_values"] = [tfm(image.convert("RGB")) for image in examples["image"]]
        return examples

    dataset.set_transform(transform_images)

    def collate_fn(batch):
        return torch.stack([item["pixel_values"] for item in batch])

    device = get_device()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CFG.data.num_workers,
        pin_memory=device == "cuda",
    )


def kl_loss(mu: Tensor, logvar: Tensor) -> Tensor:
    return 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)


def train_autoencoder(*, device: str, dataloader: DataLoader) -> AutoencoderKL:
    ae = AutoencoderKL(
        in_channels=CFG.train.in_channels,
        base_channels=CFG.ae.base_channels,
        latent_channels=CFG.ae.latent_channels,
        downsample_factor=CFG.ae.downsample_factor,
    ).to(device)

    ckpt_path = Path(CFG.ae.ckpt_path)
    if ckpt_path.exists():
        payload = load_torch(ckpt_path)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        ae.load_state_dict(state, strict=True)
        if isinstance(payload, dict) and set(payload.keys()) != {"model"}:
            torch.save({"model": ae.state_dict()}, ckpt_path)
        return ae

    optimizer = AdamW(ae.parameters(), lr=CFG.ae.lr)
    amp_enabled = device == "cuda"
    scaler = make_grad_scaler(device, enabled=amp_enabled)
    dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    ae.train()

    for epoch in range(int(CFG.ae.epochs)):
        for step, batch in enumerate(dataloader):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)

            if amp_enabled:
                with torch.amp.autocast(device_type="cuda", dtype=dtype, cache_enabled=False):
                    recon, mu, logvar = ae(x)
                    recon_loss = torch.mean((recon - x) ** 2)
                    loss = recon_loss + CFG.ae.kl_weight * kl_loss(mu, logvar)
            else:
                recon, mu, logvar = ae(x)
                recon_loss = torch.mean((recon - x) ** 2)
                loss = recon_loss + CFG.ae.kl_weight * kl_loss(mu, logvar)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % 50 == 0:
                print(f"[AE] epoch={epoch+1} step={step} loss={loss.item():.6f} recon={recon_loss.item():.6f}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": ae.state_dict()}, ckpt_path)
    return ae


def save_grid(images: Tensor, *, path: str, nrow: int) -> None:
    images = (images * 0.5 + 0.5).clamp(0, 1)
    grid = make_grid(images, nrow=nrow, padding=2)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out)


def main() -> None:
    seed_everything(CFG.train.seed)
    device = get_device()

    out_dir = Path(CFG.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ae_loader = build_dataloader(batch_size=CFG.ae.batch_size)
    ae = train_autoencoder(device=device, dataloader=ae_loader)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    latent_size = CFG.train.image_size // CFG.ae.downsample_factor
    if latent_size * CFG.ae.downsample_factor != CFG.train.image_size:
        raise ValueError("image_size must be divisible by downsample_factor")

    model = DiT(
        input_size=latent_size,
        patch_size=CFG.dit.patch_size,
        in_channels=CFG.ae.latent_channels,
        hidden_size=CFG.dit.hidden_size,
        depth=CFG.dit.depth,
        num_heads=CFG.dit.num_heads,
        mlp_ratio=CFG.dit.mlp_ratio,
        use_checkpointing=CFG.dit.use_checkpointing,
    ).to(device)

    ema = EMA(model, decay=CFG.train.ema_decay)
    optimizer = AdamW(model.parameters(), lr=CFG.train.lr)
    schedule = make_linear_schedule(
        num_train_timesteps=CFG.diffusion.num_train_timesteps,
        beta_start=CFG.diffusion.beta_start,
        beta_end=CFG.diffusion.beta_end,
        device=device,
    )

    ckpt_path = Path(CFG.train.diffusion_ckpt_path)
    start_epoch = 0
    global_step = 0
    if ckpt_path.exists():
        payload, start_epoch, global_step = load_diffusion_checkpoint(ckpt_path)
        if isinstance(payload, dict) and "model" in payload:
            model.load_state_dict(payload["model"], strict=True)
            if "ema" in payload and isinstance(payload["ema"], dict):
                ema.shadow = payload["ema"]
            if CFG.train.save_optimizer and "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])

    train_loader = build_dataloader(batch_size=CFG.train.batch_size)

    amp_enabled = device == "cuda"
    scaler = make_grad_scaler(device, enabled=amp_enabled)
    dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    accum_steps = max(1, int(CFG.train.effective_batch_size // CFG.train.batch_size))
    mse = nn.MSELoss()

    if device == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(
            f"device=cuda total/free={total_bytes/(1024**3):.2f}/{free_bytes/(1024**3):.2f}GB "
            f"batch={CFG.train.batch_size} accum={accum_steps} latent={CFG.ae.latent_channels}x{latent_size}x{latent_size}"
        )
    else:
        print(f"device={device} batch={CFG.train.batch_size} accum={accum_steps}")

    if start_epoch > 0 or global_step > 0:
        print(f"resuming: epoch={start_epoch} global_step={global_step}")

    for epoch in range(int(start_epoch), int(CFG.train.epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running = 0.0
        steps = 0

        for step, batch in enumerate(train_loader):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)

            with torch.no_grad():
                mu, logvar = ae.encode(x)
                z = ae.reparameterize(mu, logvar)
                z = z * CFG.diffusion.latent_scale

            noise = torch.randn_like(z)
            t = torch.randint(0, CFG.diffusion.num_train_timesteps, (z.shape[0],), device=device)
            zt = q_sample(x0=z, noise=noise, t=t, schedule=schedule)

            if amp_enabled:
                with torch.amp.autocast(device_type="cuda", dtype=dtype, cache_enabled=False):
                    pred = model(zt, t)
                    loss = mse(pred.float(), noise.float()) / accum_steps
            else:
                pred = model(zt, t)
                loss = mse(pred, noise) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                if CFG.train.grad_clip_norm > 0:
                    if amp_enabled and hasattr(scaler, "unscale_"):
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(CFG.train.grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                global_step += 1

            running += float(loss.item() * accum_steps)
            steps += 1
            if step % 50 == 0:
                print(f"epoch={epoch+1} step={step} loss={running/max(1,steps):.6f}")

        if steps % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            global_step += 1

        if (epoch + 1) % int(CFG.train.sample_every_epochs) == 0:
            model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            model.eval()
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
            save_grid(images, path=str(out_dir / f"sample_epoch_{epoch+1}.png"), nrow=int(math.sqrt(CFG.train.sample_n)))
            model.load_state_dict(model_state, strict=True)

        if (epoch + 1) % int(CFG.train.save_every_epochs) == 0:
            save_diffusion_checkpoint(
                path=ckpt_path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                save_optimizer=bool(CFG.train.save_optimizer),
            )


if __name__ == "__main__":
    main()
