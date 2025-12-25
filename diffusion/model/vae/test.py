from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from .loader import build_datasets, get_dataset_spec
from .factory import create_model, load_model_checkpoint
from .utils.device import get_device


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="celeba", choices=["celeba", "mnist"])
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="runs/vae_eval")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--celeba-train-ratio", type=float, default=0.95)
    p.add_argument("--limit", type=int, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    model_name = ckpt_args.get("model", ckpt_args.get("model_name", args.dataset))
    if args.dataset == "celeba":
        default_latent_dim = 256
        default_image_size = 64
        default_base_channels = 32
    else:
        default_latent_dim = 32
        default_image_size = 28
        default_base_channels = 32

    ds_spec = get_dataset_spec(args.dataset)
    default_latent_dim = 256 if args.dataset == "celeba" else 32
    latent_dim = int(ckpt_args.get("latent_dim", default_latent_dim))
    image_size = int(ckpt_args.get("image_size", default_image_size))
    base_channels = int(ckpt_args.get("base_channels", default_base_channels))
    beta = float(ckpt_args.get("beta", 1.0))
    recon_loss = ckpt_args.get("recon_loss", None)
    seed = int(ckpt_args.get("seed", 42))

    _, test_ds, _ = build_datasets(
        args.dataset,
        data_root=args.data_root,
        image_size=image_size,
        seed=seed,
        train_ratio=args.celeba_train_ratio,
        limit=args.limit,
    )
    in_channels = ds_spec.in_channels

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = create_model(
        model_name,
        latent_dim=latent_dim,
        image_size=image_size,
        in_channels=in_channels,
        base_channels=base_channels,
    )
    model.beta = beta
    if recon_loss is not None:
        model.recon_loss = recon_loss
    load_model_checkpoint(model, args.ckpt, map_location="cpu")
    model.to(device)
    model.eval()

    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    n_batches = 0
    first_batch = None
    with torch.inference_mode():
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            recon, mu, logvar = model(x)
            losses = model.compute_loss(recon, x, mu, logvar)
            totals["loss"] += float(losses.loss.item())
            totals["recon"] += float(losses.recon.item())
            totals["kl"] += float(losses.kl.item())
            n_batches += 1
            if first_batch is None:
                first_batch = (x.detach().cpu(), recon.detach().cpu())

    denom = max(n_batches, 1)
    print(
        f"loss={totals['loss']/denom:.4f} recon={totals['recon']/denom:.4f} kl={totals['kl']/denom:.4f} batches={n_batches}"
    )

    if first_batch is not None:
        x, recon = first_batch
        n = min(args.n_samples, x.shape[0])
        pair = torch.cat([x[:n], recon[:n]], dim=0)
        if ds_spec.data_space == "minus_one_one":
            pair = (pair * 0.5 + 0.5).clamp(0, 1)
        grid = make_grid(pair, nrow=n, padding=2)
        save_image(grid, out_dir / "recon_grid.png")

    samples = model.sample(args.n_samples, device=device).detach().cpu()
    if ds_spec.data_space == "minus_one_one":
        samples = (samples * 0.5 + 0.5).clamp(0, 1)
    grid = make_grid(samples, nrow=int(args.n_samples**0.5), padding=2)
    save_image(grid, out_dir / "samples_grid.png")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()

