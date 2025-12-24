from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from time import time

import torch
from torch.utils.data import DataLoader

from .loader.celeba import CelebaDataConfig, build_celeba_datasets
from .loader.mnist import MnistDataConfig, build_mnist_datasets
from .factory import create_model
from .utils.checkpoint import save_checkpoint
from .utils.device import get_device
from .utils.seed import seed_everything


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="celeba", choices=["celeba", "mnist"])
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--model", type=str, default=None, choices=["celeba", "mnist", None])
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--recon-loss", type=str, default=None, choices=["mse", "bce", None])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-dir", type=str, default="runs/vae")
    p.add_argument("--ckpt-every", type=int, default=1)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--celeba-train-ratio", type=float, default=0.95)
    p.add_argument("--limit", type=int, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)

    dataset = args.dataset
    model_name = args.model or dataset

    if dataset == "celeba":
        cfg = CelebaDataConfig(
            root=args.data_root,
            image_size=args.image_size,
            train_ratio=args.celeba_train_ratio,
            seed=args.seed,
            limit=args.limit,
        )
        train_ds, test_ds = build_celeba_datasets(cfg)
        in_channels = 3
        recon_loss = args.recon_loss or "mse"
    else:
        cfg = MnistDataConfig(root=args.data_root, image_size=args.image_size)
        train_ds, test_ds = build_mnist_datasets(cfg)
        in_channels = 1
        recon_loss = args.recon_loss or "bce"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = create_model(
        model_name,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        in_channels=in_channels,
        base_channels=args.base_channels,
    )
    model.beta = float(args.beta)
    model.recon_loss = recon_loss  # type: ignore[assignment]
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))

    start_epoch = 1
    global_step = 0

    save_dir = Path(args.save_dir) / f"{dataset}_{model_name}_z{args.latent_dim}_s{args.image_size}"
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(payload["model"], strict=True)
        if "optimizer" in payload:
            opt.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("step", 0))

    extra = {
        "args": vars(args),
        "data_cfg": asdict(cfg),
        "dataset": dataset,
        "model_name": model_name,
        "device": str(device),
    }

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time()
        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}

        for it, batch in enumerate(train_loader, start=1):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=bool(args.amp)):
                recon, mu, logvar = model(x)
                losses = model.compute_loss(recon, x, mu, logvar)

            scaler.scale(losses.loss).backward()
            scaler.step(opt)
            scaler.update()

            global_step += 1
            running["loss"] += float(losses.loss.item())
            running["recon"] += float(losses.recon.item())
            running["kl"] += float(losses.kl.item())

            if it % args.log_every == 0:
                denom = float(args.log_every)
                msg = (
                    f"epoch={epoch} step={global_step} it={it}/{len(train_loader)} "
                    f"loss={running['loss']/denom:.4f} recon={running['recon']/denom:.4f} kl={running['kl']/denom:.4f}"
                )
                print(msg)
                running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}

        model.eval()
        test_accum = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        n_batches = 0
        with torch.inference_mode():
            for batch in test_loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(device)
                recon, mu, logvar = model(x)
                losses = model.compute_loss(recon, x, mu, logvar)
                test_accum["loss"] += float(losses.loss.item())
                test_accum["recon"] += float(losses.recon.item())
                test_accum["kl"] += float(losses.kl.item())
                n_batches += 1

        test_msg = (
            f"epoch={epoch} time={time()-t0:.1f}s "
            f"val_loss={test_accum['loss']/max(n_batches,1):.4f} "
            f"val_recon={test_accum['recon']/max(n_batches,1):.4f} "
            f"val_kl={test_accum['kl']/max(n_batches,1):.4f}"
        )
        print(test_msg)

        if epoch % args.ckpt_every == 0:
            ckpt_path = save_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=opt,
                epoch=epoch,
                step=global_step,
                extra=extra,
            )

    last_path = save_dir / "last.pt"
    save_checkpoint(last_path, model=model, optimizer=opt, epoch=args.epochs, step=global_step, extra=extra)
    print(f"saved: {last_path}")


if __name__ == "__main__":
    main()

