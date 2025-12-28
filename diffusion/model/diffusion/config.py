from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    dataset_type: str = "hf"
    hf_name: str = "nielsr/CelebA-faces"
    hf_split: str = "train"
    num_workers: int = 0


@dataclass(frozen=True)
class AutoencoderConfig:
    ckpt_path: str = "results/ldm_ae_last.pt"
    base_channels: int = 64
    latent_channels: int = 4
    downsample_factor: int = 8
    kl_weight: float = 1e-6
    lr: float = 2e-4
    batch_size: int = 64
    epochs: int = 1


@dataclass(frozen=True)
class DiTConfig:
    patch_size: int = 1
    hidden_size: int = 256
    depth: int = 8
    num_heads: int = 4
    mlp_ratio: float = 4.0
    use_checkpointing: bool = True


@dataclass(frozen=True)
class DiffusionConfig:
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    ddim_steps: int = 50
    ddim_eta: float = 0.0
    latent_scale: float = 0.18215


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    image_size: int = 64
    in_channels: int = 3
    batch_size: int = 8
    effective_batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-4
    ema_decay: float = 0.9999
    grad_clip_norm: float = 1.0
    save_every_epochs: int = 1
    sample_every_epochs: int = 1
    sample_n: int = 8
    out_dir: str = "results"
    diffusion_ckpt_path: str = "results/ldm_dit_last.pt"
    save_optimizer: bool = False


@dataclass(frozen=True)
class Config:
    data: DataConfig = DataConfig()
    ae: AutoencoderConfig = AutoencoderConfig()
    dit: DiTConfig = DiTConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    train: TrainConfig = TrainConfig()


CFG = Config()
