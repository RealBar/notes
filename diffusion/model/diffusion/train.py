import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image
from datasets import load_dataset
from dit import DiT
import glob
from contextlib import nullcontext

# Hyperparameters
IMAGE_SIZE = 64
LR = 1e-4
EPOCHS = 5
TIMESTEPS = 1000

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

TOTAL_VRAM_GB = 0.0
FREE_VRAM_GB = 0.0
if DEVICE == "cuda":
    TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    FREE_VRAM_GB = free_bytes / (1024**3)

SMOKE_TEST = os.environ.get("SMOKE_TEST") == "1"

LOW_VRAM = DEVICE == "cuda" and TOTAL_VRAM_GB <= 7.5
VERY_LOW_FREE_VRAM = DEVICE == "cuda" and FREE_VRAM_GB > 0 and FREE_VRAM_GB < 1.6
LOW_MEM = LOW_VRAM or VERY_LOW_FREE_VRAM or SMOKE_TEST

EFFECTIVE_BATCH_SIZE = 64
BATCH_SIZE = 2 if VERY_LOW_FREE_VRAM else 8 if LOW_MEM else 64
GRAD_ACCUM_STEPS = max(1, EFFECTIVE_BATCH_SIZE // BATCH_SIZE)

PATCH_SIZE = 8 if VERY_LOW_FREE_VRAM else 4 if LOW_MEM else 2
HIDDEN_SIZE = 192 if VERY_LOW_FREE_VRAM else 256 if LOW_MEM else 384
DEPTH = 6 if VERY_LOW_FREE_VRAM else 8 if LOW_MEM else 12
NUM_HEADS = 3 if VERY_LOW_FREE_VRAM else 4 if LOW_MEM else 6
USE_CHECKPOINTING = LOW_MEM

DIT_KWARGS = dict(
    input_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    hidden_size=HIDDEN_SIZE,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    use_checkpointing=USE_CHECKPOINTING,
)

SAMPLE_N = 2 if VERY_LOW_FREE_VRAM else 4 if LOW_MEM else 8

if SMOKE_TEST:
    EPOCHS = 1
    TIMESTEPS = 10
    GRAD_ACCUM_STEPS = 1
    SAMPLE_N = 2

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def get_dataloader():
    if SMOKE_TEST:
        images = torch.rand(32, 3, IMAGE_SIZE, IMAGE_SIZE) * 2 - 1
        dataset = TensorDataset(images)
        return DataLoader(dataset, batch_size=min(BATCH_SIZE, 8), shuffle=True, num_workers=0)
    # Load dataset from huggingface
    # We use "nielsr/CelebA-faces"
    print("Loading dataset...")
    dataset = load_dataset("nielsr/CelebA-faces", split="train")
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def transform_images(examples):
        examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
        return examples

    dataset.set_transform(transform_images)

    def collate_fn(batch):
        return torch.stack([item["pixel_values"] for item in batch])

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    return dataloader

class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(DEVICE)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n,)).to(DEVICE)

    def sample(self, model, n):
        model.eval()
        amp_enabled = DEVICE == "cuda"
        amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
        autocast_ctx = torch.cuda.amp.autocast if amp_enabled else (lambda **kwargs: nullcontext())
        with torch.no_grad():
            x = torch.randn((n, 3, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)
            for i in reversed(range(1, self.timesteps)):
                t = (torch.ones(n, device=DEVICE) * i).long()
                with autocast_ctx(dtype=amp_dtype, cache_enabled=False):
                    predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x

def train():
    os.makedirs("results", exist_ok=True)
    
    dataloader = get_dataloader()
    
    model = DiT(**DIT_KWARGS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    diffusion = Diffusion(timesteps=TIMESTEPS)

    amp_enabled = DEVICE == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    autocast_ctx = torch.cuda.amp.autocast if amp_enabled else (lambda **kwargs: nullcontext())
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print(f"Starting training on {DEVICE}...")
    if DEVICE == "cuda":
        print(
            f"CUDA VRAM total/free: {TOTAL_VRAM_GB:.2f}/{FREE_VRAM_GB:.2f} GB | low_mem={LOW_MEM} | batch={BATCH_SIZE} | accum={GRAD_ACCUM_STEPS}"
        )

    start_epoch = 0
    if not SMOKE_TEST:
        checkpoints = sorted(glob.glob("results/dit_epoch_*.pt"))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"Resuming from {latest_checkpoint}...")
            try:
                model.load_state_dict(torch.load(latest_checkpoint, map_location=DEVICE))
                try:
                    start_epoch = int(latest_checkpoint.split("_")[-1].split(".")[0])
                except ValueError:
                    pass
            except RuntimeError as e:
                print(f"Failed to load checkpoint {latest_checkpoint}: {e}")
                print("Starting training from scratch...")
                start_epoch = 0
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for i, images in enumerate(dataloader):
            if isinstance(images, (tuple, list)):
                images = images[0]
            images = images.to(DEVICE, non_blocking=True)

            with autocast_ctx(dtype=amp_dtype, cache_enabled=False):
                t = diffusion.sample_timesteps(images.shape[0])
                x_t, noise = diffusion.noise_images(images, t)
                predicted_noise = model(x_t, t)
                loss = loss_fn(noise.float(), predicted_noise.float())
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Step {i} | Loss: {(loss.item() * GRAD_ACCUM_STEPS):.4f}")

        if len(dataloader) % GRAD_ACCUM_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Inference
        print(f"Sampling images for Epoch {epoch+1}...")
        sampled_images = diffusion.sample(model, n=SAMPLE_N)
        save_image(sampled_images, f"results/sample_epoch_{epoch+1}.png")

        # Save model checkpoint
        save_path = f"results/dit_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

if __name__ == "__main__":
    train()
