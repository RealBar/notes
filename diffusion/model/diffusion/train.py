import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from datasets import load_dataset
from dit import DiT
import os
import glob

# Hyperparameters
IMAGE_SIZE = 32 
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 5
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"

def get_dataloader():
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
        with torch.no_grad():
            x = torch.randn((n, 3, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)
            for i in reversed(range(1, self.timesteps)):
                t = (torch.ones(n) * i).long().to(DEVICE)
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
    
    model = DiT(input_size=IMAGE_SIZE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    diffusion = Diffusion(timesteps=TIMESTEPS)

    print(f"Starting training on {DEVICE}...")

    start_epoch = 0
    checkpoints = sorted(glob.glob("results/dit_epoch_*.pt"))
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"Resuming from {latest_checkpoint}...")
        model.load_state_dict(torch.load(latest_checkpoint, map_location=DEVICE))
        try:
            start_epoch = int(latest_checkpoint.split("_")[-1].split(".")[0])
        except ValueError:
            pass
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        for i, images in enumerate(dataloader):
            images = images.to(DEVICE)
            t = diffusion.sample_timesteps(images.shape[0])
            x_t, noise = diffusion.noise_images(images, t)
            
            predicted_noise = model(x_t, t)
            loss = loss_fn(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Step {i} | Loss: {loss.item():.4f}")
        
        # Inference
        print(f"Sampling images for Epoch {epoch+1}...")
        sampled_images = diffusion.sample(model, n=8)
        save_image(sampled_images, f"results/sample_epoch_{epoch+1}.png")

        # Save model checkpoint
        save_path = f"results/dit_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

if __name__ == "__main__":
    train()
