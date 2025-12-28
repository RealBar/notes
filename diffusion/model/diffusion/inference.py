import torch
from torchvision.utils import save_image
from dit import DiT
from train import Diffusion, IMAGE_SIZE, TIMESTEPS, DEVICE, DIT_KWARGS
import argparse
import os
import glob

def inference(checkpoint_path=None, num_images=8):
    # Determine checkpoint
    if checkpoint_path is None:
        checkpoints = sorted(glob.glob("results/dit_epoch_*.pt"))
        if not checkpoints:
            print("No checkpoints found in results/")
            return
        checkpoint_path = checkpoints[-1]
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    model = DiT(**DIT_KWARGS).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    try:
        model.load_state_dict(ckpt)
    except RuntimeError as e:
        print(f"Error loading checkpoint: {e}")
        print(f"The checkpoint might be trained with a different resolution. Current IMAGE_SIZE is {IMAGE_SIZE}.")
        return
    
    diffusion = Diffusion(timesteps=TIMESTEPS)
    
    print(f"Sampling {num_images} images on {DEVICE}...")
    images = diffusion.sample(model, n=num_images)
    
    os.makedirs("results", exist_ok=True)
    save_path = "results/inference_result.png"
    save_image(images, save_path)
    print(f"Saved inference result to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--n", type=int, default=8, help="Number of images to generate")
    args = parser.parse_args()
    
    inference(args.checkpoint, args.n)
