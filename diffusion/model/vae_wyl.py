
from torch import nn
import torch

class VAEBase(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim