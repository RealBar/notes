from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, *, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x_norm1 = self.norm1(x)
        x_norm1 = x_norm1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out = self.attn(x_norm1, x_norm1, x_norm1, need_weights=False)[0]
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm2 = self.norm2(x)
        x_norm2 = x_norm2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class DiT(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int = 256,
        depth: int = 8,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.use_checkpointing = bool(use_checkpointing)

        self.x_embed = nn.Conv2d(self.in_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

        num_patches = (self.input_size // self.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.hidden_size), requires_grad=True)
        self.t_embedder = TimestepEmbedder(self.hidden_size)

        self.blocks = nn.ModuleList(
            [DiTBlock(self.hidden_size, int(num_heads), mlp_ratio=float(mlp_ratio)) for _ in range(int(depth))]
        )
        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.out_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        def _basic_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.pos_embed, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: Tensor) -> Tensor:
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.x_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        t = self.t_embedder(t)

        for block in self.blocks:
            if self.training and self.use_checkpointing:
                try:
                    x = checkpoint(block, x, t, use_reentrant=False)
                except TypeError:
                    x = checkpoint(block, x, t)
            else:
                x = block(x, t)

        x = self.final_layer(x, t)
        return self.unpatchify(x)

