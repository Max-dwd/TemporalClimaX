import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from timm.layers import DropPath
from omegaconf import DictConfig
from functools import partial
import lightning.pytorch as pl

import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
#from einops import rearrange, repeat
from timm.models.layers import DropPath

from performer_pytorch import Performer

# Calling

def get_model(cfg: DictConfig):
    # Create model based on configuration
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
    print(model_kwargs)
    if cfg.model.type == "TemporalClimaX":
        model = TemporalClimaX(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model


# === Vision Transformer Block ===
# adapted from https://github.com/microsoft/ClimaX
"""
Pre‑Norm, DropPath, MLP with GELU.
"""
class ViTBlock(nn.Module):
    """Pre‑Norm Vision‑Transformer block with multi‑head self‑attention and MLP."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, drop: float = 0.1, drop_attn: float = 0.1,
                 drop_path: float = 0.1, act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads,
                                          bias=qkv_bias,
                                          dropout=drop_attn,
                                          batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x: (B, N, D)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MLP(nn.Module):
    """Simple 2‑layer feed‑forward MLP"""
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None,
                 act_layer: nn.Module = nn.GELU, drop: float = 0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
class ViTTemporalEncoder(nn.Module):
    def __init__(self, depth_t=4, dim=768, num_heads=12,
                 mlp_ratio=4, drop=0.1, drop_path=0.1):
        super().__init__()
        self.layers = nn.Sequential(*[
            ViTBlock(dim, num_heads,
                     mlp_ratio=mlp_ratio,
                     drop=drop,
                     drop_path=drop_path * i / depth_t)
            for i in range(depth_t)
        ])
    def forward(self, z):          # (B, T, D)
        return self.layers(z)
    

# === Time Positional Embedding ===
# adapted from adapted from https://github.com/microsoft/ClimaX
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
    
def _init_sinusoidal(t_pe: nn.Parameter) -> None:
    """Overwrite `t_pe` with a 1-D sin-cos positional embedding."""
    pe = get_1d_sincos_pos_embed_from_grid(t_pe.shape[-1],
                                           np.arange(t_pe.shape[1]))
    with torch.no_grad():
        t_pe.copy_(torch.from_numpy(pe).float().unsqueeze(0))


# === TemporalClimaX Backbone ===
class TemporalClimaX(nn.Module):
    def __init__(
        self,
        n_input_channels=5,
        n_output_channels=2,
        img_size=(48, 72),
        patch_size=4,
        time_history=120,
        embed_dim=1024,
        spatial_depth=12,
        temporal_depth=12,
        heads=12,
        mlp_ratio=4.0,
        drop=0.1,
        drop_path=0.1,
    ):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W
        self.T = time_history
        D = embed_dim
        self.P_H = img_size[0] // patch_size
        self.P_W = img_size[1] // patch_size
        self.L   = self.P_H * self.P_W
        self.p = patch_size

        # 1) per-variable Conv patch embeds
        self.patch_embed = nn.ModuleList([
            nn.Conv2d(1, D, kernel_size=patch_size, stride=patch_size) 
            for _ in range(n_input_channels)
        ])

        # 2) var-ID + spatial + time PEs
        self.var_id_pe = nn.Parameter(torch.randn(n_input_channels, D))
        self.pos_spatial = nn.Parameter(torch.randn(self.L, D))
        self.pos_time = nn.Parameter(torch.zeros(1, self.T, D))
        _init_sinusoidal(self.pos_time) 

        # 3) variable aggregation
        self.agg_q = nn.Parameter(torch.zeros(1, D))
        self.agg_att = nn.MultiheadAttention(D, heads, batch_first=True)

        # 4) spatial ViT blocks
        self.spatial_encoder = nn.Sequential(*[
            ViTBlock(D, heads, mlp_ratio=4, drop_path=0.1*(i/spatial_depth))
            for i in range(spatial_depth)
        ])

        self.norm_spatial = nn.LayerNorm(D)

        # 5) temporal ViT blocks
        self.temporal_encoder = ViTTemporalEncoder(
            depth_t=temporal_depth//2,
            dim=D,
            num_heads=heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
        )

        # 6) decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_output_channels * self.H * self.W)
        )
        
    def forward(self, x):
        B, T, V, H, W = x.shape
        D = self.patch_embed[0].out_channels

        # ——— 1) spatial tokenisation per (B,T) frame ———
        xt = x.view(B * T, V, H, W)                # B*T, V, H, W
        tokens = []
        for v in range(V):
            z = self.patch_embed[v](xt[:, v:v+1])  # B*T, D, H//p, W//p
            z = z.flatten(2).transpose(1, 2)       # B*T, L, D, where L = (H // p) * (W // p)
            z = z + self.var_id_pe[v]              # add variable PE
            tokens.append(z)
        z = torch.stack(tokens, dim=2)             # B*T, L, V, D
        B_T, L, V, D = z.shape

        # ——— 2) variable aggregation ———
        z = z.reshape(B_T * L, V, D)            # B*T*L, V, D
        q = self.agg_q.expand(B_T * L, 1, D)    # q
        z, _ = self.agg_att(q, z, z)            # ⭐️ B*T*L, 1, D
        z = z.squeeze(1).reshape(B_T, L, D)     # B*T, L, D

        # ——— 3) spatial self-attention ———
        z = z + self.pos_spatial                # add patch PE
        z = self.spatial_encoder(z)             # ⭐️ B*T, L, D
        z = self.norm_spatial(z)
        
        # ——— 4) mean pooling on L, then temporal self-attention ———
        z = z.mean(dim=1)                       # B*T, D
        z = z.view(B, T, D)                     # B, T, D
        z = z + self.pos_time                   # add time PE
        z = self.temporal_encoder(z)            # ⭐️ B, T, D

        # ——— 5) decode (T, D) to (T, 2, H, W) ———
        out = self.decoder(z)                   # B, T, 2*H*W
        out = out.view(B, T, 2, H, W)           # B, T, 2, H, W
        return out
