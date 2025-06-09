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
        # --- Register heat map once ---
        for m in model.modules():
            if isinstance(m, Heat2D):
                m.register_resolution(model.P_H, model.P_W)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model


# === vHeat Spatial Attention ===
# adapted from https://github.com/MzeroMiko/vHeat
class Heat2D(nn.Module):
    """
    du/dt -k(d2u/dx2 + d2u/dy2) = 0;
    du/dx_{x=0, x=a} = 0
    du/dy_{y=0, y=b} = 0
    =>
    A_{n, m} = C(a, b, n==0, m==0) * sum_{0}^{a}{ sum_{0}^{b}{\phi(x, y)cos(n\pi/ax)cos(m\pi/by)dxdy }}
    core = cos(n\pi/ax)cos(m\pi/by)exp(-[(n\pi/a)^2 + (m\pi/b)^2]kt)
    u_{x, y, t} = sum_{0}^{\infinite}{ sum_{0}^{\infinite}{ core } }
    
    assume a = N, b = M; x in [0, N], y in [0, M]; n in [0, N], m in [0, M]; with some slight change
    => 
    (\phi(x, y) = linear(dwconv(input(x, y))))
    A(n, m) = DCT2D(\phi(x, y))
    u(x, y, t) = IDCT2D(A(n, m) * exp(-[(n\pi/a)^2 + (m\pi/b)^2])**kt)
    """    
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )

        # ── ONE-SHOT lookup table build (CPU fp32 by default) ──
        #    callers can overwrite via register_resolution() afterwards.
        self.register_resolution(res, res)

    # --------------------------------------------------------------------- #
    # Public helper – call once after constructing the module (or whenever
    # you want to switch grid size *outside* of the forward-pass).
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def register_resolution(self, H: int, W: int,
                            device: torch.device | str | None = None,
                            dtype : torch.dtype         | None = None):

        device = torch.device(device) if device is not None else torch.device("cpu")
        dtype  = dtype or torch.float32

        cos_n = self.get_cos_map(H, device=device, dtype=dtype)     # (H,H)
        cos_m = self.get_cos_map(W, device=device, dtype=dtype)     # (W,W)
        k_exp = self.get_decay_map((H, W), device=device, dtype=dtype)  # (H,W)

        for name, tensor in [("weight_cosn", cos_n),
                            ("weight_cosm", cos_m),
                            ("weight_exp",  k_exp)]:
            if name in self._buffers:
                # ‼ If size changed, just replace the buffer
                if self._buffers[name].shape != tensor.shape:
                    self._buffers[name] = tensor
                else:
                    self._buffers[name].copy_(tensor)
            else:
                self.register_buffer(name, tensor, persistent=False)

        self.H_res, self.W_res = H, W
    
    @torch.no_grad()
    def infer_init_heat2d(self, freq: torch.Tensor):
        """
        Pre-computes the *powered* decay map k_exp for inference:
          k_exp[n,m] = weight_exp[n,m] ** to_k(freq)[n,m]

        freq : (N_pixels, hidden_dim) or any shape you choose
        """
        # 1) get the raw decay map that we already buffered
        #    shape = (H_res, W_res)
        decay = self.weight_exp         # buffer from register_resolution

        # 2) compute the exponent per mode
        #    to_k(freq) must produce shape (H_res * W_res, ) or similar
        k_factor = self.to_k(freq)      # e.g. (H*W, hidden_dim) or (H, W, hidden_dim)

        # 3) broadcast decay → (H, W, 1) and raise to k_factor
        k_exp = torch.pow(decay[..., None], k_factor)

        # 4) register as a non-trainable buffer for forward uses:
        self.register_buffer("k_exp", k_exp, persistent=False)
        # and drop the to_k network since you won’t need it again
        del self.to_k
        self.infer_mode = True

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        # cos((x + 0.5) / N * n * \pi) which is also the form of DCT and IDCT
        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * f(x) )
        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * F(n) )
        # returns: (Res_n, Res_x)
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        # exp(-[(n\pi/a)^2 + (m\pi/b)^2])
        # returns: (Res_h, Res_w)
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    # --------------------------------------------------------------------- #
    # forward – pure tensor arithmetic; no allocation of new tables
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, H, W = x.shape
        if (H, W) != (self.H_res, self.W_res):
            raise ValueError(
                f"Heat2D was registered for {(self.H_res, self.W_res)} but "
                f"got input {(H, W)}; call `register_resolution` first.")

        # move lookup tables if tensor is on a different device / dtype
        if self.weight_cosn.device != x.device or self.weight_cosn.dtype != x.dtype:
            self.weight_cosn = self.weight_cosn.to(device=x.device, dtype=x.dtype)
            self.weight_cosm = self.weight_cosm.to(device=x.device, dtype=x.dtype)
            self.weight_exp  = self.weight_exp.to(device=x.device, dtype=x.dtype)

        x = self.dwconv(x)  # (B,C,H,W)
        x = self.linear(x.permute(0, 2, 3, 1))        # (B,H,W,2C)
        x, z = x.chunk(2, dim=-1)                     # split

        # ---- forward spectral diffusion (unchanged maths) ----
        N, M = self.weight_cosn.size(0), self.weight_cosm.size(0)

        x = F.conv1d(x.reshape(B, H, -1),   
                     self.weight_cosn.reshape(N, H, 1))
        x = F.conv1d(x.reshape(-1, W, C),   
                     self.weight_cosm.reshape(M, W, 1)).reshape(B, N, M, C)

        if self.infer_mode:
            x = torch.einsum("bnmc,nmc->bnmc", x, self.k_exp)
        else:
            decay = torch.pow(self.weight_exp[..., None], self.to_k(freq_embed))
            x = torch.einsum("bnmc,nmc->bnmc", x, decay)

        x = F.conv1d(x.reshape(B, N, -1),   
                     self.weight_cosn.t().reshape(H, N, 1))
        x = F.conv1d(x.reshape(-1, M, C),   
                     self.weight_cosm.t().reshape(W, M, 1)).reshape(B, H, W, C)

        x = self.out_norm(x)
        x = x * F.silu(z)
        x = self.out_linear(x).permute(0, 3, 1, 2).contiguous()
        return x
    
class HeatBlock(nn.Module):
    def __init__(
        self,
        res: int = 14,
        infer_mode = False,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        mlp_ratio: float = 4.0,
        post_norm = True,
        layer_scale = None,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.op = Heat2D(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None
        
        self.infer_mode = infer_mode
        
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)

    def _forward(self, x, freq_embed):
        # post-norm branch (most common)
        if self.post_norm:
            y = self.op(x, freq_embed)                         # (B,C,H,W)
            y = self._norm_cl(y, self.norm1)                   # LayerNorm
            if self.layer_scale:
                y = self.gamma1[:, None, None] * y
            x = x + self.drop_path(y)

            if self.mlp_branch:
                y = self._mlp_cl(self._norm_cl(x, self.norm2), self.mlp)
                if self.layer_scale:
                    y = self.gamma2[:, None, None] * y
                x = x + self.drop_path(y)
            return x

        # pre-norm branch (rare for HeatBlock but keep it correct)
        y = self.op(self._norm_cl(x, self.norm1), freq_embed)
        if self.layer_scale:
            y = self.gamma1[:, None, None] * y
        x = x + self.drop_path(y)

        if self.mlp_branch:
            y = self._mlp_cl(self._norm_cl(x, self.norm2), self.mlp)
            if self.layer_scale:
                y = self.gamma2[:, None, None] * y
            x = x + self.drop_path(y)
        return x

    # ---- helper ----------------------------------------------------------
    @staticmethod
    def _norm_cl(x, norm):
        """Layer-norm on the channel dim even for 4-D tensors."""
        if x.dim() == 4:                       # (B,C,H,W) ➜ (B,H,W,C)
            x = x.permute(0, 2, 3, 1)
            x = norm(x)
            return x.permute(0, 3, 1, 2)       # back to (B,C,H,W)
        return norm(x)                         # 3-D case

    @staticmethod
    def _mlp_cl(x, mlp):
        if x.dim() == 4:
            shp = x.shape                      # save (B,C,H,W)
            x = x.permute(0, 2, 3, 1).reshape(-1, shp[1])  # (B*H*W,C)
            x = mlp(x)
            x = x.view(shp[0], shp[2], shp[3], shp[1]).permute(0, 3, 1, 2)
            return x
        return mlp(x)                          # 3-D case
        
    def forward(self, input: torch.Tensor, freq_embed=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, freq_embed)
        else:
            return self._forward(input, freq_embed)

class SpatialHeatEncoder(nn.Module):
    def __init__(self, depth, res_h, res_w, dim, mlp_ratio, drop_path):
        super().__init__()
        self.res_h, self.res_w = res_h, res_w          # P_H, P_W
        self.blocks = nn.ModuleList([
            HeatBlock(
                res=res_h,             # height of patch grid
                hidden_dim=dim,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path * i / max(1, depth - 1),
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        for blk in self.blocks:
            # P_H, P_W come from SpatialHeatEncoder.__init__
            blk.op.register_resolution(self.res_h, self.res_w)

    def forward(self, x):                  # x: (B*T, L, D)
        B_T, L, D = x.shape
        H, W = self.res_h, self.res_w

        x = x.transpose(1, 2).reshape(B_T, D, H, W)     # (B*T, D, H, W)

        # ------------ new lines ------------
        # use the (learned) channel features themselves as a per-pixel “frequency” code
        # shape: (B*T, H, W, D)  — exactly what Heat2D expects
                # --- frequency embedding ---
        # Heat2D expects (H, W, D) without a batch dim.
        # A simple, stable choice is the mean over the batch/time axis.
        # Result: freq_embed.shape == (H, W, D)
        freq_embed = x.mean(dim=0).permute(1, 2, 0)            

        for blk in self.blocks:
            x = blk(x, freq_embed=freq_embed)           

        x = self.norm(x.flatten(2).transpose(1, 2))      # back to (B*T, L, D)
        return x


# === Vision Transformer Block ===
# adapted from https://github.com/microsoft/ClimaX
"""
Pre‑Norm, DropPath, MLP with GELU.
"""
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


# === Performer Encoder ===
# adapted from https://github.com/lucidrains/performer-pytorch
class SpatioTemporalPerformerEncoder(nn.Module):
    """
    Attends over the flattened (T × L) spatial tokens in linear time.
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        ff_mult: float = 4.0,
        local_attn_heads: int = 0,
        causal: bool = False,
        kernel_features: int = 256,
        ff_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ff_glu: bool = False,
    ):
        """
        Args:
          dim            – token embedding dim (same D from spatial encoder)
          depth          – number of Performer layers
          heads          – attention heads
          ff_mult        – MLP expansion factor
          local_attn_heads – if >0, use exact local attention on these heads
          causal         – whether to mask future tokens
          kernel_size    – projection dimension for FAVOR+ sketches
        """
        super().__init__()
        self.performer = Performer(
            dim,                   
            depth,                   
            heads,                   
            dim // heads,            # dim per head

            # core FAVOR+ parameter
            nb_features = kernel_features,

            # feed-forward config
            ff_mult = ff_mult,
            ff_dropout = ff_dropout,
            ff_glu = False,        # gated-relu MLP?

            # attention dropout
            attn_dropout = attn_dropout,
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, L, D)
        → flatten to (B, T*L, D)
        → Performer
        → reshape back to (B, T, L, D)
        """
        B, T, L, D = x.shape
        x_flat = x.view(B, T * L, D)        # (B, T*L, D)
        y_flat = self.performer(x_flat)     # (B, T*L, D)
        return y_flat.view(B, T, L, D)      # (B, T, L, D)
        

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
        """
        Optional: Use vHeat blocks as spatial encoder
        self.spatial_encoder = SpatialHeatEncoder(
                depth=depth,
                res_h=self.P_H,
                res_w=self.P_W,
                dim=D,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path
        )
        """

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
        """
        Optional: Use performer as spatio-temporal encoder over (TxL) tokens
        self.temporal_encoder = SpatioTemporalPerformerEncoder(
            dim = D,
            depth = temporal_depth,
            heads = heads,
            ff_mult = 4,
            local_attn_heads = 0,      
            causal = False,             
            kernel_features = 256,     
        )
        """

        # 6) decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_output_channels * self.H * self.W)
        )
        """
        Optional: Performer decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_output_channels * self.p, self.p)
        )
        """
        
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