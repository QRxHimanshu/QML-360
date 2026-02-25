# supergraph_module.py
# Shared module: ERP -> node map (CNN + posenc [+ grads]) -> supergraph pooling + edges

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Config for Supergraph
# -----------------------------
@dataclass
class SupergraphConfig:
    H: int = 64
    W: int = 128
    K_h: int = 8
    K_w: int = 16
    d: int = 32

    # Optional finite-difference gradient channels appended before projection
    use_grad_channels: bool = False
    grad_channels: int = 2  # dx, dy

    # If True, build and return dense adjacency [K,K]
    return_dense_adj: bool = False


# -----------------------------
# Positional encoding on sphere
# -----------------------------
def build_spherical_posenc(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Returns P of shape [1,3,H,W] with:
      p = (sinθ cosφ, sinθ sinφ, cosθ)
    where θ in [0,π], φ in [0,2π).
    """
    r = torch.arange(H, device=device).float() + 0.5
    c = torch.arange(W, device=device).float() + 0.5

    theta = math.pi * (r / H)          # [H]
    phi = 2.0 * math.pi * (c / W)      # [W]
    theta = theta[:, None]             # [H,1]
    phi = phi[None, :]                 # [1,W]

    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)

    x = sin_t * cos_p
    y = sin_t * sin_p
    z = cos_t.expand(H, W)

    return torch.stack([x, y, z], dim=0).unsqueeze(0)  # [1,3,H,W]


# -----------------------------
# Supergraph edges
# -----------------------------
def build_supergraph_edges(K_h: int, K_w: int, wrap_longitude: bool = True) -> torch.Tensor:
    """
    4-neighborhood edges on K_h x K_w grid with longitude wrap.
    Returns directed edge_index [2,E] with both directions included.
    """
    edges = []
    for r in range(K_h):
        for c in range(K_w):
            u = r * K_w + c

            # horizontal neighbor
            if wrap_longitude:
                v = r * K_w + ((c + 1) % K_w)
                edges.append((u, v))
                edges.append((v, u))
            else:
                if c + 1 < K_w:
                    v = r * K_w + (c + 1)
                    edges.append((u, v))
                    edges.append((v, u))

            # vertical neighbor (no wrap)
            if r + 1 < K_h:
                v = (r + 1) * K_w + c
                edges.append((u, v))
                edges.append((v, u))

    return torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2,E]


def edge_index_to_dense_adj(edge_index: torch.Tensor, K: int) -> torch.Tensor:
    """
    Convert edge_index [2,E] to dense adjacency [K,K] (0/1).
    """
    adj = torch.zeros((K, K), dtype=torch.float32, device=edge_index.device)
    src, dst = edge_index[0], edge_index[1]
    adj[src, dst] = 1.0
    return adj


# -----------------------------
# Padding utility for spherical conv
# -----------------------------
def circular_pad_width_then_replicate_height(x: torch.Tensor, pad: int) -> torch.Tensor:
    """
    Apply circular padding in longitude (W) and replicate padding in latitude (H).
    x: [B,C,H,W]
    """
    if pad <= 0:
        return x
    x = F.pad(x, (pad, pad, 0, 0), mode="circular")
    x = F.pad(x, (0, 0, pad, pad), mode="replicate")
    return x


# -----------------------------
# Optional gradient channels
# -----------------------------
class FiniteDiffGrad(nn.Module):
    """
    Produces dx, dy in a shape-safe way:
      dx uses circular pad in width
      dy uses replicate pad in height
    Input: x [B,1,H,W] intensity channel
    Output: [B,2,H,W]
    """
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1]], dtype=torch.float32).view(1, 1, 1, 3)
        ky = torch.tensor([[-1], [0], [1]], dtype=torch.float32).view(1, 1, 3, 1)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dx = F.pad(x, (1, 1, 0, 0), mode="circular")
        dx = F.conv2d(x_dx, self.kx)

        x_dy = F.pad(x, (0, 0, 1, 1), mode="replicate")
        dy = F.conv2d(x_dy, self.ky)

        return torch.cat([dx, dy], dim=1)


# -----------------------------
# Tiny CNN encoder
# -----------------------------
class TinySphericalCNN(nn.Module):
    """
    Minimal CNN encoder ψϕ producing [B,d,H,W]
    Uses spherical padding: circular in longitude, replicate in latitude.
    """
    def __init__(self, in_ch: int, d: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, d, 3, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, d, 3, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(d)
        self.conv3 = nn.Conv2d(d, d, 3, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
            x = circular_pad_width_then_replicate_height(x, pad=1)
            x = conv(x)
            x = bn(x)
            x = F.gelu(x)
        return x


# -----------------------------
# Supergraph builder: ERP -> X, edge_index
# -----------------------------
class SupergraphBuilder(nn.Module):
    """
    Usage:
      builder = SupergraphBuilder(cfg, in_ch=1 or 3)
      X, edge_index = builder(I)  # I: [B,in_ch,H,W]
    """
    def __init__(self, cfg: SupergraphConfig, in_ch: int):
        super().__init__()
        self.cfg = cfg

        assert cfg.H % cfg.K_h == 0 and cfg.W % cfg.K_w == 0, "H,W must be divisible by K_h,K_w"
        self.t_h = cfg.H // cfg.K_h
        self.t_w = cfg.W // cfg.K_w
        self.K = cfg.K_h * cfg.K_w

        self.encoder = TinySphericalCNN(in_ch=in_ch, d=cfg.d)
        self.use_grad = cfg.use_grad_channels
        self.grad = FiniteDiffGrad() if self.use_grad else None

        in_proj = cfg.d + 3 + (cfg.grad_channels if self.use_grad else 0)
        self.proj = nn.Conv2d(in_proj, cfg.d, kernel_size=1)

        # buffers created lazily (device-aware)
        self.register_buffer("_posenc", torch.empty(0), persistent=False)
        self.register_buffer("_edge_index", torch.empty(0, dtype=torch.long), persistent=False)

    def _ensure_buffers(self, device: torch.device) -> None:
        if self._posenc.numel() == 0 or self._posenc.shape[-2:] != (self.cfg.H, self.cfg.W) or self._posenc.device != device:
            self._posenc = build_spherical_posenc(self.cfg.H, self.cfg.W, device=device)

        if self._edge_index.numel() == 0 or self._edge_index.device != device:
            self._edge_index = build_supergraph_edges(self.cfg.K_h, self.cfg.K_w, wrap_longitude=True).to(device)

    def forward(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        I: [B,in_ch,H,W]
        Returns:
          X: [B,K,d] supernode features
          edge_index: [2,E] (directed, with both directions)
          adj (optional): [K,K] dense adjacency if cfg.return_dense_adj=True else None
        """
        B, C, H, W = I.shape
        assert H == self.cfg.H and W == self.cfg.W, "Input ERP size must match cfg.H/cfg.W"

        self._ensure_buffers(I.device)

        Fmap = self.encoder(I)  # [B,d,H,W]
        P = self._posenc.expand(B, -1, -1, -1)  # [B,3,H,W]
        feats = [Fmap, P]

        if self.use_grad:
            # intensity channel for grads: if RGB, convert to gray; if 1ch, use itself
            if C == 1:
                gray = I
            else:
                gray = (0.2989 * I[:, 0:1] + 0.5870 * I[:, 1:2] + 0.1140 * I[:, 2:3])
            g = self.grad(gray)  # [B,2,H,W]
            feats.append(g)

        Hcat = torch.cat(feats, dim=1)  # [B, d+3(+2), H, W]
        Hmap = self.proj(Hcat)          # [B, d, H, W]

        # Tile pooling -> supernodes
        Xgrid = F.avg_pool2d(Hmap, kernel_size=(self.t_h, self.t_w), stride=(self.t_h, self.t_w))
        # Xgrid: [B,d,K_h,K_w] -> [B,K,d]
        X = Xgrid.permute(0, 2, 3, 1).reshape(B, self.K, self.cfg.d)

        edge_index = self._edge_index
        adj = edge_index_to_dense_adj(edge_index, self.K) if self.cfg.return_dense_adj else None
        return X, edge_index, adj