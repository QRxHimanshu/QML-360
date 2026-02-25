#data generataion scripts for QML-360
# generate_sph_cifar10_erp.py
# Generates SPH-CIFAR10-style ERP proxy dataset and saves to .pt

from __future__ import annotations
import math
import os
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from torchvision import datasets, transforms
except Exception:
    datasets = None
    transforms = None


@dataclass
class GenConfig:
    # Output ERP resolution
    H: int = 64
    W: int = 128

    # Patch size (CIFAR native 32)
    patch_size: int = 32

    # Augmentations
    yaw_shift: bool = True
    max_yaw_shift: Optional[int] = None  # default W//2
    pitch_warp: bool = False
    pitch_warp_amp: float = 2.0
    pitch_warp_freq: float = 1.0

    # CIFAR augmentations before placement 
    random_crop: bool = True
    random_flip: bool = True

    # Placement strategy
    place_near_equator: bool = True
    equator_band: float = 0.5  # fraction of height band around equator (e.g., 0.5 => middle 50%)

    # Normalization (store normalized or raw)
    # If store_normalized=True, saved X already normalized.
    store_normalized: bool = False
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std:  Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

    # Output
    out_dir: str = "./data_spherical"
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def pitch_warp_rgb(x: torch.Tensor, amp: float, freq: float) -> torch.Tensor:
    """
    x: [3,H,W]
    """
    C, H, W = x.shape
    phase = random.uniform(0, 2 * math.pi)

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing="ij",
    )
    x01 = (xx + 1) / 2.0
    disp_pix = amp * torch.sin(2 * math.pi * freq * x01 + phase)
    disp_norm = disp_pix / (H / 2.0)

    grid = torch.stack([xx, yy + disp_norm], dim=-1).unsqueeze(0)  # [1,H,W,2]
    xb = x.unsqueeze(0)  # [1,3,H,W]
    warped = F.grid_sample(xb, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped.squeeze(0)  # [3,H,W]


def place_patch_on_erp(img: torch.Tensor, cfg: GenConfig) -> torch.Tensor:
    """
    img: [3,32,32] in [0,1]
    returns ERP: [3,H,W]
    """
    C, ph, pw = img.shape
    H, W = cfg.H, cfg.W
    canvas = torch.zeros((3, H, W), dtype=torch.float32)

    # choose placement
    if cfg.place_near_equator:
        band = int(H * cfg.equator_band)
        top = (H - band) // 2
        r0 = random.randint(top, top + band - ph)
        c0 = W // 4 - pw // 2
    else:
        r0 = random.randint(0, H - ph)
        c0 = random.randint(0, W - pw)

    r0 = max(0, min(H - ph, r0))
    c0 = max(0, min(W - pw, c0))
    canvas[:, r0:r0 + ph, c0:c0 + pw] = img

    # yaw shift (longitude wrap)
    if cfg.yaw_shift:
        max_yaw = cfg.max_yaw_shift if cfg.max_yaw_shift is not None else (W // 2)
        shift = random.randint(-max_yaw, max_yaw)
        canvas = torch.roll(canvas, shifts=shift, dims=2)

    # optional pitch warp
    if cfg.pitch_warp:
        canvas = pitch_warp_rgb(canvas, cfg.pitch_warp_amp, cfg.pitch_warp_freq)

    return canvas


def generate_split(split: str, cfg: GenConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    if datasets is None or transforms is None:
        raise RuntimeError("torchvision is required. Please install torchvision.")

    train = (split == "train")

    # base transform 
    tfms = []
    if train and cfg.random_crop:
        tfms.append(transforms.RandomCrop(32, padding=4))
    if train and cfg.random_flip:
        tfms.append(transforms.RandomHorizontalFlip())
    tfms.append(transforms.ToTensor())  # [3,32,32] in [0,1]
    if cfg.store_normalized:
        tfms.append(transforms.Normalize(cfg.mean, cfg.std))

    tfm = transforms.Compose(tfms)
    base = datasets.CIFAR10(root="./data", train=train, download=True, transform=tfm)

    X = torch.empty((len(base), 3, cfg.H, cfg.W), dtype=torch.float32)
    y = torch.empty((len(base),), dtype=torch.long)

    for i in tqdm(range(len(base)), desc=f"Generating {split}"):
        img, label = base[i]  # img is either raw or normalized depending on store_normalized
        # If normalized, placement still works fine (values not in 0..1, but model can handle).
        X[i] = place_patch_on_erp(img, cfg)
        y[i] = int(label)

    return X, y


def main():
    cfg = GenConfig()
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    Xtr, ytr = generate_split("train", cfg)
    Xte, yte = generate_split("test", cfg)

    norm_tag = "_norm" if cfg.store_normalized else "_raw"
    out_train = os.path.join(cfg.out_dir, f"sph_cifar10_erp_train{norm_tag}_H{cfg.H}_W{cfg.W}.pt")
    out_test  = os.path.join(cfg.out_dir, f"sph_cifar10_erp_test{norm_tag}_H{cfg.H}_W{cfg.W}.pt")

    torch.save({"X": Xtr, "y": ytr, "cfg": cfg.__dict__}, out_train)
    torch.save({"X": Xte, "y": yte, "cfg": cfg.__dict__}, out_test)

    print("Saved:")
    print(" ", out_train)
    print(" ", out_test)


if __name__ == "__main__":
    main()