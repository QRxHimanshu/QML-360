#data generataion scripts for QML-360
# generate_sph_mnist_erp.py
# Generates SPH-MNIST-style ERP proxy dataset and saves to .pt

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

    # Digit patch size (resized MNIST digit)
    digit_size: int = 28

    # Augmentations
    yaw_shift: bool = True
    max_yaw_shift: Optional[int] = None  # default W//2
    pitch_warp: bool = False            # expensive; enable if needed
    pitch_warp_amp: float = 2.0
    pitch_warp_freq: float = 1.0

    # Placement strategy
    # If True: place near equator; else random anywhere
    place_near_equator: bool = True
    equator_jitter: int = 2

    # Output
    out_dir: str = "./data_spherical"
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def pitch_warp_2d(x: torch.Tensor, amp: float, freq: float) -> torch.Tensor:
    """
    x: [1,H,W]
    warp: y' = y + amp*sin(2π*freq*(x/W) + phase)
    """
    _, H, W = x.shape
    phase = random.uniform(0, 2 * math.pi)

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing="ij",
    )
    x01 = (xx + 1) / 2.0
    disp_pix = amp * torch.sin(2 * math.pi * freq * x01 + phase)  # [H,W]
    disp_norm = disp_pix / (H / 2.0)

    grid = torch.stack([xx, yy + disp_norm], dim=-1).unsqueeze(0)  # [1,H,W,2]
    xb = x.unsqueeze(0)  # [1,1,H,W]
    warped = F.grid_sample(xb, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped.squeeze(0)  # [1,H,W]


def make_erp_from_mnist(img_28: torch.Tensor, cfg: GenConfig) -> torch.Tensor:
    """
    img_28: [1,28,28] float in [0,1]
    returns ERP: [1,H,W]
    """
    img = img_28.squeeze(0)  # [28,28]

    # resize digit if requested
    if cfg.digit_size != 28:
        img = F.interpolate(
            img[None, None],
            size=(cfg.digit_size, cfg.digit_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    H, W = cfg.H, cfg.W
    canvas = torch.zeros((H, W), dtype=torch.float32)

    dh, dw = img.shape

    if cfg.place_near_equator:
        r0 = H // 2 - dh // 2 + random.randint(-cfg.equator_jitter, cfg.equator_jitter)
        c0 = W // 4 - dw // 2
    else:
        r0 = random.randint(0, H - dh)
        c0 = random.randint(0, W - dw)

    r0 = max(0, min(H - dh, r0))
    c0 = max(0, min(W - dw, c0))

    canvas[r0:r0 + dh, c0:c0 + dw] = img

    if cfg.yaw_shift:
        max_yaw = cfg.max_yaw_shift if cfg.max_yaw_shift is not None else (W // 2)
        shift = random.randint(-max_yaw, max_yaw)
        canvas = torch.roll(canvas, shifts=shift, dims=1)

    erp = canvas.unsqueeze(0)  # [1,H,W]

    if cfg.pitch_warp:
        erp = pitch_warp_2d(erp, cfg.pitch_warp_amp, cfg.pitch_warp_freq)

    return erp.clamp(0, 1)


def generate_split(split: str, cfg: GenConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    if datasets is None or transforms is None:
        raise RuntimeError("torchvision is required. Please install torchvision.")

    train = (split == "train")
    tfm = transforms.Compose([transforms.ToTensor()])  # [1,28,28]
    base = datasets.MNIST(root="./data", train=train, download=True, transform=tfm)

    X = torch.empty((len(base), 1, cfg.H, cfg.W), dtype=torch.float32)
    y = torch.empty((len(base),), dtype=torch.long)

    for i in tqdm(range(len(base)), desc=f"Generating {split}"):
        img, label = base[i]
        X[i] = make_erp_from_mnist(img, cfg)
        y[i] = int(label)

    return X, y


def main():
    cfg = GenConfig()
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    Xtr, ytr = generate_split("train", cfg)
    Xte, yte = generate_split("test", cfg)

    out_train = os.path.join(cfg.out_dir, f"sph_mnist_erp_train_H{cfg.H}_W{cfg.W}.pt")
    out_test  = os.path.join(cfg.out_dir, f"sph_mnist_erp_test_H{cfg.H}_W{cfg.W}.pt")

    torch.save({"X": Xtr, "y": ytr, "cfg": cfg.__dict__}, out_train)
    torch.save({"X": Xte, "y": yte, "cfg": cfg.__dict__}, out_test)

    print("Saved:")
    print(" ", out_train)
    print(" ", out_test)


if __name__ == "__main__":
    main()