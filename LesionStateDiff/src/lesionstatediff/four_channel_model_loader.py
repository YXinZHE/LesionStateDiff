from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import torch

def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def assert_allowed_parent_path(path: str | Path) -> None:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Parent checkpoint does not exist: {checkpoint}")


def build_smis4ch_unet(sample_size: int = 768):
    import diffusers
    return diffusers.UNet2DModel(
        sample_size=sample_size,
        in_channels=4,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


def clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "unet_state_dict", "state_dict", "model", "unet"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return clean_state_dict(checkpoint[key])
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return clean_state_dict(checkpoint)
    raise ValueError("Could not extract model state_dict from checkpoint")


def find_conv_key(state: Dict[str, torch.Tensor], suffix: str) -> str:
    matches = [k for k in state if k == suffix or k.endswith("." + suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one {suffix}, found {matches[:5]}")
    return matches[0]


def expand_conv_in_2ch_to_4ch(parent_state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], str]:
    state = dict(parent_state)
    key = find_conv_key(state, "conv_in.weight")
    w = state[key]
    if w.ndim != 4 or w.shape[1] != 2:
        raise ValueError(f"Parent conv_in must be [C,2,K,K], got {tuple(w.shape)} at {key}")
    new_w = torch.zeros((w.shape[0], 4, w.shape[2], w.shape[3]), dtype=w.dtype, device=w.device)
    new_w[:, 0] = w[:, 0]
    new_w[:, 1] = 0
    new_w[:, 2] = 0
    new_w[:, 3] = w[:, 1]
    state[key] = new_w
    return state, key


def load_parent_into_smis4ch(parent_checkpoint: str | Path, map_location="cpu", strict: bool = True):
    assert_allowed_parent_path(parent_checkpoint)
    ckpt = torch.load(parent_checkpoint, map_location=map_location)
    state = extract_state_dict(ckpt)
    expanded, conv_key = expand_conv_in_2ch_to_4ch(state)
    model = build_smis4ch_unet(768)
    result = model.load_state_dict(expanded, strict=strict)
    return model, {"conv_in_key": conv_key, "load_result": str(result)}
