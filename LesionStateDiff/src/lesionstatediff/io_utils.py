from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from .constants import CLASS_INDEX, ORIGINAL_SIZE, PAD, PADDED_SIZE, VALID_CLASS_VALUES


def require_no_test_path(path: str | Path, label: str = "path") -> None:
    resolved = Path(path).resolve()
    if any(part.lower() == "test" for part in resolved.parts):
        raise ValueError(f"Refusing {label} containing a test path component: {resolved}")


def patient_id_from_stem(stem: str) -> str:
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


def read_grayscale_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale image at {path}, got {arr.shape}")
    return arr


def _threshold_channel(channel: np.ndarray) -> np.ndarray:
    return channel if channel.dtype == np.bool_ else channel > 0


def rgba_or_chw_to_class_index(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        values = {int(v) for v in np.unique(arr)}
        if not values.issubset(VALID_CLASS_VALUES):
            raise ValueError(f"Invalid class-index values: {sorted(values)}")
        return arr.astype(np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported mask shape: {arr.shape}")
    if arr.shape[0] == 4 and arr.shape[-1] != 4:
        ch = arr[:4]
    elif arr.shape[-1] >= 4:
        ch = np.moveaxis(arr[..., :4], -1, 0)
    else:
        raise ValueError(f"Cannot decode four OCT mask channels from {arr.shape}")
    out = np.zeros(ch.shape[1:], dtype=np.uint8)
    lm = _threshold_channel(ch[0])
    out[lm] = CLASS_INDEX["LM"]
    occupied = lm.copy()
    for idx in (2, 3, 4):
        raw = _threshold_channel(ch[idx - 1])
        allowed = raw & ~occupied & ~lm
        out[allowed] = idx
        occupied |= allowed
    return out


def read_class_index_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        arr = data["mask"] if "mask" in data.files else data[data.files[0]]
    else:
        arr = np.asarray(Image.open(path))
    out = rgba_or_chw_to_class_index(arr)
    values = {int(v) for v in np.unique(out)}
    if not values.issubset(VALID_CLASS_VALUES):
        raise ValueError(f"Invalid decoded mask values: {sorted(values)}")
    return out


def pad_image_reflect(arr: np.ndarray) -> np.ndarray:
    if tuple(arr.shape) != (ORIGINAL_SIZE, ORIGINAL_SIZE):
        raise ValueError(f"Expected image {ORIGINAL_SIZE}x{ORIGINAL_SIZE}, got {arr.shape}")
    out = np.pad(arr, ((PAD, PAD), (PAD, PAD)), mode="reflect")
    assert out.shape == (PADDED_SIZE, PADDED_SIZE)
    return out


def pad_mask_zero(arr: np.ndarray) -> np.ndarray:
    if tuple(arr.shape) != (ORIGINAL_SIZE, ORIGINAL_SIZE):
        raise ValueError(f"Expected mask {ORIGINAL_SIZE}x{ORIGINAL_SIZE}, got {arr.shape}")
    values = {int(v) for v in np.unique(arr)}
    if not values.issubset(VALID_CLASS_VALUES):
        raise ValueError(f"Invalid mask values before pad: {sorted(values)}")
    out = np.pad(arr.astype(np.uint8), ((PAD, PAD), (PAD, PAD)), mode="constant", constant_values=0)
    assert out.shape == (PADDED_SIZE, PADDED_SIZE)
    return out


def image_to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0) * 2.0 - 1.0


def mask_to_tensor_raw_class(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)


def crop_768_to_750(t: torch.Tensor) -> torch.Tensor:
    return t[..., PAD:-PAD, PAD:-PAD]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
