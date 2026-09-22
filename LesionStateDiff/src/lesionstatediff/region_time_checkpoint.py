from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .four_channel_model_loader import (
    build_smis4ch_unet,
    extract_state_dict,
    find_conv_key,
    sha256_file,
)


def remap_parent_channels(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Map old [xt, masked, soft, seg] weights to [x_tau, masked, seg, tau]."""
    state = dict(state_dict)
    key = find_conv_key(state, "conv_in.weight")
    old = state[key]
    if old.ndim != 4 or old.shape[1] != 4:
        raise ValueError(f"Expected a four-channel parent conv_in, got {tuple(old.shape)}")
    new = torch.zeros_like(old)
    new[:, 0] = old[:, 0]
    new[:, 1] = old[:, 1]
    new[:, 2] = old[:, 3]
    new[:, 3] = 0
    state[key] = new
    report = {
        "conv_in_key": key,
        "old_channel_order": ["x_t", "masked_oct", "soft_mask", "segmentation"],
        "new_channel_order": ["x_tau", "masked_oct", "segmentation", "tau"],
        "mapping": {
            "new_0_x_tau": "old_0_x_t",
            "new_1_masked_oct": "old_1_masked_oct",
            "new_2_segmentation": "old_3_segmentation",
            "new_3_tau": "zero_initialized",
        },
        "discarded_parent_channel": "old_2_soft_mask",
    }
    return state, report


def _extract_ema(payload: Any) -> dict[str, torch.Tensor] | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("ema_state_dict")
    if not isinstance(value, dict) or not value:
        return None
    try:
        return extract_state_dict(value)
    except ValueError:
        return value


def load_region_time_parent(checkpoint: str | Path, map_location: str = "cpu"):
    path = Path(checkpoint)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    parent_state = extract_state_dict(payload)
    mapped_state, channel_report = remap_parent_channels(parent_state)
    model = build_smis4ch_unet(768)
    load_result = model.load_state_dict(mapped_state, strict=True)
    ema = _extract_ema(payload)
    mapped_ema = None
    if ema is not None:
        mapped_ema, _ = remap_parent_channels(ema)
    meta = {
        "checkpoint": str(path.resolve()),
        "sha256": sha256_file(path),
        "checkpoint_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
        "loaded_layers": len(mapped_state),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "initialized_layers": [f"{channel_report['conv_in_key']} channel 3 (tau)"],
        "channel_remap": channel_report,
        "parent_epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "parent_global_step": payload.get("global_step") if isinstance(payload, dict) else None,
        "parent_config": payload.get("config") if isinstance(payload, dict) else None,
    }
    return model, mapped_ema, payload, meta


def save_region_time_checkpoint(
    path: str | Path,
    model,
    *,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_loss: float,
    config: dict[str, Any],
    parent_meta: dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_loss": float(best_loss),
        "config": config,
        "parent": parent_meta,
        "fold1_test_used": False,
        "model_name": "OCT-Hard-Region-Time-Conditioned-Diffusion-B",
        "unet_name": "diffusers.UNet2DModel",
        "input_channels": 4,
        "output_channels": 1,
        "old_reverse_source_reinjection": False,
        "old_final_soft_blend": False,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
