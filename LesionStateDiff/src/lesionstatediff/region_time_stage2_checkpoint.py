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


def load_region_time_stage2_base(
    checkpoint: str | Path,
    *,
    expected_sha256: str | None = None,
    map_location: str = "cpu",
):
    """Strictly load an already-trained Region-Time checkpoint without remapping."""
    path = Path(checkpoint)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state = extract_state_dict(payload)
    conv_key = find_conv_key(state, "conv_in.weight")
    conv_in = state[conv_key]
    if conv_in.ndim != 4 or conv_in.shape[1] != 4:
        raise ValueError(f"Stage2 base must be four-channel, got {tuple(conv_in.shape)}")
    checkpoint_sha = sha256_file(path)
    if expected_sha256 and checkpoint_sha != expected_sha256:
        raise RuntimeError(f"Checkpoint SHA mismatch: {checkpoint_sha} != {expected_sha256}")
    if not isinstance(payload, dict):
        raise ValueError("Stage2 base checkpoint must contain metadata")
    if payload.get("model_name") != "OCT-Hard-Region-Time-Conditioned-Diffusion-B":
        raise ValueError(f"Unexpected model_name: {payload.get('model_name')}")
    if int(payload.get("epoch", -1)) != 15:
        raise ValueError(f"Stage2 must start from total epoch 15, got {payload.get('epoch')}")
    config = payload.get("config") or {}
    expected_inputs = ["x_tau", "hard_masked_oct", "segmentation", "tau_normalized"]
    if config.get("input_channels") != expected_inputs:
        raise ValueError(f"Unexpected Region-Time input contract: {config.get('input_channels')}")
    model = build_smis4ch_unet(768)
    result = model.load_state_dict(state, strict=True)
    meta: dict[str, Any] = {
        "checkpoint": str(path.resolve()),
        "sha256": checkpoint_sha,
        "checkpoint_keys": sorted(payload.keys()),
        "loaded_layers": len(state),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "channel_remap_applied": False,
        "conv_in_key": conv_key,
        "conv_in_shape": list(conv_in.shape),
        "input_channels": expected_inputs,
        "output_channels": 1,
        "base_total_epoch": int(payload["epoch"]),
        "base_global_step": int(payload.get("global_step", 0)),
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
    }
    return model, payload, meta
