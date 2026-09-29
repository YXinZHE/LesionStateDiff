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


EXPECTED_INPUTS = ["x_tau", "hard_masked_oct", "segmentation", "tau_normalized"]


def load_region_time_stage3_base(
    checkpoint: str | Path,
    *,
    expected_sha256: str | None = None,
    map_location: str = "cpu",
):
    """Strictly load the required Stage2 epoch-5 (total epoch 20) model."""
    path = Path(checkpoint)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Stage3 base checkpoint must contain metadata")

    state = extract_state_dict(payload)
    conv_key = find_conv_key(state, "conv_in.weight")
    conv_in = state[conv_key]
    if conv_in.ndim != 4 or conv_in.shape[1] != 4:
        raise ValueError(f"Stage3 base must be four-channel, got {tuple(conv_in.shape)}")

    checkpoint_sha = sha256_file(path)
    if expected_sha256 and checkpoint_sha != expected_sha256:
        raise RuntimeError(f"Checkpoint SHA mismatch: {checkpoint_sha} != {expected_sha256}")
    if payload.get("model_name") != "OCT-Hard-Region-Time-Conditioned-Diffusion-B":
        raise ValueError(f"Unexpected model_name: {payload.get('model_name')}")
    if int(payload.get("epoch", -1)) != 20:
        raise ValueError(f"Stage3 must start from total epoch 20, got {payload.get('epoch')}")
    if int(payload.get("global_step", -1)) != 16000:
        raise ValueError(f"Stage3 base must be global_step 16000, got {payload.get('global_step')}")

    config = payload.get("config") or {}
    actual_inputs = config.get("input_channels")
    input_contract_source = "config.input_channels"
    if actual_inputs is None:
        actual_inputs = (config.get("base_checkpoint_meta") or {}).get("input_channels")
        input_contract_source = "config.base_checkpoint_meta.input_channels"
    if actual_inputs != EXPECTED_INPUTS:
        raise ValueError(f"Unexpected Region-Time input contract: {actual_inputs}")

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
        "input_channels": EXPECTED_INPUTS,
        "input_contract_source": input_contract_source,
        "output_channels": 1,
        "base_total_epoch": int(payload["epoch"]),
        "base_global_step": int(payload["global_step"]),
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
    }
    return model, payload, meta
