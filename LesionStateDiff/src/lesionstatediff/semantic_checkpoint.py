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
from .semantic_conditioning import SemanticConditionedUNet


EXPECTED_INPUTS = ["x_tau", "hard_masked_oct", "segmentation", "tau_normalized"]
B3_MODEL_NAME = "OCT-Hard-Region-Time-B3-Spatial-Class-Semantic-Conditioned-Diffusion"


def load_region_time_b3_base(
    checkpoint: str | Path,
    *,
    expected_sha256: str | None = None,
    map_location: str = "cpu",
):
    """Strictly load B1 Stage3 epoch 4 and zero-initialize the B3 branch."""
    path = Path(checkpoint)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("B3 base checkpoint must contain metadata")
    state = extract_state_dict(payload)
    conv_key = find_conv_key(state, "conv_in.weight")
    conv_in = state[conv_key]
    if conv_in.ndim != 4 or tuple(conv_in.shape[1:]) != (4, 3, 3):
        raise ValueError(f"B3 base must keep four-channel conv_in, got {tuple(conv_in.shape)}")

    checkpoint_sha = sha256_file(path)
    if expected_sha256 and checkpoint_sha != expected_sha256:
        raise RuntimeError(f"Checkpoint SHA mismatch: {checkpoint_sha} != {expected_sha256}")
    if payload.get("model_name") != "OCT-Hard-Region-Time-Conditioned-Diffusion-B":
        raise ValueError(f"Unexpected model_name: {payload.get('model_name')}")
    if int(payload.get("epoch", -1)) != 24 or int(payload.get("global_step", -1)) != 22400:
        raise ValueError(
            "B3 must start from Stage3 epoch4 metadata epoch/global_step=24/22400, got "
            f"{payload.get('epoch')}/{payload.get('global_step')}"
        )
    parent_config = payload.get("config") or {}
    if parent_config.get("task_name") != (
        "2026-09-02-OCT-HardRegionTime-Stage3-8000Exposure-5Epoch-Refinement"
    ):
        raise ValueError(f"B3 base is not the audited Stage3 run: {parent_config.get('task_name')}")
    actual_inputs = parent_config.get("input_channels")
    if actual_inputs is None:
        actual_inputs = (parent_config.get("base_checkpoint_meta") or {}).get("input_channels")
    if actual_inputs != EXPECTED_INPUTS:
        raise ValueError(f"Unexpected Region-Time input contract: {actual_inputs}")

    unet = build_smis4ch_unet(768)
    result = unet.load_state_dict(state, strict=True)
    model = SemanticConditionedUNet(unet, embedding_dim=32, mid_channels=512)
    if not model.semantic_encoder.projection_is_zero():
        raise RuntimeError("B3 semantic projection is not zero-initialized")
    new_keys = [f"semantic_encoder.{key}" for key in model.semantic_encoder.state_dict()]
    meta: dict[str, Any] = {
        "checkpoint": str(path.resolve()),
        "sha256": checkpoint_sha,
        "checkpoint_keys": sorted(payload.keys()),
        "loaded_layers": len(state),
        "base_missing_keys": list(result.missing_keys),
        "base_unexpected_keys": list(result.unexpected_keys),
        "new_module_missing_keys": new_keys,
        "unexpected_keys": [],
        "conv_in_key": conv_key,
        "conv_in_shape": list(conv_in.shape),
        "input_channels": EXPECTED_INPUTS,
        "output_channels": 1,
        "base_total_epoch": 24,
        "base_global_step": 22400,
        "mid_block_channels": 512,
        "semantic_embedding_dim": 32,
        "semantic_projection_zero_initialized": True,
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
    }
    return model, payload, meta


def save_semantic_checkpoint(
    path: str | Path,
    model: SemanticConditionedUNet,
    *,
    optimizer,
    scheduler,
    epoch: int,
    b3_epoch: int,
    global_step: int,
    best_loss: float,
    config: dict[str, Any],
    parent_meta: dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": model.unet.state_dict(),
        "semantic_state_dict": model.semantic_encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch),
        "b3_epoch": int(b3_epoch),
        "global_step": int(global_step),
        "best_loss": float(best_loss),
        "config": config,
        "parent": parent_meta,
        "fold1_test_used": False,
        "model_name": B3_MODEL_NAME,
        "unet_name": "diffusers.UNet2DModel",
        "input_channels": 4,
        "output_channels": 1,
        "semantic_num_classes": 5,
        "semantic_embedding_dim": 32,
        "semantic_mid_channels": 512,
        "semantic_injection": "pre_mid_block_addition",
        "source_reinjection_used": False,
        "soft_blend_used": False,
        "posthoc_mask_restore_used": False,
        "tau_rule": {"Background": 0, "LM": 0, "FC": "t", "LC": "t", "VV": "t"},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def reload_semantic_checkpoint(
    checkpoint: str | Path,
    *,
    map_location: str = "cpu",
):
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if payload.get("model_name") != B3_MODEL_NAME:
        raise ValueError(f"Unexpected B3 model_name: {payload.get('model_name')}")
    unet = build_smis4ch_unet(768)
    base_result = unet.load_state_dict(payload["model_state_dict"], strict=True)
    model = SemanticConditionedUNet(unet, embedding_dim=32, mid_channels=512)
    semantic_result = model.semantic_encoder.load_state_dict(
        payload["semantic_state_dict"], strict=True
    )
    return model, payload, {
        "base_missing_keys": list(base_result.missing_keys),
        "base_unexpected_keys": list(base_result.unexpected_keys),
        "semantic_missing_keys": list(semantic_result.missing_keys),
        "semantic_unexpected_keys": list(semantic_result.unexpected_keys),
    }
