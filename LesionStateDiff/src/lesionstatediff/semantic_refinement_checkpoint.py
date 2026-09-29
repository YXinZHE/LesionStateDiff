from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .four_channel_model_loader import build_smis4ch_unet, sha256_file
from .semantic_checkpoint import (
    B3_MODEL_NAME,
    EXPECTED_INPUTS,
    save_semantic_checkpoint,
)
from .semantic_conditioning import SemanticConditionedUNet


EXPECTED_PARENT_EPOCH = 27
EXPECTED_PARENT_B3_EPOCH = 3
EXPECTED_PARENT_GLOBAL_STEP = 27200


def semantic_norms(model: SemanticConditionedUNet) -> dict[str, float]:
    encoder = model.semantic_encoder
    return {
        "embedding_norm": float(encoder.embedding.weight.detach().float().norm().cpu()),
        "projection_weight_norm": float(
            encoder.projection.weight.detach().float().norm().cpu()
        ),
        "projection_bias_norm": float(
            encoder.projection.bias.detach().float().norm().cpu()
        ),
    }


def load_b3_refinement_parent(
    checkpoint: str | Path,
    *,
    expected_sha256: str,
    map_location: str = "cpu",
) -> tuple[SemanticConditionedUNet, dict[str, Any], dict[str, Any]]:
    """Strictly restore the trained B3 epoch-3 model without optimizer state."""
    path = Path(checkpoint)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("B3 refinement parent must be a checkpoint dictionary")

    checkpoint_sha = sha256_file(path)
    if checkpoint_sha != expected_sha256:
        raise RuntimeError(
            f"Checkpoint SHA mismatch: {checkpoint_sha} != {expected_sha256}"
        )
    if payload.get("model_name") != B3_MODEL_NAME:
        raise ValueError(f"Unexpected B3 model_name: {payload.get('model_name')}")
    if int(payload.get("epoch", -1)) != EXPECTED_PARENT_EPOCH:
        raise ValueError(f"Expected parent total epoch 27, got {payload.get('epoch')}")
    if int(payload.get("b3_epoch", -1)) != EXPECTED_PARENT_B3_EPOCH:
        raise ValueError(f"Expected parent B3 epoch 3, got {payload.get('b3_epoch')}")
    if int(payload.get("global_step", -1)) != EXPECTED_PARENT_GLOBAL_STEP:
        raise ValueError(
            f"Expected parent global step 27200, got {payload.get('global_step')}"
        )
    if payload.get("input_channels") != 4 or payload.get("output_channels") != 1:
        raise ValueError("B3 refinement parent must use the unchanged 4-to-1 UNet")
    if payload.get("semantic_embedding_dim") != 32:
        raise ValueError("B3 refinement parent semantic embedding must be 32-d")
    if payload.get("semantic_mid_channels") != 512:
        raise ValueError("B3 refinement parent projection must output 512 channels")
    if payload.get("semantic_injection") != "pre_mid_block_addition":
        raise ValueError("B3 refinement parent injection location changed")
    if payload.get("tau_rule") != {
        "Background": 0,
        "LM": 0,
        "FC": "t",
        "LC": "t",
        "VV": "t",
    }:
        raise ValueError(f"B1 tau policy was not retained: {payload.get('tau_rule')}")

    config = payload.get("config") or {}
    if config.get("input_channels") != EXPECTED_INPUTS:
        raise ValueError(f"Unexpected input contract: {config.get('input_channels')}")

    unet = build_smis4ch_unet(768)
    unet_result = unet.load_state_dict(payload["model_state_dict"], strict=True)
    model = SemanticConditionedUNet(unet, embedding_dim=32, mid_channels=512)
    semantic_result = model.semantic_encoder.load_state_dict(
        payload["semantic_state_dict"], strict=True
    )
    if unet_result.missing_keys or unet_result.unexpected_keys:
        raise RuntimeError("UNet strict reload reported incompatible keys")
    if semantic_result.missing_keys or semantic_result.unexpected_keys:
        raise RuntimeError("Semantic strict reload reported incompatible keys")

    checkpoint_semantic = payload["semantic_state_dict"]
    loaded_semantic = model.semantic_encoder.state_dict()
    for key, value in checkpoint_semantic.items():
        if not torch.equal(value.cpu(), loaded_semantic[key].cpu()):
            raise RuntimeError(f"Semantic state changed during strict reload: {key}")
    norms = semantic_norms(model)
    if norms["projection_weight_norm"] <= 0.0:
        raise RuntimeError("Trained B3 projection was incorrectly zero or reinitialized")

    meta: dict[str, Any] = {
        "checkpoint": str(path.resolve()),
        "sha256": checkpoint_sha,
        "checkpoint_keys": sorted(payload.keys()),
        "loaded_unet_layers": len(payload["model_state_dict"]),
        "loaded_semantic_layers": len(payload["semantic_state_dict"]),
        "unet_missing_keys": list(unet_result.missing_keys),
        "unet_unexpected_keys": list(unet_result.unexpected_keys),
        "semantic_missing_keys": list(semantic_result.missing_keys),
        "semantic_unexpected_keys": list(semantic_result.unexpected_keys),
        "conv_in_shape": list(model.unet.conv_in.weight.shape),
        "input_channels": EXPECTED_INPUTS,
        "output_channels": 1,
        "parent_total_epoch": int(payload["epoch"]),
        "parent_b3_epoch": int(payload["b3_epoch"]),
        "parent_global_step": int(payload["global_step"]),
        "parent_best_loss": float(payload.get("best_loss", float("nan"))),
        "semantic_embedding_dim": 32,
        "semantic_projection_channels": 512,
        "semantic_injection": "pre_mid_block_addition",
        "semantic_state_loaded": True,
        "semantic_state_reinitialized_after_load": False,
        "parent_optimizer_state_loaded": False,
        "parent_scheduler_state_loaded": False,
        "semantic_norms_after_load": norms,
    }
    return model, payload, meta


__all__ = [
    "load_b3_refinement_parent",
    "save_semantic_checkpoint",
    "semantic_norms",
]
