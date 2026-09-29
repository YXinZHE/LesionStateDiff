from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .class_semantic_encoder import SpatialClassSemanticEncoder
from .region_time import (
    build_tau_map,
    hard_masked_source,
    lesion_mask_from_segmentation,
    q_sample_region,
    region_time_ddim_step,
)


class SemanticConditionedUNet(nn.Module):
    """Inject a zero-initialized semantic feature before the UNet mid block."""

    def __init__(
        self,
        unet: nn.Module,
        *,
        num_classes: int = 5,
        embedding_dim: int = 32,
        mid_channels: int = 512,
    ) -> None:
        super().__init__()
        if not hasattr(unet, "mid_block"):
            raise ValueError("UNet must expose mid_block for B3 semantic injection")
        self.unet = unet
        self.semantic_encoder = SpatialClassSemanticEncoder(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            output_channels=mid_channels,
        )
        self._semantic_segmentation: torch.Tensor | None = None
        self._last_semantic_feature: torch.Tensor | None = None
        self._hook = self.unet.mid_block.register_forward_pre_hook(self._inject_semantic)

    def _inject_semantic(self, _module: nn.Module, inputs: tuple[Any, ...]):
        if self._semantic_segmentation is None:
            raise RuntimeError("semantic_seg must be provided for every B3 forward call")
        if not inputs:
            raise RuntimeError("UNet mid_block received no hidden-state input")
        hidden = inputs[0]
        semantic = self.semantic_encoder(
            self._semantic_segmentation,
            spatial_size=tuple(hidden.shape[-2:]),
            output_dtype=hidden.dtype,
        )
        if semantic.shape != hidden.shape:
            raise RuntimeError(
                f"semantic/UNet mid feature mismatch: {tuple(semantic.shape)} vs {tuple(hidden.shape)}"
            )
        self._last_semantic_feature = semantic
        return (hidden + semantic, *inputs[1:])

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        *,
        semantic_seg: torch.Tensor,
        **kwargs: Any,
    ):
        self._semantic_segmentation = semantic_seg
        try:
            return self.unet(sample, timestep, **kwargs)
        finally:
            self._semantic_segmentation = None

    def semantic_activation_map(self, spatial_size: tuple[int, int]) -> torch.Tensor:
        if self._last_semantic_feature is None:
            raise RuntimeError("No semantic activation is available before a forward call")
        activation = self._last_semantic_feature.detach().float().abs().mean(dim=1, keepdim=True)
        return F.interpolate(activation, size=spatial_size, mode="bilinear", align_corners=False)


@torch.no_grad()
def sample_region_time_ddim_b3(
    model: SemanticConditionedUNet,
    scheduler,
    source_image: torch.Tensor,
    segmentation: torch.Tensor,
    generation_noise: torch.Tensor,
    num_inference_steps: int,
) -> dict[str, torch.Tensor]:
    """B1 hard region-time DDIM with mid-block semantic conditioning only."""
    lesion = lesion_mask_from_segmentation(segmentation).to(source_image.dtype)
    masked = hard_masked_source(source_image, lesion)
    scheduler.set_timesteps(num_inference_steps, device=source_image.device)
    timesteps = [int(value.item()) for value in scheduler.timesteps]
    first_t = torch.full(
        (source_image.shape[0],), timesteps[0], device=source_image.device, dtype=torch.long
    )
    tau_indices, initial_tau_condition = build_tau_map(
        segmentation, first_t, scheduler.config.num_train_timesteps
    )
    sample, _ = q_sample_region(
        source_image,
        tau_indices,
        scheduler.alphas_cumprod,
        noise=generation_noise,
        lesion_mask=lesion,
    )
    initial = sample.clone()
    for index, timestep in enumerate(timesteps):
        previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        batch_t = torch.full(
            (source_image.shape[0],), timestep, device=source_image.device, dtype=torch.long
        )
        _, tau_condition = build_tau_map(
            segmentation, batch_t, scheduler.config.num_train_timesteps
        )
        model_input = torch.cat([sample, masked, segmentation, tau_condition], dim=1)
        prediction = model(model_input, batch_t, semantic_seg=segmentation).sample
        sample = region_time_ddim_step(
            sample,
            prediction,
            lesion,
            scheduler.alphas_cumprod,
            timestep,
            previous,
            clip_sample=bool(getattr(scheduler.config, "clip_sample", True)),
        )
    return {
        "generated": sample,
        "initial_state": initial,
        "lesion_mask": lesion,
        "masked_source": masked,
        "initial_tau_condition": initial_tau_condition,
        "semantic_activation": model.semantic_activation_map(tuple(source_image.shape[-2:])),
    }
