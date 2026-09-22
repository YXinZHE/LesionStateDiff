from __future__ import annotations

from typing import Iterable

import torch


LESION_CLASS_VALUES = (2, 3, 4)


def lesion_mask_from_segmentation(segmentation: torch.Tensor) -> torch.Tensor:
    """Return the hard FC/LC/VV edit mask as a float tensor."""
    if segmentation.ndim != 4 or segmentation.shape[1] != 1:
        raise ValueError(
            "segmentation must have shape [B,1,H,W], got "
            f"{tuple(segmentation.shape)}"
        )
    mask = torch.zeros_like(segmentation, dtype=torch.bool)
    for value in LESION_CLASS_VALUES:
        mask |= segmentation == float(value)
    return mask.to(dtype=torch.float32)


def build_tau_map(
    segmentation: torch.Tensor,
    timesteps: torch.Tensor,
    num_train_timesteps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build integer tau indices and a normalized tau conditioning channel.

    Background and LM receive tau=0. FC, LC, and VV receive the sample's
    scalar timestep. The normalized channel is used only as UNet input.
    """
    if timesteps.ndim != 1 or timesteps.shape[0] != segmentation.shape[0]:
        raise ValueError(
            "timesteps must have shape [B] matching segmentation batch, got "
            f"{tuple(timesteps.shape)}"
        )
    if num_train_timesteps <= 1:
        raise ValueError("num_train_timesteps must be greater than one")
    lesion = lesion_mask_from_segmentation(segmentation)
    t = timesteps.to(device=segmentation.device, dtype=torch.long).view(-1, 1, 1, 1)
    tau_indices = (lesion.to(torch.long) * t).long()
    tau_condition = tau_indices.to(segmentation.dtype) / float(num_train_timesteps - 1)
    return tau_indices, tau_condition


def hard_masked_source(image: torch.Tensor, lesion_mask: torch.Tensor) -> torch.Tensor:
    if image.shape != lesion_mask.shape:
        raise ValueError(f"image/mask shape mismatch: {image.shape} vs {lesion_mask.shape}")
    return image * (1.0 - lesion_mask.to(dtype=image.dtype))


def _alpha_from_tau(
    alphas_cumprod: torch.Tensor,
    tau_indices: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    flat = alphas_cumprod.to(device=tau_indices.device, dtype=dtype)
    if tau_indices.min().item() < 0 or tau_indices.max().item() >= flat.numel():
        raise ValueError("tau index is outside the scheduler alpha table")
    return flat[tau_indices.long()]


def q_sample_region(
    x0: torch.Tensor,
    tau_indices: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    noise: torch.Tensor | None = None,
    lesion_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply forward diffusion only to FC/LC/VV pixels.

    tau=0 is treated as an identity state for inactive pixels. This explicit
    identity is necessary because scheduler alpha_bar[0] is close to, but not
    exactly, one.
    """
    if x0.shape != tau_indices.shape:
        raise ValueError(f"x0/tau shape mismatch: {x0.shape} vs {tau_indices.shape}")
    if noise is None:
        noise = torch.randn_like(x0)
    if noise.shape != x0.shape:
        raise ValueError(f"noise shape mismatch: {noise.shape} vs {x0.shape}")
    active = (tau_indices > 0).to(dtype=x0.dtype) if lesion_mask is None else lesion_mask.to(dtype=x0.dtype)
    alpha = _alpha_from_tau(alphas_cumprod, tau_indices, x0.dtype)
    noised = alpha.sqrt() * x0 + (1.0 - alpha).clamp_min(0.0).sqrt() * noise
    x_tau = torch.where(active > 0.5, noised, x0)
    effective_noise = noise * active
    return x_tau, effective_noise


def region_time_ddim_step(
    sample: torch.Tensor,
    noise_prediction: torch.Tensor,
    lesion_mask: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    current_timestep: int,
    previous_timestep: int,
    clip_sample: bool = True,
) -> torch.Tensor:
    """One deterministic DDIM update with tau=0 identity outside lesions.

    This is a scheduler equation, not source-image reinjection: inactive pixels
    are the tau=0 state and therefore use the identity transition at every step.
    """
    if sample.shape != noise_prediction.shape or sample.shape != lesion_mask.shape:
        raise ValueError("sample, prediction, and lesion_mask must have identical shapes")
    alpha_table = alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
    alpha_t = alpha_table[int(current_timestep)]
    if previous_timestep >= 0:
        alpha_prev = alpha_table[int(previous_timestep)]
    else:
        alpha_prev = sample.new_tensor(1.0)
    pred_x0 = (sample - (1.0 - alpha_t).clamp_min(0.0).sqrt() * noise_prediction) / alpha_t.sqrt()
    if clip_sample:
        pred_x0 = pred_x0.clamp(-1.0, 1.0)
    target_previous = alpha_prev.sqrt() * pred_x0 + (1.0 - alpha_prev).clamp_min(0.0).sqrt() * noise_prediction
    return torch.where(lesion_mask > 0.5, target_previous, sample)


@torch.no_grad()
def sample_region_time_ddim(
    model,
    scheduler,
    source_image: torch.Tensor,
    segmentation: torch.Tensor,
    generation_noise: torch.Tensor,
    num_inference_steps: int,
) -> dict[str, torch.Tensor]:
    """Generate with hard region-time DDIM and no source reinjection/blending."""
    lesion = lesion_mask_from_segmentation(segmentation).to(source_image.dtype)
    masked = hard_masked_source(source_image, lesion)
    scheduler.set_timesteps(num_inference_steps, device=source_image.device)
    timesteps: list[int] = [int(t.item()) for t in scheduler.timesteps]
    first_t = torch.full(
        (source_image.shape[0],), timesteps[0], device=source_image.device, dtype=torch.long
    )
    tau_indices, tau_condition = build_tau_map(
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
        prediction = model(model_input, batch_t).sample
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
        "tau_condition": tau_condition,
    }
