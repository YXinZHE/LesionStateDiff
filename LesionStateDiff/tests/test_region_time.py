from __future__ import annotations

import torch

from lesionstatediff.region_time import (
    build_tau_map,
    hard_masked_source,
    lesion_mask_from_segmentation,
    q_sample_region,
    region_time_ddim_step,
)


def sample_segmentation() -> torch.Tensor:
    seg = torch.zeros(2, 1, 8, 8)
    seg[:, :, :2, :] = 1
    seg[0, :, 2:4, 2:4] = 2
    seg[0, :, 4:6, 2:4] = 3
    seg[1, :, 3:5, 4:6] = 4
    return seg


def test_tau_map_is_zero_for_background_and_lm() -> None:
    seg = sample_segmentation()
    t = torch.tensor([17, 29])
    tau, condition = build_tau_map(seg, t, 1000)
    assert tau.shape == seg.shape
    assert torch.all(tau[seg <= 1] == 0)
    assert torch.all(tau[0][seg[0] == 2] == 17)
    assert torch.all(tau[1][seg[1] == 4] == 29)
    assert condition.min() >= 0
    assert condition.max() <= 1


def test_region_forward_keeps_inactive_pixels_exactly() -> None:
    seg = sample_segmentation()
    lesion = lesion_mask_from_segmentation(seg)
    x0 = torch.randn_like(seg)
    noise = torch.randn_like(seg)
    tau, _ = build_tau_map(seg, torch.tensor([500, 700]), 1000)
    alphas = torch.linspace(0.999, 0.001, 1000)
    x_tau, effective_noise = q_sample_region(x0, tau, alphas, noise, lesion)
    inactive = lesion == 0
    assert torch.equal(x_tau[inactive], x0[inactive])
    assert torch.count_nonzero(effective_noise[inactive]) == 0
    assert not torch.equal(x_tau[lesion > 0], x0[lesion > 0])


def test_hard_masked_source_only_removes_lesions() -> None:
    seg = sample_segmentation()
    lesion = lesion_mask_from_segmentation(seg)
    x0 = torch.randn_like(seg)
    masked = hard_masked_source(x0, lesion)
    assert torch.count_nonzero(masked[lesion > 0]) == 0
    assert torch.equal(masked[lesion == 0], x0[lesion == 0])


def test_region_ddim_step_has_identity_transition_outside_lesions() -> None:
    seg = sample_segmentation()
    lesion = lesion_mask_from_segmentation(seg)
    sample = torch.randn_like(seg)
    prediction = torch.randn_like(seg)
    alphas = torch.linspace(0.999, 0.001, 1000)
    previous = region_time_ddim_step(
        sample, prediction, lesion, alphas, current_timestep=900, previous_timestep=800
    )
    assert torch.equal(previous[lesion == 0], sample[lesion == 0])
    assert not torch.equal(previous[lesion > 0], sample[lesion > 0])

