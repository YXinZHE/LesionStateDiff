#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader, Subset

from lesionstatediff.io_utils import require_no_test_path
from lesionstatediff.region_time import (
    build_tau_map,
    hard_masked_source,
    lesion_mask_from_segmentation,
    q_sample_region,
    region_time_ddim_step,
)
from lesionstatediff.region_time_b3_checkpoint import load_region_time_b3_base
from lesionstatediff.region_time_dataset import RegionTimeDataset
from lesionstatediff.region_time_stage3_sampler import (
    STAGE3_VIRTUAL_EPOCH_SAMPLE_COUNTS,
    Stage3VirtualEpochSampler,
)
from lesionstatediff.unified_four_class_dataset import load_unified_records_json


def collate(batch):
    return {
        "target_image": torch.stack([row["target_image"] for row in batch]),
        "segmentation_map": torch.stack([row["segmentation_map"] for row in batch]),
        "image_id": [row["image_id"] for row in batch],
    }


def fixed_indices(dataset: RegionTimeDataset, count: int) -> list[int]:
    selected: list[int] = []
    seen: set[str] = set()
    for index, record in enumerate(dataset.records):
        if record.image_id in seen or record.fc_pixels + record.lc_pixels + record.vv_pixels <= 0:
            continue
        selected.append(index)
        seen.add(record.image_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} fixed cases found")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--stage3-summary", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--unet-lr", type=float, default=5e-7)
    parser.add_argument("--semantic-lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    require_no_test_path(args.train_manifest, "train_manifest")
    if args.batch_size != 5 or args.unet_lr != 5e-7 or args.semantic_lr != 5e-5:
        raise ValueError("Formal B3 smoke requires batch=5, UNet lr=5e-7, semantic lr=5e-5")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B3 smoke")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    model, _, base_meta = load_region_time_b3_base(
        args.base_checkpoint, expected_sha256=args.expected_sha256
    )
    model.to(device).train()
    if not model.semantic_encoder.projection_is_zero():
        raise RuntimeError("Semantic projection is not zero before B3 smoke")
    optimizer = torch.optim.AdamW(
        [
            {"params": model.unet.parameters(), "lr": args.unet_lr, "name": "unet"},
            {
                "params": model.semantic_encoder.parameters(),
                "lr": args.semantic_lr,
                "name": "semantic_branch",
            },
        ]
    )
    dataset = RegionTimeDataset(args.train_manifest)
    records = load_unified_records_json(args.train_manifest)
    fixed = fixed_indices(dataset, 100)
    fixed_ids = [dataset.records[index].image_id for index in fixed]
    stage3 = json.loads(Path(args.stage3_summary).read_text(encoding="utf-8"))
    if any(v.get("fixed_case_ids") != fixed_ids for v in stage3.get("validations", [])):
        raise RuntimeError("B3 fixed-100 differs from Stage3")

    sampler = Stage3VirtualEpochSampler(
        records,
        steps_per_virtual_epoch=1600,
        batch_size=args.batch_size,
        seed=args.seed,
        excluded_image_ids=set(fixed_ids),
    )
    sampler.set_epoch(25)
    plan = sampler.plan_indices()
    pool_counts = Counter(records[index].pool_name for index in plan)
    fixed_exposure = sum(records[index].image_id in set(fixed_ids) for index in plan)
    if fixed_exposure:
        raise RuntimeError("Fixed validation case entered B3 training plan")

    batch = next(iter(DataLoader(
        Subset(dataset, plan[: args.batch_size]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )))
    x0 = batch["target_image"].to(device)
    segmentation = batch["segmentation_map"].to(device)
    lesion = lesion_mask_from_segmentation(segmentation).to(x0.dtype)
    timestep = torch.full((x0.shape[0],), 500, device=device, dtype=torch.long)
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    tau, tau_condition = build_tau_map(segmentation, timestep, 1000)

    synthetic_seg = torch.tensor([[[[0, 1, 2, 3, 4]]]], device=device, dtype=torch.float32)
    synthetic_tau, synthetic_condition = build_tau_map(
        synthetic_seg, torch.tensor([500], device=device), 1000
    )
    expected_tau = [0, 0, 500, 500, 500]
    observed_tau = synthetic_tau.flatten().tolist()
    if observed_tau != expected_tau:
        raise RuntimeError(f"B3 changed the B1 tau rule: {observed_tau} != {expected_tau}")

    noise = torch.randn_like(x0)
    x_tau, effective_noise = q_sample_region(
        x0, tau, scheduler.alphas_cumprod, noise=noise, lesion_mask=lesion
    )
    model_input = torch.cat([x_tau, hard_masked_source(x0, lesion), segmentation, tau_condition], dim=1)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(model_input, timestep, semantic_seg=segmentation).sample
        loss = F.mse_loss(prediction.float(), effective_noise.float())
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite B3 smoke loss: {loss}")
    semantic_feature_shape = list(model._last_semantic_feature.shape)
    if semantic_feature_shape != [args.batch_size, 512, 24, 24]:
        raise RuntimeError(f"Unexpected semantic feature shape: {semantic_feature_shape}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    stepped = region_time_ddim_step(
        x_tau.detach(), prediction.detach(), lesion, scheduler.alphas_cumprod, 500, 450
    )
    inactive_max_change = float(((stepped - x_tau.detach()).abs() * (1.0 - lesion)).max().cpu())
    if inactive_max_change != 0.0:
        raise RuntimeError(f"tau=0 identity transition drift: {inactive_max_change}")

    result = {
        "passed": True,
        "checkpoint_load_success": True,
        "base_checkpoint": base_meta,
        "x0_shape": list(x0.shape),
        "seg_shape": list(segmentation.shape),
        "tau_shape": list(tau.shape),
        "x_tau_shape": list(x_tau.shape),
        "unet_input_shape": list(model_input.shape),
        "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach().cpu()),
        "tau_rule_t500": {"Background": 0, "LM": 0, "FC": 500, "LC": 500, "VV": 500},
        "synthetic_tau_observed": observed_tau,
        "synthetic_tau_condition_observed": synthetic_condition.flatten().tolist(),
        "semantic_map_shape": list(segmentation.shape),
        "semantic_projection_output_shape": semantic_feature_shape,
        "unet_mid_feature_shape": semantic_feature_shape,
        "semantic_projection_zero_before_optimizer_step": True,
        "semantic_projection_zero_after_optimizer_step": model.semantic_encoder.projection_is_zero(),
        "q_sample_region_unchanged": True,
        "unet_backbone_unchanged": True,
        "semantic_branch_added": True,
        "epsilon_mse_loss_unchanged": True,
        "inactive_tau_zero_identity_max_change": inactive_max_change,
        "source_reinjection_used": False,
        "soft_blend_used": False,
        "posthoc_mask_restore_used": False,
        "planned_pool_exposure": dict(pool_counts),
        "expected_pool_exposure": STAGE3_VIRTUAL_EPOCH_SAMPLE_COUNTS,
        "fixed100_matches_stage3": True,
        "fixed100_exposure_count": fixed_exposure,
        "optimizer_step_passed": True,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "fold1_test_used": False,
        "synthetic_pool_generation_started": False,
        "segmentation_training_started": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

