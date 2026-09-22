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
from lesionstatediff.region_time_b3_refinement_checkpoint import (
    load_b3_refinement_parent,
    semantic_norms,
)
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
    parser.add_argument("--parent-b3-summary", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--unet-lr", type=float, default=1e-7)
    parser.add_argument("--semantic-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    require_no_test_path(args.train_manifest, "train_manifest")
    if args.batch_size != 5 or args.unet_lr != 1e-7 or args.semantic_lr != 1e-5:
        raise ValueError(
            "Formal B3 refinement smoke requires batch=5, UNet lr=1e-7, semantic lr=1e-5"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B3 refinement smoke")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    model, _, parent_meta = load_b3_refinement_parent(
        args.base_checkpoint, expected_sha256=args.expected_sha256
    )
    parent_norms = semantic_norms(model)
    model.to(device).train()
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
    if optimizer.param_groups[0]["lr"] != 1e-7 or optimizer.param_groups[1]["lr"] != 1e-5:
        raise RuntimeError("Optimizer parameter-group learning rates are incorrect")

    dataset = RegionTimeDataset(args.train_manifest)
    records = load_unified_records_json(args.train_manifest)
    fixed = fixed_indices(dataset, 100)
    fixed_ids = [dataset.records[index].image_id for index in fixed]
    parent_summary = json.loads(Path(args.parent_b3_summary).read_text(encoding="utf-8"))
    parent_validations = parent_summary.get("validations", [])
    if not parent_validations or any(v.get("fixed_case_ids") != fixed_ids for v in parent_validations):
        raise RuntimeError("Refinement fixed-100 differs from the parent B3 run")
    parent_e3 = next((v for v in parent_validations if int(v.get("b3_epoch", -1)) == 3), None)
    if parent_e3 is None:
        raise RuntimeError("Parent B3 epoch3 Fixed100 record is missing")
    if int(parent_e3.get("fixed_seed_base", -1)) != 900003 or int(parent_e3.get("ddim_steps", -1)) != 25:
        raise RuntimeError("Parent B3 epoch3 does not use fixed seed 900003 and DDIM 25")

    sampler = Stage3VirtualEpochSampler(
        records,
        steps_per_virtual_epoch=1600,
        batch_size=args.batch_size,
        seed=args.seed,
        excluded_image_ids=set(fixed_ids),
    )
    sampler.set_epoch(28)
    plan = sampler.plan_indices()
    pool_counts = Counter(records[index].pool_name for index in plan)
    fixed_exposure = sum(records[index].image_id in set(fixed_ids) for index in plan)
    if dict(pool_counts) != STAGE3_VIRTUAL_EPOCH_SAMPLE_COUNTS:
        raise RuntimeError(f"Sampling plan changed: {dict(pool_counts)}")
    if fixed_exposure:
        raise RuntimeError("Fixed validation case entered refinement training plan")

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
    synthetic_tau, _ = build_tau_map(synthetic_seg, torch.tensor([500], device=device), 1000)
    observed_tau = synthetic_tau.flatten().tolist()
    if observed_tau != [0, 0, 500, 500, 500]:
        raise RuntimeError(f"B1 tau policy changed: {observed_tau}")

    noise = torch.randn_like(x0)
    x_tau, effective_noise = q_sample_region(
        x0, tau, scheduler.alphas_cumprod, noise=noise, lesion_mask=lesion
    )
    masked = hard_masked_source(x0, lesion)
    model_input = torch.cat([x_tau, masked, segmentation, tau_condition], dim=1)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(model_input, timestep, semantic_seg=segmentation).sample
        loss = F.mse_loss(prediction.float(), effective_noise.float())
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite refinement smoke loss: {loss}")
    semantic_feature_shape = list(model._last_semantic_feature.shape)
    semantic_activation = float(model._last_semantic_feature.detach().float().abs().mean().cpu())
    if semantic_feature_shape != [args.batch_size, 512, 24, 24]:
        raise RuntimeError(f"Unexpected semantic feature shape: {semantic_feature_shape}")
    if semantic_activation <= 0.0:
        raise RuntimeError("Loaded semantic branch has zero activation")

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
        "parent_checkpoint": parent_meta,
        "parent_model_state_loaded": True,
        "parent_semantic_state_loaded": True,
        "parent_semantic_state_reinitialized": False,
        "parent_optimizer_state_loaded": False,
        "parent_scheduler_state_loaded": False,
        "tau_policy": {"BG": 0, "LM": 0, "FC": "1.0*t", "LC": "1.0*t", "VV": "1.0*t"},
        "x0_shape": list(x0.shape),
        "segmentation_shape": list(segmentation.shape),
        "tau_shape": list(tau.shape),
        "x_tau_shape": list(x_tau.shape),
        "hard_masked_oct_shape": list(masked.shape),
        "tau_normalized_shape": list(tau_condition.shape),
        "unet_input_shape": list(model_input.shape),
        "semantic_embedding_map_shape": [args.batch_size, 32, 768, 768],
        "semantic_projection_shape": semantic_feature_shape,
        "mid_feature_shape": semantic_feature_shape,
        "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach().cpu()),
        "unet_trainable_parameters": sum(p.numel() for p in model.unet.parameters() if p.requires_grad),
        "semantic_trainable_parameters": sum(
            p.numel() for p in model.semantic_encoder.parameters() if p.requires_grad
        ),
        "unet_lr": optimizer.param_groups[0]["lr"],
        "semantic_lr": optimizer.param_groups[1]["lr"],
        "semantic_norms_before_step": parent_norms,
        "semantic_activation_abs_mean": semantic_activation,
        "planned_pool_exposure": dict(pool_counts),
        "fixed100_matches_parent_b3": True,
        "fixed100_exposure_count": fixed_exposure,
        "inactive_tau_zero_identity_max_change": inactive_max_change,
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

