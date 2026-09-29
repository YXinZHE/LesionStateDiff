#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image
from torch.utils.data import DataLoader, Subset

from lesionstatediff.io_utils import crop_768_to_750, require_no_test_path
from lesionstatediff.region_time import (
    build_tau_map,
    hard_masked_source,
    lesion_mask_from_segmentation,
    q_sample_region,
    sample_region_time_ddim,
)
from lesionstatediff.region_time_checkpoint import (
    load_region_time_parent,
    save_region_time_checkpoint,
)
from lesionstatediff.region_time_dataset import RegionTimeDataset
from lesionstatediff.unified_four_class_dataset import (
    POOL_ORDER,
    VIRTUAL_EPOCH_SAMPLE_COUNTS,
    UnifiedVirtualEpochSampler,
    load_unified_records_json,
)


def collate(batch):
    return {
        "target_image": torch.stack([row["target_image"] for row in batch]),
        "segmentation_map": torch.stack([row["segmentation_map"] for row in batch]),
        "image_id": [row["image_id"] for row in batch],
        "patient_id": [row["patient_id"] for row in batch],
        "pool_name": [row["pool_name"] for row in batch],
        "target_role": [row["target_role"] for row in batch],
    }


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    value = crop_768_to_750(image).detach().cpu().float().squeeze().numpy()
    return np.clip((value + 1.0) * 127.5, 0, 255).astype(np.uint8)


def mask_to_uint8(mask: torch.Tensor) -> np.ndarray:
    return crop_768_to_750(mask).detach().cpu().squeeze().numpy().astype(np.uint8)


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def fixed_indices(dataset: RegionTimeDataset, count: int) -> list[int]:
    selected: list[int] = []
    seen: set[str] = set()
    for index, record in enumerate(dataset.records):
        if record.image_id in seen:
            continue
        if record.fc_pixels + record.lc_pixels + record.vv_pixels <= 0:
            continue
        selected.append(index)
        seen.add(record.image_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} unique lesion-positive fixed cases found")
    return selected


@torch.no_grad()
def generate_fixed100(
    model,
    dataset: RegionTimeDataset,
    indices: list[int],
    output_dir: Path,
    epoch: int,
    device: torch.device,
    eval_batch_size: int,
    ddim_steps: int,
    seed: int,
) -> dict[str, object]:
    model.eval()
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    epoch_dir = output_dir / "fixed_eval" / f"epoch_{epoch:03d}"
    outside_maxima: list[float] = []
    inside_means: list[float] = []
    generated_count = 0
    offset = 0
    for batch in loader:
        image = batch["target_image"].to(device)
        segmentation = batch["segmentation_map"].to(device)
        noises = []
        for local_index in range(image.shape[0]):
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + offset + local_index)
            noises.append(
                torch.randn(
                    image[local_index].shape,
                    device=device,
                    dtype=image.dtype,
                    generator=generator,
                )
            )
        generation_noise = torch.stack(noises)
        result = sample_region_time_ddim(
            model,
            scheduler,
            image,
            segmentation,
            generation_noise,
            ddim_steps,
        )
        generated = result["generated"]
        lesion = result["lesion_mask"]
        difference = (generated - image).abs()
        inactive = lesion < 0.5
        outside_maxima.append(float(difference[inactive].max().detach().cpu()))
        inside_means.append(float(difference[lesion > 0.5].mean().detach().cpu()))
        tau_view = lesion * 255.0
        for local_index, image_id in enumerate(batch["image_id"]):
            sample_dir = epoch_dir / image_id
            save_png(sample_dir / "original.png", tensor_to_uint8(image[local_index]))
            save_png(sample_dir / "segmentation.png", mask_to_uint8(segmentation[local_index]))
            save_png(sample_dir / "tau.png", mask_to_uint8(tau_view[local_index]))
            save_png(sample_dir / "generated.png", tensor_to_uint8(generated[local_index]))
            diff_u8 = np.clip(
                crop_768_to_750(difference[local_index]).detach().cpu().float().squeeze().numpy()
                * 127.5,
                0,
                255,
            ).astype(np.uint8)
            save_png(sample_dir / "difference.png", diff_u8)
            generated_count += 1
        offset += image.shape[0]
    summary = {
        "epoch": epoch,
        "generated_count": generated_count,
        "ddim_steps": ddim_steps,
        "fixed_seed_base": seed,
        "outside_region_max_abs_diff": max(outside_maxima),
        "inside_region_mean_abs_diff": float(np.mean(inside_means)),
        "background_stable": max(outside_maxima) == 0.0,
        "source_reinjection_used": False,
        "final_blend_used": False,
        "fold1_test_used": False,
    }
    (epoch_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model.train()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fixed-count", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--save-epochs", nargs="+", type=int, default=[5, 10, 15, 20])
    args = parser.parse_args()

    require_no_test_path(args.train_manifest, "train_manifest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal training")
    if 4000 % args.batch_size != 0:
        raise ValueError("batch_size must divide 4000 to preserve fixed exposures per epoch")
    steps_per_epoch = 4000 // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    model, _, _, parent_meta = load_region_time_parent(args.parent_checkpoint)
    if parent_meta["sha256"] != args.expected_parent_sha256:
        raise RuntimeError(
            f"Parent SHA mismatch: {parent_meta['sha256']} != {args.expected_parent_sha256}"
        )
    model.to(device).train()
    scheduler = DDIMScheduler(num_train_timesteps=args.num_train_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, total_steps // 10),
        num_training_steps=total_steps,
    )

    output = Path(args.output_dir)
    for name in ("checkpoints", "logs", "configs", "fixed_eval", "reports"):
        (output / name).mkdir(parents=True, exist_ok=True)
    records = load_unified_records_json(args.train_manifest)
    dataset = RegionTimeDataset(args.train_manifest)
    sampler = UnifiedVirtualEpochSampler(
        records,
        steps_per_virtual_epoch=steps_per_epoch,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    fixed = fixed_indices(dataset, args.fixed_count)
    config = vars(args) | {
        "model_name": "OCT-Hard-Region-Time-Conditioned-Diffusion-B",
        "steps_per_virtual_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "samples_per_virtual_epoch": 4000,
        "input_channels": ["x_tau", "hard_masked_oct", "segmentation", "tau_normalized"],
        "output_channels": ["epsilon_prediction"],
        "region_time_trainable_parameters": 0,
        "region_time_lr": None,
        "region_time_lr_reason": "Rule-based tau map has no trainable parameters",
        "loss": "global_epsilon_mse_against_effective_noise",
        "soft_mask_used": False,
        "source_reinjection_used": False,
        "final_blend_used": False,
        "fold1_test_used": False,
        "parent": parent_meta,
    }
    (output / "configs" / "resolved_region_time_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log_path = output / "logs" / "train.jsonl"
    best_loss = float("inf")
    global_step = 0
    milestone_summaries: list[dict[str, object]] = []
    save_epochs = set(args.save_epochs)
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate,
        )
        epoch_loss = 0.0
        batch_count = 0
        pool_counts: Counter[str] = Counter()
        role_counts: Counter[str] = Counter()
        start = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        for batch in loader:
            image = batch["target_image"].to(device, non_blocking=True)
            segmentation = batch["segmentation_map"].to(device, non_blocking=True)
            lesion = lesion_mask_from_segmentation(segmentation).to(image.dtype)
            timestep = torch.randint(
                0, args.num_train_timesteps, (image.shape[0],), device=device, dtype=torch.long
            )
            tau, tau_condition = build_tau_map(
                segmentation, timestep, args.num_train_timesteps
            )
            noise = torch.randn_like(image)
            x_tau, effective_noise = q_sample_region(
                image, tau, scheduler.alphas_cumprod, noise=noise, lesion_mask=lesion
            )
            masked = hard_masked_source(image, lesion)
            model_input = torch.cat([x_tau, masked, segmentation, tau_condition], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(model_input, timestep).sample
                loss = F.mse_loss(prediction.float(), effective_noise.float())
            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN/Inf loss at epoch={epoch}, step={global_step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            global_step += 1
            epoch_loss += float(loss.detach().cpu())
            batch_count += 1
            pool_counts.update(batch["pool_name"])
            role_counts.update(batch["target_role"])
        mean_loss = epoch_loss / max(1, batch_count)
        best_loss = min(best_loss, mean_loss)
        expected_ok = all(
            pool_counts.get(pool, 0) == VIRTUAL_EPOCH_SAMPLE_COUNTS[pool] for pool in POOL_ORDER
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_steps": batch_count,
            "loss": mean_loss,
            "best_loss": best_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "pool_exposure": dict(pool_counts),
            "role_exposure": dict(role_counts),
            "exposure_counts_ok": expected_ok,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "mean_step_seconds": (time.time() - start) / max(1, batch_count),
            "nan_inf": False,
            "fold1_test_used": False,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not expected_ok:
            raise RuntimeError(f"Exposure mismatch at epoch {epoch}: {dict(pool_counts)}")
        if epoch in save_epochs:
            checkpoint_path = output / "checkpoints" / f"epoch_{epoch}.pt"
            save_region_time_checkpoint(
                checkpoint_path,
                model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                config=config,
                parent_meta=parent_meta,
            )
            fixed_summary = generate_fixed100(
                model,
                dataset,
                fixed,
                output,
                epoch,
                device,
                args.eval_batch_size,
                args.ddim_steps,
                args.seed + 900000,
            )
            milestone_summaries.append(fixed_summary)

    final_path = output / "checkpoints" / "final_region_time.pt"
    save_region_time_checkpoint(
        final_path,
        model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        epoch=args.epochs,
        global_step=global_step,
        best_loss=best_loss,
        config=config,
        parent_meta=parent_meta,
    )
    summary = {
        "completed": True,
        "epochs": args.epochs,
        "global_step": global_step,
        "best_loss": best_loss,
        "final_checkpoint": str(final_path),
        "milestone_checkpoints": [
            str(output / "checkpoints" / f"epoch_{epoch}.pt") for epoch in sorted(save_epochs)
        ],
        "fixed_validation": milestone_summaries,
        "fold1_test_used": False,
        "soft_mask_used": False,
        "source_reinjection_used": False,
        "final_blend_used": False,
    }
    (output / "reports" / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

