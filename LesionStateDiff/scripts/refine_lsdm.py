#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from lesionstatediff.io_utils import crop_768_to_750, require_no_test_path
from lesionstatediff.region_time import (
    build_tau_map,
    hard_masked_source,
    lesion_mask_from_segmentation,
    q_sample_region,
    sample_region_time_ddim,
)
from lesionstatediff.region_time_checkpoint import save_region_time_checkpoint
from lesionstatediff.region_time_dataset import RegionTimeDataset
from lesionstatediff.region_time_stage2_checkpoint import load_region_time_stage2_base
from lesionstatediff.unified_four_class_dataset import (
    POOL_ORDER,
    VIRTUAL_EPOCH_SAMPLE_COUNTS,
    UnifiedVirtualEpochSampler,
    load_unified_records_json,
)


CLASS_NAMES = {0: "Background", 1: "LM", 2: "FC", 3: "LC", 4: "VV"}


def collate(batch):
    return {
        "target_image": torch.stack([row["target_image"] for row in batch]),
        "segmentation_map": torch.stack([row["segmentation_map"] for row in batch]),
        "image_id": [row["image_id"] for row in batch],
        "patient_id": [row["patient_id"] for row in batch],
        "pool_name": [row["pool_name"] for row in batch],
        "target_role": [row["target_role"] for row in batch],
    }


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


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    value = crop_768_to_750(image).detach().cpu().float().squeeze().numpy()
    return np.clip((value + 1.0) * 127.5, 0, 255).astype(np.uint8)


def mask_to_uint8(mask: torch.Tensor) -> np.ndarray:
    return crop_768_to_750(mask).detach().cpu().squeeze().numpy().astype(np.uint8)


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def build_contact_sheet(epoch_dir: Path, image_ids: list[str], count: int = 20) -> Path:
    selected = image_ids[:count]
    cell_width, cell_height = 390, 150
    sheet = Image.new("RGB", (cell_width * 2, cell_height * 10), "white")
    draw = ImageDraw.Draw(sheet)
    for index, image_id in enumerate(selected):
        row, column = divmod(index, 2)
        x, y = column * cell_width, row * cell_height
        original = Image.open(epoch_dir / "original" / f"{image_id}.png").convert("L").resize((120, 120))
        generated = Image.open(epoch_dir / "generated" / f"{image_id}.png").convert("L").resize((120, 120))
        difference = Image.open(epoch_dir / "difference" / f"{image_id}.png").convert("L").resize((120, 120))
        sheet.paste(original.convert("RGB"), (x, y + 20))
        sheet.paste(generated.convert("RGB"), (x + 125, y + 20))
        sheet.paste(difference.convert("RGB"), (x + 250, y + 20))
        draw.text((x, y + 2), f"{image_id}: original | generated | difference", fill="black")
    path = epoch_dir / "contact_sheet.png"
    sheet.save(path)
    return path


@torch.no_grad()
def generate_validation(
    model,
    dataset: RegionTimeDataset,
    indices: list[int],
    output_dir: Path,
    total_epoch: int,
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
    epoch_dir = output_dir / "validation_stage2" / f"epoch{total_epoch}"
    for name in ("original", "generated", "tau", "difference", "segmentation"):
        (epoch_dir / name).mkdir(parents=True, exist_ok=True)
    diff_sums = {value: 0.0 for value in CLASS_NAMES}
    pixel_counts = {value: 0 for value in CLASS_NAMES}
    nonempty_counts = {value: 0 for value in CLASS_NAMES}
    generated_ids: list[str] = []
    black = white = near_constant = 0
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
        for value in CLASS_NAMES:
            region = segmentation == float(value)
            count = int(region.sum().item())
            if count:
                diff_sums[value] += float(difference[region].sum().detach().cpu())
                pixel_counts[value] += count
                nonempty_counts[value] += int(region.flatten(1).any(dim=1).sum().item())
        for local_index, image_id in enumerate(batch["image_id"]):
            original_u8 = tensor_to_uint8(image[local_index])
            generated_u8 = tensor_to_uint8(generated[local_index])
            segmentation_u8 = mask_to_uint8(segmentation[local_index])
            tau_u8 = mask_to_uint8(lesion[local_index] * 255.0)
            difference_u8 = np.clip(
                crop_768_to_750(difference[local_index]).detach().cpu().float().squeeze().numpy()
                * 127.5,
                0,
                255,
            ).astype(np.uint8)
            save_png(epoch_dir / "original" / f"{image_id}.png", original_u8)
            save_png(epoch_dir / "generated" / f"{image_id}.png", generated_u8)
            save_png(epoch_dir / "segmentation" / f"{image_id}.png", segmentation_u8)
            save_png(epoch_dir / "tau" / f"{image_id}.png", tau_u8)
            save_png(epoch_dir / "difference" / f"{image_id}.png", difference_u8)
            black += int(generated_u8.mean() < 2.0)
            white += int(generated_u8.mean() > 253.0)
            near_constant += int(generated_u8.std() < 2.0)
            generated_ids.append(image_id)
        offset += image.shape[0]
    mean_changes = {
        CLASS_NAMES[value]: diff_sums[value] / max(1, pixel_counts[value]) for value in CLASS_NAMES
    }
    if mean_changes["Background"] > 1e-7 or mean_changes["LM"] > 1e-7:
        raise RuntimeError(f"Protection drift detected at total epoch {total_epoch}: {mean_changes}")
    contact_sheet = build_contact_sheet(epoch_dir, generated_ids)
    summary = {
        "total_epoch": total_epoch,
        "generated_count": len(generated_ids),
        "fixed_case_ids": generated_ids,
        "fixed_seed_base": seed,
        "ddim_steps": ddim_steps,
        "mean_abs_change": mean_changes,
        "nonempty_image_counts": {
            CLASS_NAMES[value]: nonempty_counts[value] for value in CLASS_NAMES
        },
        "black_image_count": black,
        "white_image_count": white,
        "near_constant_image_count": near_constant,
        "background_protection_passed": mean_changes["Background"] <= 1e-7,
        "lm_protection_passed": mean_changes["LM"] <= 1e-7,
        "contact_sheet": str(contact_sheet),
        "fold1_test_used": False,
    }
    (epoch_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model.train()
    return summary


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--base-total-epoch", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--fixed-count", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--ddim-steps", type=int, default=25)
    args = parser.parse_args()

    require_no_test_path(args.train_manifest, "train_manifest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage2 training")
    if args.stage2_epochs != 10 or args.base_total_epoch != 15:
        raise ValueError("Formal Stage2 is fixed to total epochs 16-25")
    if 4000 % args.batch_size != 0:
        raise ValueError("batch_size must divide 4000 to preserve exposure counts")
    steps_per_epoch = 4000 // args.batch_size
    stage2_steps = steps_per_epoch * args.stage2_epochs
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    model, _, base_meta = load_region_time_stage2_base(
        args.base_checkpoint, expected_sha256=args.expected_sha256
    )
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=stage2_steps,
    )
    noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_timesteps)
    records = load_unified_records_json(args.train_manifest)
    dataset = RegionTimeDataset(args.train_manifest)
    sampler = UnifiedVirtualEpochSampler(
        records,
        steps_per_virtual_epoch=steps_per_epoch,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    fixed = fixed_indices(dataset, args.fixed_count)
    output = Path(args.output_dir)
    for name in ("checkpoints", "logs", "configs", "validation_stage2", "reports"):
        (output / name).mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "training_total_epoch_range": [16, 25],
        "steps_per_epoch": steps_per_epoch,
        "stage2_optimizer_steps": stage2_steps,
        "base_global_step": base_meta["base_global_step"],
        "optimizer_reinitialized": True,
        "scheduler_type": "cosine_without_warmup",
        "scheduler_reinitialized": True,
        "base_optimizer_loaded": False,
        "base_scheduler_loaded": False,
        "region_time_trainable_parameters": 0,
        "region_time_learning_rate": None,
        "region_time_learning_rate_reason": "Hard tau-map builder is rule-based",
        "loss": "epsilon_prediction_mse",
        "forward_diffusion": "q_sample_region",
        "tau_rule": {"Background": 0, "LM": 0, "FC": "t", "LC": "t", "VV": "t"},
        "soft_mask_used": False,
        "source_reinjection_used": False,
        "architecture_changed": False,
        "data_split_changed": False,
        "fold1_test_used": False,
        "base_checkpoint_meta": base_meta,
    }
    (output / "configs" / "resolved_stage2_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    global_step = int(base_meta["base_global_step"])
    best_loss = float("inf")
    epoch_csv = output / "logs" / "epoch_loss.csv"
    epoch_jsonl = output / "logs" / "stage2_train.jsonl"
    validations: list[dict[str, object]] = []
    for stage2_epoch in range(1, args.stage2_epochs + 1):
        total_epoch = args.base_total_epoch + stage2_epoch
        sampler.set_epoch(total_epoch)
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
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        for batch in loader:
            x0 = batch["target_image"].to(device, non_blocking=True)
            segmentation = batch["segmentation_map"].to(device, non_blocking=True)
            lesion = lesion_mask_from_segmentation(segmentation).to(x0.dtype)
            timestep = torch.randint(
                0, args.num_train_timesteps, (x0.shape[0],), device=device, dtype=torch.long
            )
            tau, tau_condition = build_tau_map(
                segmentation, timestep, args.num_train_timesteps
            )
            noise = torch.randn_like(x0)
            x_tau, effective_noise = q_sample_region(
                x0, tau, noise_scheduler.alphas_cumprod, noise=noise, lesion_mask=lesion
            )
            masked = hard_masked_source(x0, lesion)
            model_input = torch.cat([x_tau, masked, segmentation, tau_condition], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(model_input, timestep).sample
                loss = F.mse_loss(prediction.float(), effective_noise.float())
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"NaN/Inf loss at stage2_epoch={stage2_epoch}, total_epoch={total_epoch}"
                )
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
        exposure_ok = all(
            pool_counts.get(pool, 0) == VIRTUAL_EPOCH_SAMPLE_COUNTS[pool] for pool in POOL_ORDER
        )
        if not exposure_ok:
            raise RuntimeError(f"Exposure mismatch at total epoch {total_epoch}: {dict(pool_counts)}")
        row = {
            "stage2_epoch": stage2_epoch,
            "total_epoch": total_epoch,
            "global_step": global_step,
            "optimizer_steps": batch_count,
            "loss": mean_loss,
            "best_stage2_loss": best_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "pool_exposure": json.dumps(dict(pool_counts), sort_keys=True),
            "role_exposure": json.dumps(dict(role_counts), sort_keys=True),
            "exposure_counts_ok": exposure_ok,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "mean_step_seconds": (time.time() - started) / max(1, batch_count),
            "fold1_test_used": False,
        }
        append_csv(epoch_csv, row)
        with epoch_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if stage2_epoch in (5, 10):
            checkpoint_name = f"stage2_epoch{stage2_epoch}.pt"
            save_region_time_checkpoint(
                output / "checkpoints" / checkpoint_name,
                model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                epoch=total_epoch,
                global_step=global_step,
                best_loss=best_loss,
                config=config,
                parent_meta=base_meta,
            )
            validations.append(
                generate_validation(
                    model,
                    dataset,
                    fixed,
                    output,
                    total_epoch,
                    device,
                    args.eval_batch_size,
                    args.ddim_steps,
                    args.seed + 900000,
                )
            )

    final_path = output / "checkpoints" / "stage2_final.pt"
    save_region_time_checkpoint(
        final_path,
        model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        epoch=25,
        global_step=global_step,
        best_loss=best_loss,
        config=config,
        parent_meta=base_meta,
    )
    summary = {
        "completed": True,
        "base_checkpoint": base_meta,
        "stage2_epochs": 10,
        "total_epoch_range": [16, 25],
        "global_step": global_step,
        "best_stage2_loss": best_loss,
        "checkpoints": {
            "stage2_epoch5_total_epoch20": str(output / "checkpoints" / "stage2_epoch5.pt"),
            "stage2_epoch10_total_epoch25": str(output / "checkpoints" / "stage2_epoch10.pt"),
            "stage2_final_total_epoch25": str(final_path),
        },
        "validations": validations,
        "recommended_checkpoint": "pending_manual_visual_review",
        "fold1_test_used": False,
        "synthetic_pool_generation_started": False,
        "segmentation_training_started": False,
    }
    (output / "reports" / "stage2_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

