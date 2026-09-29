#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from diffusers import DDIMScheduler
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
from lesionstatediff.region_time_stage3_checkpoint import load_region_time_stage3_base
from lesionstatediff.virtual_epoch_sampler import (
    STAGE3_VIRTUAL_EPOCH_SAMPLE_COUNTS,
    Stage3VirtualEpochSampler,
)
from lesionstatediff.unified_four_class_dataset import POOL_ORDER, load_unified_records_json


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
        if record.image_id in seen or record.fc_pixels + record.lc_pixels + record.vv_pixels <= 0:
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
    stage3_epoch: int,
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
    epoch_dir = output_dir / "validation_stage3" / f"epoch{stage3_epoch}"
    for name in ("original", "generated", "tau", "difference", "segmentation"):
        (epoch_dir / name).mkdir(parents=True, exist_ok=True)

    diff_sums = {value: 0.0 for value in CLASS_NAMES}
    pixel_counts = {value: 0 for value in CLASS_NAMES}
    nonempty_counts = {value: 0 for value in CLASS_NAMES}
    generated_ids: list[str] = []
    black = white = near_constant = nan_inf = 0
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
        nan_inf += int((~torch.isfinite(generated)).any().item())

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
        raise RuntimeError(f"Protection drift detected at Stage3 epoch {stage3_epoch}: {mean_changes}")
    if nan_inf:
        raise RuntimeError(f"NaN/Inf generated outputs at Stage3 epoch {stage3_epoch}: {nan_inf}")

    contact_sheet = build_contact_sheet(epoch_dir, generated_ids)
    summary = {
        "stage3_epoch": stage3_epoch,
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
        "nan_inf_count": nan_inf,
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


def write_run_config(path: Path, config: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_final_report(
    path: Path,
    *,
    summary: dict[str, object],
    epoch_rows: list[dict[str, object]],
    validations: list[dict[str, object]],
) -> None:
    base = summary["base_checkpoint"]
    checkpoints = summary["checkpoints"]
    lines = [
        "# Stage3 Refinement Report",
        "",
        "## Task Information",
        "",
        f"- Task: `{summary['task_name']}`",
        f"- Start time: `{summary['start_time']}`",
        f"- End time: `{summary['end_time']}`",
        f"- Device: `{summary['device']['name']}`",
        f"- Output directory: `{summary['output_dir']}`",
        "",
        "## Training Configuration",
        "",
        f"- Base checkpoint: `{base['checkpoint']}`",
        f"- Base SHA256: `{base['sha256']}`",
        "- Stage3 epoch range: 1-5 (total epoch 21-25)",
        "- New optimizer steps: 8000",
        f"- Final global step: {summary['global_step']}",
        "- Batch size: 5",
        "- Samples per virtual epoch: 8000",
        "- Optimizer steps per epoch: 1600",
        "- UNet learning rate: 1e-6",
        "- RegionTime learning rate: not applicable; the tau builder has no trainable parameters",
        "- Scheduler: constant",
        "- Precision: BF16 autocast",
        "- Training timesteps: 1000",
        "- DDIM validation steps: 25",
        "- Seed: 3",
        "- Fold1 test used: `false`",
        "- Synthetic pool generation started: `false`",
        "- Segmentation training started: `false`",
        "",
        "## Checkpoint Reload Audit",
        "",
        f"- Loaded layers: {base['loaded_layers']}",
        f"- Missing keys: `{base['missing_keys']}`",
        f"- Unexpected keys: `{base['unexpected_keys']}`",
        f"- Channel remapping applied: `{str(base['channel_remap_applied']).lower()}`",
        f"- Input channels: `{base['input_channels']}`",
        f"- conv_in.weight shape: `{base['conv_in_shape']}`",
        f"- Output channels: {base['output_channels']}",
        "- Parent optimizer state loaded: `false`",
        "- Parent scheduler state loaded: `false`",
        "",
        "## Epoch Losses",
        "",
    ]
    lines.extend(
        markdown_table(
            ["Stage3 epoch", "Total epoch", "Global step", "Loss", "LR"],
            [
                [
                    row["stage3_epoch"],
                    row["total_epoch"],
                    row["global_step"],
                    f"{float(row['loss']):.9f}",
                    f"{float(row['lr']):.3e}",
                ]
                for row in epoch_rows
            ],
        )
    )
    lines.extend(["", "## Fixed-100 Validation", ""])
    lines.extend(
        markdown_table(
            ["Epoch", "BG", "LM", "FC", "LC", "VV", "Black/white/constant", "Protection"],
            [
                [
                    item["stage3_epoch"],
                    f"{item['mean_abs_change']['Background']:.8f}",
                    f"{item['mean_abs_change']['LM']:.8f}",
                    f"{item['mean_abs_change']['FC']:.8f}",
                    f"{item['mean_abs_change']['LC']:.8f}",
                    f"{item['mean_abs_change']['VV']:.8f}",
                    f"{item['black_image_count']}/{item['white_image_count']}/{item['near_constant_image_count']}",
                    "passed" if item["background_protection_passed"] and item["lm_protection_passed"] else "failed",
                ]
                for item in validations
            ],
        )
    )
    lines.extend(["", "## Outputs", "", "Checkpoints:", ""])
    lines.extend([f"- `{value}`" for value in checkpoints.values()])
    lines.extend(["", "Validation and contact sheets:", ""])
    lines.extend([f"- `{item['contact_sheet']}`" for item in validations])
    lines.extend(
        [
            "",
            f"Machine-readable summary: `{summary['machine_readable_summary']}`",
            "",
            "## Recommendation Gate",
            "",
            "- Engineering run: passed",
            "- Background and LM protection: passed",
            "- Checkpoint recommendation: pending manual visual review",
            "",
            "## Constraint Checklist",
            "",
            "- [x] Model structure unchanged",
            "- [x] A backbone unchanged",
            "- [x] B tau rule unchanged",
            "- [x] No soft mask",
            "- [x] No source reinjection",
            "- [x] No final blend",
            "- [x] Loss unchanged",
            "- [x] Fold1 test not used",
            "- [x] Synthetic pool generation not started",
            "- [x] Segmentation training not started",
            "- [x] Learning rate did not decay to zero",
            "",
            "## Known Limitation",
            "",
            "Stage3 excludes the fixed-100 image IDs from all new sampler plans. The Stage2 parent was trained with a sampler that did not explicitly exclude these IDs, so the fixed-100 set is a deterministic engineering comparison set rather than an independent unbiased validation set.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--stage2-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage3-epochs", type=int, default=5)
    parser.add_argument("--base-total-epoch", type=int, default=20)
    parser.add_argument("--virtual-epoch-samples", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--unet-lr", type=float, default=1e-6)
    parser.add_argument("--region-time-lr", type=float, default=3e-6)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--fixed-count", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--step-log-interval", type=int, default=50)
    args = parser.parse_args()

    require_no_test_path(args.train_manifest, "train_manifest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage3 training")
    if args.stage3_epochs != 5 or args.base_total_epoch != 20:
        raise ValueError("Formal Stage3 is fixed to five epochs after total epoch 20")
    if args.virtual_epoch_samples != 8000:
        raise ValueError("Formal Stage3 requires exactly 8000 samples per virtual epoch")
    if args.batch_size != 5:
        raise ValueError("Formal Stage3 requires batch size 5")
    if args.virtual_epoch_samples % args.batch_size:
        raise ValueError("batch_size must divide virtual_epoch_samples")
    if args.unet_lr != 1e-6:
        raise ValueError("Formal Stage3 UNet learning rate must be 1e-6")

    steps_per_epoch = args.virtual_epoch_samples // args.batch_size
    stage3_steps = steps_per_epoch * args.stage3_epochs
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    start_time = datetime.now(timezone.utc).astimezone().isoformat()

    model, _, base_meta = load_region_time_stage3_base(
        args.base_checkpoint, expected_sha256=args.expected_sha256
    )
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": args.unet_lr, "name": "unet"}]
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_timesteps)

    records = load_unified_records_json(args.train_manifest)
    dataset = RegionTimeDataset(args.train_manifest)
    fixed = fixed_indices(dataset, args.fixed_count)
    fixed_ids = [dataset.records[index].image_id for index in fixed]
    fixed_id_set = set(fixed_ids)

    stage2_summary = json.loads(Path(args.stage2_summary).read_text(encoding="utf-8"))
    for prior_validation in stage2_summary.get("validations", []):
        if prior_validation.get("fixed_case_ids") != fixed_ids:
            raise RuntimeError("Stage3 fixed-100 cases differ from Stage2")

    sampler = Stage3VirtualEpochSampler(
        records,
        steps_per_virtual_epoch=steps_per_epoch,
        batch_size=args.batch_size,
        seed=args.seed,
        excluded_image_ids=fixed_id_set,
    )
    output = Path(args.output_dir)
    if (output / "checkpoints" / "stage3_final.pt").exists():
        raise FileExistsError(f"Stage3 final output already exists: {output}")
    for name in ("checkpoints", "logs", "configs", "validation_stage3", "reports"):
        (output / name).mkdir(parents=True, exist_ok=True)

    run_config = vars(args) | {
        "task_name": "2026-09-02-OCT-HardRegionTime-Stage3-8000Exposure-5Epoch-Refinement",
        "training_total_epoch_range": [21, 25],
        "steps_per_epoch": steps_per_epoch,
        "stage3_optimizer_steps": stage3_steps,
        "base_global_step": base_meta["base_global_step"],
        "optimizer": "AdamW",
        "optimizer_reinitialized": True,
        "scheduler_type": "constant_lambda_lr",
        "scheduler_min_lr": args.unet_lr,
        "scheduler_reinitialized": True,
        "base_optimizer_loaded": False,
        "base_scheduler_loaded": False,
        "region_time_trainable_parameters": 0,
        "region_time_learning_rate": None,
        "requested_region_time_learning_rate": args.region_time_lr,
        "region_time_learning_rate_reason": "Hard tau-map builder is rule-based",
        "virtual_epoch_sample_counts": STAGE3_VIRTUAL_EPOCH_SAMPLE_COUNTS,
        "loss": "epsilon_prediction_mse",
        "forward_diffusion": "q_sample_region",
        "tau_rule": {"Background": 0, "LM": 0, "FC": "t", "LC": "t", "VV": "t"},
        "input_channels": ["x_tau", "hard_masked_oct", "segmentation", "tau_normalized"],
        "soft_mask_used": False,
        "source_reinjection_used": False,
        "final_blend_used": False,
        "architecture_changed": False,
        "data_split_changed": False,
        "fixed_case_count": len(fixed_ids),
        "fixed_cases_excluded_from_stage3_training": True,
        "fixed_case_ids": fixed_ids,
        "parent_fixed_cases_may_have_prior_training_exposure": True,
        "fold1_test_used": False,
        "synthetic_pool_generation_started": False,
        "segmentation_training_started": False,
        "base_checkpoint_meta": base_meta,
    }
    write_run_config(output / "configs" / "run_config.yaml", run_config)

    global_step = int(base_meta["base_global_step"])
    best_loss = float("inf")
    epoch_csv = output / "logs" / "train_loss_by_epoch.csv"
    step_csv = output / "logs" / "train_loss_by_step.csv"
    epoch_jsonl = output / "logs" / "stage3_train.jsonl"
    validation_csv = output / "reports" / "stage3_fixed100_summary.csv"
    epoch_rows: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []

    for stage3_epoch in range(1, args.stage3_epochs + 1):
        total_epoch = args.base_total_epoch + stage3_epoch
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
        fixed_exposure_count = 0
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)

        for batch in loader:
            x0 = batch["target_image"].to(device, non_blocking=True)
            segmentation = batch["segmentation_map"].to(device, non_blocking=True)
            lesion = lesion_mask_from_segmentation(segmentation).to(x0.dtype)
            timestep = torch.randint(
                0,
                args.num_train_timesteps,
                (x0.shape[0],),
                device=device,
                dtype=torch.long,
            )
            tau, tau_condition = build_tau_map(
                segmentation, timestep, args.num_train_timesteps
            )
            noise = torch.randn_like(x0)
            x_tau, effective_noise = q_sample_region(
                x0,
                tau,
                noise_scheduler.alphas_cumprod,
                noise=noise,
                lesion_mask=lesion,
            )
            masked = hard_masked_source(x0, lesion)
            model_input = torch.cat([x_tau, masked, segmentation, tau_condition], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(model_input, timestep).sample
                loss = F.mse_loss(prediction.float(), effective_noise.float())
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"NaN/Inf loss at Stage3 epoch={stage3_epoch}, total_epoch={total_epoch}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            global_step += 1
            batch_count += 1
            loss_value = float(loss.detach().cpu())
            epoch_loss += loss_value
            pool_counts.update(batch["pool_name"])
            role_counts.update(batch["target_role"])
            fixed_exposure_count += sum(image_id in fixed_id_set for image_id in batch["image_id"])

            if batch_count == 1 or batch_count % args.step_log_interval == 0 or batch_count == steps_per_epoch:
                append_csv(
                    step_csv,
                    {
                        "stage3_epoch": stage3_epoch,
                        "total_epoch": total_epoch,
                        "epoch_step": batch_count,
                        "global_step": global_step,
                        "loss": loss_value,
                        "lr": optimizer.param_groups[0]["lr"],
                        "fold1_test_used": False,
                    },
                )

        mean_loss = epoch_loss / max(1, batch_count)
        best_loss = min(best_loss, mean_loss)
        exposure_ok = all(
            pool_counts.get(pool, 0) == STAGE3_VIRTUAL_EPOCH_SAMPLE_COUNTS[pool]
            for pool in POOL_ORDER
        )
        if not exposure_ok:
            raise RuntimeError(f"Exposure mismatch at Stage3 epoch {stage3_epoch}: {dict(pool_counts)}")
        if fixed_exposure_count:
            raise RuntimeError(
                f"Fixed validation samples entered Stage3 training: {fixed_exposure_count}"
            )
        if optimizer.param_groups[0]["lr"] < 5e-7:
            raise RuntimeError(f"Stage3 learning rate fell below the allowed floor: {optimizer.param_groups[0]['lr']}")

        row = {
            "stage3_epoch": stage3_epoch,
            "total_epoch": total_epoch,
            "global_step": global_step,
            "optimizer_steps": batch_count,
            "loss": mean_loss,
            "best_stage3_loss": best_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "pool_exposure": json.dumps(dict(pool_counts), sort_keys=True),
            "role_exposure": json.dumps(dict(role_counts), sort_keys=True),
            "exposure_counts_ok": exposure_ok,
            "fixed_validation_exposure_count": fixed_exposure_count,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "mean_step_seconds": (time.time() - started) / max(1, batch_count),
            "fold1_test_used": False,
        }
        epoch_rows.append(row)
        append_csv(epoch_csv, row)
        with epoch_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        checkpoint_path = output / "checkpoints" / f"stage3_epoch{stage3_epoch}.pt"
        save_region_time_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            epoch=total_epoch,
            global_step=global_step,
            best_loss=best_loss,
            config=run_config,
            parent_meta=base_meta,
        )
        validation = generate_validation(
            model,
            dataset,
            fixed,
            output,
            stage3_epoch,
            total_epoch,
            device,
            args.eval_batch_size,
            args.ddim_steps,
            args.seed + 900000,
        )
        validations.append(validation)
        changes = validation["mean_abs_change"]
        append_csv(
            validation_csv,
            {
                "stage3_epoch": stage3_epoch,
                "total_epoch": total_epoch,
                "global_step": global_step,
                "loss": mean_loss,
                "background_change": changes["Background"],
                "lm_change": changes["LM"],
                "fc_change": changes["FC"],
                "lc_change": changes["LC"],
                "vv_change": changes["VV"],
                "protection": "passed",
                "note": "",
            },
        )

    final_path = output / "checkpoints" / "stage3_final.pt"
    save_region_time_checkpoint(
        final_path,
        model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        epoch=25,
        global_step=global_step,
        best_loss=best_loss,
        config=run_config,
        parent_meta=base_meta,
    )
    end_time = datetime.now(timezone.utc).astimezone().isoformat()
    checkpoint_paths = {
        f"stage3_epoch{index}": str(output / "checkpoints" / f"stage3_epoch{index}.pt")
        for index in range(1, 6)
    }
    checkpoint_paths["stage3_final"] = str(final_path)
    machine_summary = output / "reports" / "stage3_training_summary.json"
    summary = {
        "completed": True,
        "task_name": "2026-09-02-OCT-HardRegionTime-Stage3-8000Exposure-5Epoch-Refinement",
        "start_time": start_time,
        "end_time": end_time,
        "output_dir": str(output),
        "device": {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "base_checkpoint": base_meta,
        "stage3_epochs": 5,
        "total_epoch_range": [21, 25],
        "new_optimizer_steps": stage3_steps,
        "global_step": global_step,
        "best_stage3_loss": best_loss,
        "scheduler_type": "constant_lambda_lr",
        "minimum_observed_lr": min(float(row["lr"]) for row in epoch_rows),
        "checkpoints": checkpoint_paths,
        "validations": validations,
        "fixed_cases_excluded_from_stage3_training": True,
        "parent_fixed_cases_may_have_prior_training_exposure": True,
        "recommended_checkpoint": "pending_manual_visual_review",
        "fold1_test_used": False,
        "synthetic_pool_generation_started": False,
        "segmentation_training_started": False,
        "machine_readable_summary": str(machine_summary),
    }
    machine_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_final_report(
        output / "reports" / "STAGE3_REFINEMENT_REPORT.md",
        summary=summary,
        epoch_rows=epoch_rows,
        validations=validations,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

