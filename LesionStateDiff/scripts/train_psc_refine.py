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
)
from lesionstatediff.region_time_b3_checkpoint import reload_region_time_b3_checkpoint
from lesionstatediff.region_time_b3_refinement_checkpoint import (
    load_b3_refinement_parent,
    save_region_time_b3_checkpoint,
    semantic_norms,
)
from lesionstatediff.region_time_b3_semantic import sample_region_time_ddim_b3
from lesionstatediff.region_time_dataset import RegionTimeDataset
from lesionstatediff.region_time_stage3_sampler import (
    STAGE3_VIRTUAL_EPOCH_SAMPLE_COUNTS,
    Stage3VirtualEpochSampler,
)
from lesionstatediff.unified_four_class_dataset import POOL_ORDER, load_unified_records_json


TASK_NAME = "2026-09-03-OCT-B3-SemanticRefinement-Epoch3-5Epoch"
CLASS_NAMES = {0: "Background", 1: "LM", 2: "FC", 3: "LC", 4: "VV"}


def collate(batch):
    return {
        "target_image": torch.stack([row["target_image"] for row in batch]),
        "segmentation_map": torch.stack([row["segmentation_map"] for row in batch]),
        "image_id": [row["image_id"] for row in batch],
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


def image_u8(image: torch.Tensor) -> np.ndarray:
    value = crop_768_to_750(image).detach().cpu().float().squeeze().numpy()
    return np.clip((value + 1.0) * 127.5, 0, 255).astype(np.uint8)


def mask_u8(mask: torch.Tensor) -> np.ndarray:
    return crop_768_to_750(mask).detach().cpu().squeeze().numpy().astype(np.uint8)


def activation_u8(activation: torch.Tensor) -> np.ndarray:
    value = crop_768_to_750(activation).detach().cpu().float().squeeze().numpy()
    maximum = float(value.max())
    if maximum > 0:
        value = value / maximum
    return np.clip(value * 255.0, 0, 255).astype(np.uint8)


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


def build_checkpoint_comparison(
    output_dir: Path,
    image_ids: list[str],
    b1_validation_dir: Path,
    parent_b3_validation_dir: Path,
) -> Path:
    segmentation_dir = output_dir / "validation" / "epoch1" / "segmentation"
    selected: list[tuple[str, str]] = []
    used: set[str] = set()
    for class_name, class_value in (("FC", 2), ("LC", 3), ("VV", 4)):
        count = 0
        for image_id in image_ids:
            if image_id in used:
                continue
            mask = np.asarray(Image.open(segmentation_dir / f"{image_id}.png"))
            if np.any(mask == class_value):
                selected.append((class_name, image_id))
                used.add(image_id)
                count += 1
                if count == 4:
                    break
        if count != 4:
            raise RuntimeError(f"Could not select four representative {class_name} cases")

    columns = [
        ("Original", parent_b3_validation_dir / "original"),
        ("B1 S3 E4", b1_validation_dir / "generated"),
        ("B3 parent E3", parent_b3_validation_dir / "generated"),
    ]
    columns.extend(
        (f"Refine E{epoch}", output_dir / "validation" / f"epoch{epoch}" / "generated")
        for epoch in range(1, 6)
    )
    cell = 128
    label_height = 24
    row_label_width = 150
    sheet = Image.new(
        "RGB",
        (row_label_width + len(columns) * cell, label_height + len(selected) * cell),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for column_index, (title, _) in enumerate(columns):
        draw.text((row_label_width + column_index * cell + 4, 5), title, fill="black")
    for row_index, (class_name, image_id) in enumerate(selected):
        y = label_height + row_index * cell
        draw.text((4, y + 4), f"{class_name}: {image_id}", fill="black")
        for column_index, (_, image_dir) in enumerate(columns):
            image = Image.open(image_dir / f"{image_id}.png").convert("L").resize((cell, cell))
            sheet.paste(image.convert("RGB"), (row_label_width + column_index * cell, y))
    destination = output_dir / "checkpoint_comparison" / "b1_b3parent_refine_e1_e5_ddim25.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return destination


@torch.no_grad()
def generate_validation(
    model,
    dataset: RegionTimeDataset,
    indices: list[int],
    output_dir: Path,
    refine_epoch: int,
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
    epoch_dir = output_dir / "validation" / f"epoch{refine_epoch}"
    for name in ("original", "generated", "segmentation", "tau", "semantic_activation", "difference"):
        (epoch_dir / name).mkdir(parents=True, exist_ok=True)

    diff_sums = {value: 0.0 for value in CLASS_NAMES}
    pixel_counts = {value: 0 for value in CLASS_NAMES}
    generated_ids: list[str] = []
    black = white = near_constant = nan_inf = 0
    semantic_activation_sum = 0.0
    semantic_activation_count = 0
    offset = 0
    for batch in loader:
        image = batch["target_image"].to(device)
        segmentation = batch["segmentation_map"].to(device)
        noises = []
        for local_index in range(image.shape[0]):
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + offset + local_index)
            noises.append(torch.randn(image[local_index].shape, device=device, dtype=image.dtype, generator=generator))
        result = sample_region_time_ddim_b3(
            model,
            scheduler,
            image,
            segmentation,
            torch.stack(noises),
            ddim_steps,
        )
        generated = result["generated"]
        semantic_activation_sum += float(result["semantic_activation"].detach().float().abs().sum().cpu())
        semantic_activation_count += result["semantic_activation"].numel()
        difference = (generated - image).abs()
        nan_inf += int((~torch.isfinite(generated)).any().item())
        for value in CLASS_NAMES:
            region = segmentation == float(value)
            count = int(region.sum().item())
            if count:
                diff_sums[value] += float(difference[region].sum().detach().cpu())
                pixel_counts[value] += count

        for local_index, image_id in enumerate(batch["image_id"]):
            original = image_u8(image[local_index])
            generated_u8 = image_u8(generated[local_index])
            segmentation_u8 = mask_u8(segmentation[local_index])
            semantic_u8 = activation_u8(result["semantic_activation"][local_index])
            tau_u8 = activation_u8(result["initial_tau_condition"][local_index])
            difference_u8 = np.clip(
                crop_768_to_750(difference[local_index]).detach().cpu().float().squeeze().numpy() * 127.5,
                0,
                255,
            ).astype(np.uint8)
            save_png(epoch_dir / "original" / f"{image_id}.png", original)
            save_png(epoch_dir / "generated" / f"{image_id}.png", generated_u8)
            save_png(epoch_dir / "segmentation" / f"{image_id}.png", segmentation_u8)
            save_png(epoch_dir / "tau" / f"{image_id}.png", tau_u8)
            save_png(epoch_dir / "semantic_activation" / f"{image_id}.png", semantic_u8)
            save_png(epoch_dir / "difference" / f"{image_id}.png", difference_u8)
            black += int(generated_u8.mean() < 2.0)
            white += int(generated_u8.mean() > 253.0)
            near_constant += int(generated_u8.std() < 2.0)
            generated_ids.append(image_id)
        offset += image.shape[0]

    changes = {CLASS_NAMES[value]: diff_sums[value] / max(1, pixel_counts[value]) for value in CLASS_NAMES}
    if changes["Background"] > 1e-7 or changes["LM"] > 1e-7:
        raise RuntimeError(f"Protection drift at refinement epoch {refine_epoch}: {changes}")
    if nan_inf:
        raise RuntimeError(f"NaN/Inf generated outputs at refinement epoch {refine_epoch}: {nan_inf}")
    contact_sheet = build_contact_sheet(epoch_dir, generated_ids)
    summary = {
        "refine_epoch": refine_epoch,
        "branch_total_epoch": total_epoch,
        "generated_count": len(generated_ids),
        "fixed_case_ids": generated_ids,
        "fixed_seed_base": seed,
        "ddim_steps": ddim_steps,
        "mean_abs_change": changes,
        "black_image_count": black,
        "white_image_count": white,
        "near_constant_image_count": near_constant,
        "nan_inf_count": nan_inf,
        "semantic_activation_abs_mean": semantic_activation_sum / max(1, semantic_activation_count),
        "background_protection_passed": changes["Background"] <= 1e-7,
        "lm_protection_passed": changes["LM"] <= 1e-7,
        "contact_sheet": str(contact_sheet),
        "fold1_test_used": False,
    }
    (epoch_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def write_report(
    path: Path,
    summary: dict[str, object],
    epoch_rows: list[dict[str, object]],
    semantic_rows: list[dict[str, object]],
    validations: list[dict[str, object]],
    parent_validation: dict[str, object],
) -> None:
    base = summary["base_checkpoint"]
    lines = [
        "# B3 Semantic Refinement Report",
        "",
        "## Task",
        "",
        f"- Task: `{TASK_NAME}`",
        f"- Start: `{summary['start_time']}`",
        f"- End: `{summary['end_time']}`",
        f"- GPU: `{summary['device']['name']}`",
        f"- Output: `{summary['output_dir']}`",
        "- Epochs: 5",
        "- Exposures: 40000",
        "- Optimizer steps: 8000",
        "- Batch size: 5",
        "- Learning rates: UNet 1e-7; semantic branch 1e-5; constant",
        "- Precision: BF16",
        f"- Backup completed: `{str(summary['backup_completed']).lower()}`",
        f"- Backup path: `{summary['backup_path']}`",
        f"- Source code root: `{summary['source_code_root']}`",
        "- Fold1 test used: `false`",
        "- Synthetic pool started: `false`",
        "- Segmentation training started: `false`",
        "",
        "## Checkpoint Audit",
        "",
        f"- Base checkpoint: `{base['checkpoint']}`",
        f"- SHA256: `{base['sha256']}`",
        f"- Parent total epoch / B3 epoch / global step: `{base['parent_total_epoch']}/{base['parent_b3_epoch']}/{base['parent_global_step']}`",
        f"- Loaded UNet / semantic layers: `{base['loaded_unet_layers']}/{base['loaded_semantic_layers']}`",
        f"- UNet missing/unexpected: `{base['unet_missing_keys']}/{base['unet_unexpected_keys']}`",
        f"- Semantic missing/unexpected: `{base['semantic_missing_keys']}/{base['semantic_unexpected_keys']}`",
        f"- conv_in: `{base['conv_in_shape']}` unchanged",
        f"- UNet input contract: `{base['input_channels']}`",
        "- Parent model state loaded: `true`",
        "- Parent semantic state loaded: `true`",
        "- Parent semantic state reinitialized: `false`",
        "- Parent optimizer/scheduler loaded: `false/false`",
        "",
        "## Model Change",
        "",
        "- B1 tau retained: `Background=0, LM=0, FC=t, LC=t, VV=t`",
        "- Existing B3 branch retained: 5-class x 32-d pixel embedding and trained 1x1 projection.",
        "- No zero initialization was applied after loading the parent B3 semantic state.",
        "- Injection: additive 512-channel feature immediately before `mid_block`.",
        "- `q_sample_region()` changed: `false`",
        "- DDIM equation changed: `false`",
        "- A/UNet backbone, input channels and epsilon-MSE loss changed: `false`",
        "- Source reinjection / soft blend / post-hoc mask restore: `false/false/false`",
        "- Inherited tau=0 identity transition: `true`",
        "",
        "## Loss",
        "",
    ]
    lines.extend(markdown_table(
        ["Refine epoch", "Cumulative global step", "Loss", "UNet LR", "Semantic LR"],
        [[r["refine_epoch"], r["global_step"], f"{r['loss']:.9f}", f"{r['unet_lr']:.3e}", f"{r['semantic_lr']:.3e}"] for r in epoch_rows],
    ))
    lines.extend(["", "## Semantic Diagnostics", ""])
    lines.extend(markdown_table(
        ["Epoch", "Embedding norm", "Projection weight norm", "Projection bias norm", "Activation mean abs"],
        [[r["refine_epoch"], f"{r['embedding_norm']:.8f}", f"{r['projection_weight_norm']:.8f}", f"{r['projection_bias_norm']:.8f}", f"{r['semantic_activation_abs_mean']:.8f}"] for r in semantic_rows],
    ))
    comparison = [[
        "B3 parent epoch3",
        f"{parent_validation['mean_abs_change']['FC']:.8f}",
        f"{parent_validation['mean_abs_change']['LC']:.8f}",
        f"{parent_validation['mean_abs_change']['VV']:.8f}",
        f"{parent_validation['mean_abs_change']['Background']:.8f}",
        f"{parent_validation['mean_abs_change']['LM']:.8f}",
    ]]
    comparison.extend([
        [
            f"Refine epoch{v['refine_epoch']}",
            f"{v['mean_abs_change']['FC']:.8f}",
            f"{v['mean_abs_change']['LC']:.8f}",
            f"{v['mean_abs_change']['VV']:.8f}",
            f"{v['mean_abs_change']['Background']:.8f}",
            f"{v['mean_abs_change']['LM']:.8f}",
        ] for v in validations
    ])
    lines.extend(["", "## Fixed-100 Parent/Refinement Comparison", ""])
    lines.extend(markdown_table(["Method", "FC", "LC", "VV", "BG", "LM"], comparison))
    lines.extend([
        "",
        "## Visual Review",
        "",
        "- All 100 originals, generated images, segmentations, tau maps, semantic activation maps and differences are saved for every refinement epoch.",
        "- Contact sheets are listed below. Medical texture quality remains `pending manual visual review`.",
    ])
    lines.extend(f"- `{v['contact_sheet']}`" for v in validations)
    lines.extend([
        "",
        "## Outputs",
        "",
    ])
    lines.extend(f"- `{value}`" for value in summary["checkpoints"].values())
    lines.extend([
        "",
        "## Conclusion Gate",
        "",
        "- Engineering completion: passed",
        "- Semantic branch active: passed",
        "- BG/LM protection: passed",
        "- Medical visual quality: pending manual review",
        "- Recommended checkpoint: pending manual visual review",
        "- Synthetic pool generated: no",
        "- Segmentation training started: no",
        "",
        "## Strict Checklist",
        "",
        "- [x] Started from B3 epoch3",
        "- [x] B3 semantic weights loaded and not reinitialized",
        "- [x] Original B3 project backed up",
        "- [x] New output directory used",
        "- [x] UNet architecture and conv_in unchanged",
        "- [x] Four-channel input unchanged",
        "- [x] Semantic architecture and injection position unchanged",
        "- [x] B1 tau rule retained; B2 class-aware tau not used",
        "- [x] q_sample_region, DDIM equation and epsilon-MSE unchanged",
        "- [x] No soft mask, source reinjection or final blend",
        "- [x] Fold1 test unused and fixed100 excluded from new training",
        "- [x] No synthetic pool or segmentation training started",
        "",
        "## Limitation",
        "",
        "The fixed-100 IDs are excluded from B3 and Stage3 training, but the earlier Stage2 ancestor did not explicitly exclude them. These cases are therefore a deterministic engineering comparison set, not an independent unbiased validation set.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--parent-b3-summary", required=True)
    parser.add_argument("--b1-validation-dir", required=True)
    parser.add_argument("--parent-b3-validation-dir", required=True)
    parser.add_argument("--backup-path", required=True)
    parser.add_argument("--source-code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--base-total-epoch", type=int, default=27)
    parser.add_argument("--virtual-epoch-samples", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--unet-lr", type=float, default=1e-7)
    parser.add_argument("--semantic-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--fixed-count", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--step-log-interval", type=int, default=50)
    args = parser.parse_args()

    require_no_test_path(args.train_manifest, "train_manifest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B3 semantic refinement training")
    if args.epochs != 5 or args.base_total_epoch != 27:
        raise ValueError("Formal refinement requires five epochs from B3 parent total epoch 27")
    if args.virtual_epoch_samples != 8000 or args.batch_size != 5:
        raise ValueError("Formal refinement requires 8000 exposures and batch size 5 per epoch")
    if args.virtual_epoch_samples % args.batch_size:
        raise ValueError("batch size must divide virtual epoch exposures")
    if args.unet_lr != 1e-7:
        raise ValueError("Formal refinement UNet learning rate must be 1e-7")
    if args.semantic_lr != 1e-5:
        raise ValueError("Formal refinement semantic branch learning rate must be 1e-5")
    if not Path(args.backup_path).is_dir():
        raise FileNotFoundError(f"Required B3 backup is missing: {args.backup_path}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    start_time = datetime.now(timezone.utc).astimezone().isoformat()
    steps_per_epoch = args.virtual_epoch_samples // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    model, _, base_meta = load_b3_refinement_parent(
        args.base_checkpoint, expected_sha256=args.expected_sha256
    )
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
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_timesteps)

    records = load_unified_records_json(args.train_manifest)
    dataset = RegionTimeDataset(args.train_manifest)
    fixed = fixed_indices(dataset, args.fixed_count)
    fixed_ids = [dataset.records[index].image_id for index in fixed]
    fixed_id_set = set(fixed_ids)
    parent_summary = json.loads(Path(args.parent_b3_summary).read_text(encoding="utf-8"))
    prior_validations = parent_summary.get("validations", [])
    if not prior_validations or any(v.get("fixed_case_ids") != fixed_ids for v in prior_validations):
        raise RuntimeError("Refinement fixed-100 cases differ from parent B3")
    parent_validation = next(
        (v for v in prior_validations if int(v.get("b3_epoch", -1)) == 3), None
    )
    if parent_validation is None:
        raise RuntimeError("Parent B3 epoch3 Fixed100 baseline is missing")
    if int(parent_validation.get("ddim_steps", -1)) != args.ddim_steps or int(parent_validation.get("fixed_seed_base", -1)) != 900003:
        raise RuntimeError("Parent B3 epoch3 sampling is not seed=900003/DDIM=25")

    sampler = Stage3VirtualEpochSampler(
        records,
        steps_per_virtual_epoch=steps_per_epoch,
        batch_size=args.batch_size,
        seed=args.seed,
        excluded_image_ids=fixed_id_set,
    )
    output = Path(args.output_dir)
    final_path = output / "checkpoints" / "semantic_refine_final.pt"
    if final_path.exists():
        raise FileExistsError(f"B3 final output already exists: {final_path}")
    for name in ("checkpoints", "logs", "configs", "validation", "reports", "checkpoint_comparison"):
        (output / name).mkdir(parents=True, exist_ok=True)

    run_config = vars(args) | {
        "task_name": TASK_NAME,
        "branch_total_epoch_range": [28, 32],
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "total_exposures": args.virtual_epoch_samples * args.epochs,
        "base_global_step": base_meta["parent_global_step"],
        "optimizer": "AdamW",
        "optimizer_reinitialized": True,
        "scheduler": "constant LambdaLR",
        "scheduler_reinitialized": True,
        "virtual_epoch_sample_counts": STAGE3_VIRTUAL_EPOCH_SAMPLE_COUNTS,
        "loss": "epsilon_prediction_mse",
        "forward_diffusion": "q_sample_region unchanged",
        "tau_rule": {"Background": 0, "LM": 0, "FC": "t", "LC": "t", "VV": "t"},
        "semantic_num_classes": 5,
        "semantic_embedding_dim": 32,
        "semantic_projection_channels": 512,
        "semantic_projection_zero_initialized": False,
        "semantic_parent_state_loaded": True,
        "semantic_parent_state_reinitialized": False,
        "semantic_injection": "pre_mid_block_addition",
        "semantic_lr": args.semantic_lr,
        "input_channels": ["x_tau", "hard_masked_oct", "segmentation", "tau_normalized"],
        "unet_backbone_changed": False,
        "semantic_branch_added": True,
        "q_sample_region_changed": False,
        "loss_changed": False,
        "source_reinjection_used": False,
        "soft_blend_used": False,
        "posthoc_mask_restore_used": False,
        "tau_zero_identity_transition_used": True,
        "data_split_changed": False,
        "fixed_case_ids": fixed_ids,
        "fixed_cases_excluded_from_b3_training": True,
        "fold1_test_used": False,
        "synthetic_pool_generation_started": False,
        "segmentation_training_started": False,
        "backup_completed": True,
        "backup_path": args.backup_path,
        "source_code_root": args.source_code_root,
        "base_checkpoint_meta": base_meta,
    }
    (output / "configs" / "refinement_config.yaml").write_text(
        yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8"
    )

    global_step = int(base_meta["parent_global_step"])
    best_loss = float("inf")
    epoch_rows: list[dict[str, object]] = []
    semantic_rows: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    for refine_epoch in range(1, args.epochs + 1):
        total_epoch = args.base_total_epoch + refine_epoch
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
        pool_counts: Counter[str] = Counter()
        role_counts: Counter[str] = Counter()
        fixed_exposure = 0
        epoch_started = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        for epoch_step, batch in enumerate(loader, start=1):
            x0 = batch["target_image"].to(device, non_blocking=True)
            segmentation = batch["segmentation_map"].to(device, non_blocking=True)
            lesion = lesion_mask_from_segmentation(segmentation).to(x0.dtype)
            timestep = torch.randint(0, args.num_train_timesteps, (x0.shape[0],), device=device, dtype=torch.long)
            tau, tau_condition = build_tau_map(
                segmentation, timestep, args.num_train_timesteps
            )
            noise = torch.randn_like(x0)
            x_tau, effective_noise = q_sample_region(
                x0, tau, noise_scheduler.alphas_cumprod, noise=noise, lesion_mask=lesion
            )
            model_input = torch.cat([x_tau, hard_masked_source(x0, lesion), segmentation, tau_condition], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(
                    model_input, timestep, semantic_seg=segmentation
                ).sample
                loss = F.mse_loss(prediction.float(), effective_noise.float())
            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN/Inf loss at refinement epoch={refine_epoch}, step={epoch_step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            global_step += 1
            loss_value = float(loss.detach().cpu())
            epoch_loss += loss_value
            pool_counts.update(batch["pool_name"])
            role_counts.update(batch["target_role"])
            fixed_exposure += sum(image_id in fixed_id_set for image_id in batch["image_id"])
            if epoch_step == 1 or epoch_step % args.step_log_interval == 0 or epoch_step == steps_per_epoch:
                append_csv(output / "logs" / "train_loss_by_step.csv", {
                    "refine_epoch": refine_epoch,
                    "total_epoch": total_epoch,
                    "epoch_step": epoch_step,
                    "global_step": global_step,
                    "loss": loss_value,
                    "unet_lr": optimizer.param_groups[0]["lr"],
                    "semantic_lr": optimizer.param_groups[1]["lr"],
                    "fold1_test_used": False,
                })

        mean_loss = epoch_loss / steps_per_epoch
        best_loss = min(best_loss, mean_loss)
        if any(
            pool_counts.get(pool, 0) != STAGE3_VIRTUAL_EPOCH_SAMPLE_COUNTS[pool]
            for pool in POOL_ORDER
        ):
            raise RuntimeError(f"Refinement exposure mismatch: {dict(pool_counts)}")
        if fixed_exposure:
            raise RuntimeError(f"Fixed validation samples entered refinement training: {fixed_exposure}")
        row = {
            "refine_epoch": refine_epoch,
            "total_epoch": total_epoch,
            "global_step": global_step,
            "optimizer_steps": steps_per_epoch,
            "loss": mean_loss,
            "best_refinement_loss": best_loss,
            "unet_lr": optimizer.param_groups[0]["lr"],
            "semantic_lr": optimizer.param_groups[1]["lr"],
            "pool_exposure": json.dumps(dict(pool_counts), sort_keys=True),
            "role_exposure": json.dumps(dict(role_counts), sort_keys=True),
            "fixed_validation_exposure_count": fixed_exposure,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "mean_step_seconds": (time.time() - epoch_started) / steps_per_epoch,
            "fold1_test_used": False,
        }
        epoch_rows.append(row)
        append_csv(output / "logs" / "train_loss_by_epoch.csv", row)
        with (output / "logs" / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"event": "semantic_refinement_epoch_complete", **row}, ensure_ascii=False), flush=True)

        save_region_time_b3_checkpoint(
            output / "checkpoints" / f"semantic_refine_epoch{refine_epoch}.pt",
            model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            epoch=total_epoch,
            b3_epoch=3 + refine_epoch,
            global_step=global_step,
            best_loss=best_loss,
            config=run_config,
            parent_meta=base_meta,
        )
        validation = generate_validation(
            model, dataset, fixed, output, refine_epoch, total_epoch, device,
            args.eval_batch_size, args.ddim_steps, 900003,
        )
        validations.append(validation)
        diagnostics = {
            "refine_epoch": refine_epoch,
            **semantic_norms(model),
            "semantic_activation_abs_mean": validation["semantic_activation_abs_mean"],
        }
        semantic_rows.append(diagnostics)
        append_csv(output / "logs" / "semantic_diagnostics.csv", diagnostics)
        changes = validation["mean_abs_change"]
        append_csv(output / "reports" / "fixed100_summary.csv", {
            "refine_epoch": refine_epoch,
            "total_epoch": total_epoch,
            "global_step": global_step,
            "loss": mean_loss,
            "background_change": changes["Background"],
            "lm_change": changes["LM"],
            "fc_change": changes["FC"],
            "lc_change": changes["LC"],
            "vv_change": changes["VV"],
            "black_image_count": validation["black_image_count"],
            "white_image_count": validation["white_image_count"],
            "near_constant_image_count": validation["near_constant_image_count"],
            "protection": "passed",
        })

    save_region_time_b3_checkpoint(
        final_path,
        model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        epoch=32,
        b3_epoch=8,
        global_step=global_step,
        best_loss=best_loss,
        config=run_config,
        parent_meta=base_meta,
    )
    checkpoint_paths = {
        f"semantic_refine_epoch{index}": str(
            output / "checkpoints" / f"semantic_refine_epoch{index}.pt"
        )
        for index in range(1, 6)
    }
    checkpoint_paths["semantic_refine_final"] = str(final_path)
    _, _, final_reload = reload_region_time_b3_checkpoint(final_path, map_location="cpu")
    comparison_sheet = build_checkpoint_comparison(
        output,
        fixed_ids,
        Path(args.b1_validation_dir),
        Path(args.parent_b3_validation_dir),
    )
    end_time = datetime.now(timezone.utc).astimezone().isoformat()
    machine_summary = output / "reports" / "refinement_training_summary.json"
    summary = {
        "completed": True,
        "task_name": TASK_NAME,
        "start_time": start_time,
        "end_time": end_time,
        "output_dir": str(output),
        "backup_completed": True,
        "backup_path": args.backup_path,
        "source_code_root": args.source_code_root,
        "device": {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "base_checkpoint": base_meta,
        "refinement_epochs": 5,
        "branch_total_epoch_range": [28, 32],
        "new_optimizer_steps": total_steps,
        "total_exposures": args.virtual_epoch_samples * args.epochs,
        "global_step": global_step,
        "best_refinement_loss": best_loss,
        "checkpoints": checkpoint_paths,
        "parent_b3_epoch3_baseline": parent_validation,
        "validations": validations,
        "semantic_diagnostics": semantic_rows,
        "checkpoint_comparison_sheet": str(comparison_sheet),
        "final_checkpoint_reload": final_reload,
        "recommended_checkpoint": "pending_manual_visual_review",
        "fold1_test_used": False,
        "synthetic_pool_generation_started": False,
        "segmentation_training_started": False,
        "machine_readable_summary": str(machine_summary),
    }
    machine_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(
        output / "reports" / "B3_SEMANTIC_REFINEMENT_REPORT.md",
        summary,
        epoch_rows,
        semantic_rows,
        validations,
        parent_validation,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

