#!/usr/bin/env python3
"""Generate the B3 OCT formal FC/LC/VV synthetic pool.

This is an external experiment adapter. It deliberately reuses the audited B3
checkpoint loader and sampler without changing the original diffusion modules.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tarfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TASK = "2026-09-03-OCT-Ours-B3-Formal3000-SyntheticPool"
BASE_SEED = 3
PER_CLASS = 1000
DDIM_STEPS = 25
PAD = 9
ORIGINAL_SIZE = 750
PADDED_SIZE = 768
CLASS_VALUES = {"FC": 2, "LC": 3, "VV": 4}
CLASS_NAMES = {0: "Background", 1: "LM", 2: "FC", 3: "LC", 4: "VV"}


@dataclass(frozen=True)
class Candidate:
    image_path: str
    mask_path: str
    image_id: str
    patient_id: str
    lm_pixels: int
    fc_pixels: int
    lc_pixels: int
    vv_pixels: int


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def patient_id_from_stem(stem: str) -> str:
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


def read_patient_list(path: str | Path) -> set[str]:
    payload = read_json(path)
    if isinstance(payload, dict):
        values = payload.get("patients")
    else:
        values = payload
    if not isinstance(values, list):
        raise ValueError(f"Patient list is not a list: {path}")
    return {str(value) for value in values}


def read_class_index_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        raw = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        raw = data["mask"] if "mask" in data.files else data[data.files[0]]
    else:
        raw = np.asarray(Image.open(path))
    if raw.ndim == 2:
        result = raw.astype(np.uint8)
    elif raw.ndim == 3 and raw.shape[0] == 4:
        result = np.zeros(raw.shape[1:], dtype=np.uint8)
        occupied = np.zeros(raw.shape[1:], dtype=bool)
        for class_value, channel in ((1, 0), (2, 1), (3, 2), (4, 3)):
            current = raw[channel] > 0
            current &= ~occupied
            result[current] = class_value
            occupied |= current
    elif raw.ndim == 3 and raw.shape[-1] >= 4:
        result = np.zeros(raw.shape[:2], dtype=np.uint8)
        occupied = np.zeros(raw.shape[:2], dtype=bool)
        for class_value, channel in ((1, 0), (2, 1), (3, 2), (4, 3)):
            current = raw[..., channel] > 0
            current &= ~occupied
            result[current] = class_value
            occupied |= current
    else:
        raise ValueError(f"Unsupported mask shape {raw.shape}: {path}")
    values = set(int(value) for value in np.unique(result))
    if not values.issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"Invalid class-index values {sorted(values)}: {path}")
    if tuple(result.shape) != (ORIGINAL_SIZE, ORIGINAL_SIZE):
        raise ValueError(f"Expected 750x750 mask, got {result.shape}: {path}")
    return result


def read_grayscale(path: str | Path) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if tuple(array.shape) != (ORIGINAL_SIZE, ORIGINAL_SIZE):
        raise ValueError(f"Expected 750x750 image, got {array.shape}: {path}")
    return array


def iter_train_pairs(data_root: str | Path) -> Iterable[tuple[Path, Path]]:
    root = Path(data_root).resolve()
    image_dir = root / "train" / "img"
    mask_dir = root / "train" / "mask"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected train/img and train/mask under {root}")
    for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            matches = sorted(mask_dir.glob(image_path.stem + ".*"))
            if not matches:
                continue
            mask_path = matches[0]
        yield image_path, mask_path


def test_patients_from_names(data_root: str | Path) -> set[str]:
    image_dir = Path(data_root).resolve() / "test" / "img"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Expected test/img for patient audit: {image_dir}")
    return {patient_id_from_stem(path.stem) for path in image_dir.iterdir() if path.is_file()}


def fixed_ids_from_b3_config(config_path: str | Path) -> set[str]:
    import yaml

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    values = config.get("fixed_case_ids") or []
    return {str(value) for value in values}


def make_candidates(
    data_root: str | Path,
    train_patients: set[str],
    validation_patients: set[str],
    test_patients: set[str],
    fixed_case_ids: set[str],
) -> tuple[dict[str, list[Candidate]], dict[str, Any]]:
    if train_patients & validation_patients:
        raise RuntimeError("train and validation patient overlap is nonzero")
    if train_patients & test_patients:
        raise RuntimeError("train and test patient overlap is nonzero")
    if validation_patients & test_patients:
        raise RuntimeError("validation and test patient overlap is nonzero")
    source_patients = train_patients - validation_patients - test_patients
    pools = {name: [] for name in CLASS_VALUES}
    skipped_fixed: list[str] = []
    skipped_non_source: Counter[str] = Counter()
    for image_path, mask_path in iter_train_pairs(data_root):
        image_id = image_path.stem
        patient_id = patient_id_from_stem(image_id)
        if patient_id not in source_patients:
            skipped_non_source[patient_id] += 1
            continue
        if image_id in fixed_case_ids:
            skipped_fixed.append(image_id)
            continue
        mask = read_class_index_mask(mask_path)
        counts = {name: int((mask == value).sum()) for name, value in (("LM", 1), ("FC", 2), ("LC", 3), ("VV", 4))}
        candidate = Candidate(
            image_path=str(image_path.resolve()),
            mask_path=str(mask_path.resolve()),
            image_id=image_id,
            patient_id=patient_id,
            lm_pixels=counts["LM"],
            fc_pixels=counts["FC"],
            lc_pixels=counts["LC"],
            vv_pixels=counts["VV"],
        )
        for name in CLASS_VALUES:
            if counts[name] > 0:
                pools[name].append(candidate)
    audit = {
        "source_patients": sorted(source_patients),
        "source_patient_count": len(source_patients),
        "train_patients": sorted(train_patients),
        "validation_patients": sorted(validation_patients),
        "test_patients": sorted(test_patients),
        "excluded_validation_patients": sorted(validation_patients),
        "excluded_test_patients": sorted(test_patients),
        "fixed_case_ids_excluded": sorted(fixed_case_ids),
        "fixed_case_count_excluded": len(fixed_case_ids),
        "fixed_case_patients": sorted({patient_id_from_stem(value) for value in fixed_case_ids}),
        "fixed_cases_found_in_train": sorted(set(skipped_fixed)),
        "train_validation_overlap": sorted(train_patients & validation_patients),
        "train_test_overlap": sorted(train_patients & test_patients),
        "validation_test_overlap": sorted(validation_patients & test_patients),
        "source_validation_overlap": sorted(source_patients & validation_patients),
        "source_test_overlap": sorted(source_patients & test_patients),
        "candidate_counts": {name: len(values) for name, values in pools.items()},
        "candidate_unique_source_counts": {name: len({item.image_id for item in values}) for name, values in pools.items()},
        "fold1_test_used": False,
        "validation_source_used": False,
        "test_image_contents_read": False,
        "test_mask_contents_read": False,
    }
    if any(audit[key] for key in ("train_validation_overlap", "train_test_overlap", "validation_test_overlap", "source_validation_overlap", "source_test_overlap")):
        raise RuntimeError(f"Patient leakage audit failed: {audit}")
    return pools, audit


def stable_sample_seed(base_seed: int, target_class: str, image_id: str, generation_index: int, cycle: int) -> int:
    text = f"{base_seed}|{target_class}|{image_id}|{generation_index}|{cycle}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(text).digest()[:8], "big")
    return int(value % 2_000_000_000) + 1


def make_plan(candidates: list[Candidate], target_class: str, count: int, max_multiplier: int, seed: int) -> list[tuple[Candidate, int, int]]:
    if not candidates:
        raise RuntimeError(f"No candidates for {target_class}")
    ordered = list(candidates)
    random.Random(seed + CLASS_VALUES[target_class] * 100003).shuffle(ordered)
    plan: list[tuple[Candidate, int, int]] = []
    for generation_index in range(count * max_multiplier):
        cycle = generation_index // len(ordered)
        candidate = ordered[generation_index % len(ordered)]
        sample_seed = stable_sample_seed(seed, target_class, candidate.image_id, generation_index, cycle)
        plan.append((candidate, sample_seed, cycle))
    return plan


def pad_image(image: np.ndarray) -> np.ndarray:
    result = np.pad(image, ((PAD, PAD), (PAD, PAD)), mode="reflect")
    if tuple(result.shape) != (PADDED_SIZE, PADDED_SIZE):
        raise RuntimeError(f"Bad image pad shape: {result.shape}")
    return result


def pad_mask(mask: np.ndarray) -> np.ndarray:
    result = np.pad(mask, ((PAD, PAD), (PAD, PAD)), mode="constant", constant_values=0)
    if tuple(result.shape) != (PADDED_SIZE, PADDED_SIZE):
        raise RuntimeError(f"Bad mask pad shape: {result.shape}")
    return result


def image_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0) * 2.0 - 1.0


def mask_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)


def crop_768(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[..., PAD:-PAD, PAD:-PAD]


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().squeeze().numpy()
    # Round the inverse of image_tensor so inactive source pixels retain their
    # exact uint8 value after the 750x750 save; do not restore pixels post hoc.
    return np.rint(np.clip((array + 1.0) * 127.5, 0, 255)).astype(np.uint8)


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def model_audit(checkpoint: Path) -> dict[str, Any]:
    from lesionstatediff.region_time_b3_checkpoint import reload_region_time_b3_checkpoint

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_name") != "OCT-Hard-Region-Time-B3-Spatial-Class-Semantic-Conditioned-Diffusion":
        raise RuntimeError(f"Unexpected checkpoint model_name: {payload.get('model_name')}")
    if int(payload.get("b3_epoch", -1)) != 3 or int(payload.get("global_step", -1)) != 27200:
        raise RuntimeError(f"Checkpoint is not B3 epoch3: {payload.get('b3_epoch')}/{payload.get('global_step')}")
    if payload.get("fold1_test_used") is not False:
        raise RuntimeError("B3 checkpoint metadata says fold1_test_used is not false")
    if payload.get("input_channels") != 4 or payload.get("output_channels") != 1:
        raise RuntimeError("B3 checkpoint channel contract mismatch")
    model, _, reload_meta = reload_region_time_b3_checkpoint(checkpoint, map_location="cpu")
    if any(reload_meta.get(key) for key in ("base_missing_keys", "base_unexpected_keys", "semantic_missing_keys", "semantic_unexpected_keys")):
        raise RuntimeError(f"Strict B3 reload reported key differences: {reload_meta}")
    conv_keys = [key for key in payload["model_state_dict"] if key.endswith("conv_in.weight")]
    if len(conv_keys) != 1 or tuple(payload["model_state_dict"][conv_keys[0]].shape) != (128, 4, 3, 3):
        raise RuntimeError("B3 conv_in contract mismatch")
    semantic_keys = sorted(payload.get("semantic_state_dict", {}).keys())
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "model_name": payload.get("model_name"),
        "epoch": int(payload.get("epoch")),
        "b3_epoch": int(payload.get("b3_epoch")),
        "global_step": int(payload.get("global_step")),
        "input_channels": payload.get("input_channels"),
        "output_channels": payload.get("output_channels"),
        "unet_name": payload.get("unet_name"),
        "conv_in_key": conv_keys[0],
        "conv_in_shape": list(payload["model_state_dict"][conv_keys[0]].shape),
        "semantic_state_keys": semantic_keys,
        "semantic_state_count": len(semantic_keys),
        "strict_reload": True,
        "reload_meta": reload_meta,
        "source_reinjection_used": payload.get("source_reinjection_used"),
        "soft_blend_used": payload.get("soft_blend_used"),
        "posthoc_mask_restore_used": payload.get("posthoc_mask_restore_used"),
        "tau_rule": payload.get("tau_rule"),
        "fold1_test_used": False,
    }
    del model
    return result


def load_model_for_gpu(checkpoint: Path, device: torch.device):
    from lesionstatediff.region_time_b3_checkpoint import reload_region_time_b3_checkpoint

    model, payload, reload_meta = reload_region_time_b3_checkpoint(checkpoint, map_location="cpu")
    if any(reload_meta.get(key) for key in ("base_missing_keys", "base_unexpected_keys", "semantic_missing_keys", "semantic_unexpected_keys")):
        raise RuntimeError(f"GPU load strict reload failed: {reload_meta}")
    model.to(device).eval()
    return model, payload


def build_batch(items: list[tuple[Candidate, int, int]]):
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    raw_images: list[np.ndarray] = []
    raw_masks: list[np.ndarray] = []
    valid: list[tuple[Candidate, int, int]] = []
    errors: list[dict[str, Any]] = []
    for candidate, sample_seed, cycle in items:
        try:
            raw_image = read_grayscale(candidate.image_path)
            raw_mask = read_class_index_mask(candidate.mask_path)
            images.append(image_tensor(pad_image(raw_image)))
            masks.append(mask_tensor(pad_mask(raw_mask)))
            raw_images.append(raw_image)
            raw_masks.append(raw_mask)
            valid.append((candidate, sample_seed, cycle))
        except Exception as exc:  # technical candidate failure is recorded, not hidden
            errors.append({"source_basename": candidate.image_id, "reason": f"input_error:{type(exc).__name__}:{exc}", "sample_seed": sample_seed})
    if not valid:
        return None, errors
    return {
        "images": torch.stack(images),
        "masks": torch.stack(masks),
        "raw_images": raw_images,
        "raw_masks": raw_masks,
        "valid": valid,
    }, errors


def run_smoke(model, candidate_items: list[tuple[Candidate, int, int]], device: torch.device) -> dict[str, Any]:
    from lesionstatediff.region_time_b3_semantic import sample_region_time_ddim_b3

    batch, input_errors = build_batch(candidate_items)
    if batch is None:
        raise RuntimeError(f"Smoke input construction failed: {input_errors}")
    source = batch["images"].to(device)
    segmentation = batch["masks"].to(device)
    noises = []
    for index, (_, sample_seed, _) in enumerate(batch["valid"]):
        generator = torch.Generator(device=device).manual_seed(int(sample_seed))
        noises.append(torch.randn(source[index:index + 1].shape, device=device, generator=generator, dtype=source.dtype))
    with torch.inference_mode():
        result = sample_region_time_ddim_b3(model, DDIMScheduler(num_train_timesteps=1000), source, segmentation, torch.cat(noises), DDIM_STEPS)
    generated = result["generated"]
    if tuple(generated.shape[-2:]) != (PADDED_SIZE, PADDED_SIZE):
        raise RuntimeError(f"Smoke sampler output mismatch: {tuple(generated.shape)}")
    if not torch.isfinite(generated).all().item():
        raise RuntimeError("Smoke generated output has NaN/Inf")
    return {
        "passed": True,
        "batch_size": len(batch["valid"]),
        "model_input_shape": [len(batch["valid"]), 4, PADDED_SIZE, PADDED_SIZE],
        "source_shape": list(source.shape),
        "segmentation_shape": list(segmentation.shape),
        "generated_padded_shape": list(result["generated"].shape),
        "generated_native_shape": [len(batch["valid"]), 1, ORIGINAL_SIZE, ORIGINAL_SIZE],
        "ddim_steps": DDIM_STEPS,
        "input_errors": input_errors,
        "fold1_test_used": False,
    }


def prepare_dirs(out: Path) -> None:
    for name in ("images/FC", "images/LC", "images/VV", "masks/FC", "masks/LC", "masks/VV", "qc", "audits", "configs", "reports", "manifests"):
        (out / name).mkdir(parents=True, exist_ok=True)


def write_code_audit(out: Path, args: argparse.Namespace, checkpoint_audit: dict[str, Any], split_audit: dict[str, Any]) -> None:
    text = f"""# B3 Formal3000 Code and Protocol Audit

- task: `{TASK}`
- audit_time: `{now_iso()}`
- source script: `{Path(__file__).resolve()}`
- checkpoint loader: `lesionstatediff.region_time_b3_checkpoint.reload_region_time_b3_checkpoint`
- sampler: `lesionstatediff.region_time_b3_semantic.sample_region_time_ddim_b3`
- dataset source: `{Path(args.data_root).resolve() / 'train' / 'img'}` plus matching `{Path(args.data_root).resolve() / 'train' / 'mask'}`
- split manifest: train=`{args.train_patients}`, validation=`{args.validation_patients}`; test patients are enumerated by filename only from `{Path(args.data_root).resolve() / 'test' / 'img'}`
- mask format: original class-index mask with values `0/1/2/3/4`
- native resolution: `750x750`
- preprocessing: image reflect pad `750->768` with `9` pixels per side; mask constant-zero pad `750->768`
- output resolution: crop model output `768->750`; no resize to 256 or 768 for saved files
- DDIM steps: `{DDIM_STEPS}`
- seed logic: base seed `{BASE_SEED}` plus deterministic SHA256 sample-level seed
- B3 Region-Time: `Background=0, LM=0, FC=t, LC=t, VV=t`
- semantic condition: 5-class pixel embedding, 32 dimensions, zero-initialized 1x1 projection to 512 channels before `mid_block`
- input tensor contract: `[x_tau, hard_masked_oct, segmentation, tau_normalized]`, 4 channels
- output tensor contract: 1-channel generated OCT
- target classes: `FC`, `LC`, `VV`; LM is not a generation target
- source rule: only fold1 train patients, excluding fixed-100 cases and validation/test patients
- post-hoc image restoration: `false`; identity outside the B3 lesion mask is checked, not silently rewritten
- fold1_test_used: `false`
- validation_source_used: `false`

## Checkpoint Audit

```json
{json.dumps(checkpoint_audit, ensure_ascii=False, indent=2)}
```

## Source Patient Audit

```json
{json.dumps(split_audit, ensure_ascii=False, indent=2)}
```

This generation run reuses the B3 protocol and changes only the formal pool target selection and output bookkeeping; it does not change the generator, architecture, tau rule, sampler equation, or loss.
"""
    (out / "CODE_AUDIT.md").write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def duplicate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["source_image_id"]) for row in rows)
    repeated = {key: value for key, value in counts.items() if value > 1}
    excess = sum(value - 1 for value in repeated.values())
    return {
        "unique_source_basename": len(counts),
        "duplicate_source_basename_count": len(repeated),
        "duplicate_excess_count": excess,
        "duplicate_ratio": excess / max(1, len(rows)),
        "max_repeat_per_source": max(counts.values()) if counts else 0,
    }


def qc_generated(
    generated: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    target_class: str,
    candidate: Candidate,
    sample_seed: int,
    cycle: int,
    checkpoint: Path,
    checkpoint_sha: str,
    image_path: Path,
    mask_path: Path,
) -> tuple[dict[str, Any], str | None]:
    values = set(int(value) for value in np.unique(mask))
    target_pixels = int((mask == CLASS_VALUES[target_class]).sum())
    lesion = np.isin(mask, [2, 3, 4])
    background = mask == 0
    lm = mask == 1
    diff = np.abs(generated.astype(np.int16) - original.astype(np.int16))
    black = bool(generated.max() <= 3)
    white = bool(generated.min() >= 252)
    near_constant = bool(float(generated.std()) < 2.0)
    finite = bool(np.isfinite(generated).all())
    outside_max = int(diff[~lesion].max()) if (~lesion).any() else 0
    lm_max = int(diff[lm].max()) if lm.any() else 0
    target_diff = float(diff[mask == CLASS_VALUES[target_class]].mean()) if target_pixels else 0.0
    row = {
        "synthetic_id": image_path.stem,
        "target_class": target_class,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "source_image_id": candidate.image_id,
        "source_patient_id": candidate.patient_id,
        "source_split": "fold1_train",
        "source_image_path": candidate.image_path,
        "source_mask_path": candidate.mask_path,
        "sample_seed": sample_seed,
        "generation_index": None,
        "repeat_cycle": cycle,
        "generator_name": "OCT-Ours-B3-Spatial-Class-Semantic-Conditioned-Diffusion",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "native_resolution": "750x750",
        "ddim_steps": DDIM_STEPS,
        "generation_timestamp": now_iso(),
        "image_shape": "750x750",
        "mask_shape": "750x750",
        "mask_unique_values": ",".join(str(value) for value in sorted(values)),
        "target_pixels": target_pixels,
        "FC_pixels": int((mask == 2).sum()),
        "LC_pixels": int((mask == 3).sum()),
        "VV_pixels": int((mask == 4).sum()),
        "LM_pixels": int((mask == 1).sum()),
        "image_mean": float(generated.mean()),
        "image_std": float(generated.std()),
        "image_min": int(generated.min()),
        "image_max": int(generated.max()),
        "target_core_mean_abs_change": target_diff,
        "outside_lesion_max_abs_change": outside_max,
        "background_max_abs_change": int(diff[background].max()) if background.any() else 0,
        "lm_max_abs_change": lm_max,
        "black_image": black,
        "white_image": white,
        "near_constant_image": near_constant,
        "nan_inf": not finite,
        "fold1_test_used": False,
        "validation_source_used": False,
        "QC_status": "rejected",
    }
    reasons: list[str] = []
    if generated.shape != (750, 750):
        reasons.append("image_shape_invalid")
    if mask.shape != (750, 750):
        reasons.append("mask_shape_invalid")
    if not values.issubset({0, 1, 2, 3, 4}):
        reasons.append("mask_labels_invalid")
    if target_pixels <= 0:
        reasons.append("target_core_empty")
    if not finite:
        reasons.append("nan_inf")
    if black:
        reasons.append("black_image")
    if white:
        reasons.append("white_image")
    if near_constant:
        reasons.append("near_constant_image")
    if outside_max != 0:
        reasons.append("outside_lesion_drift")
    if lm_max != 0:
        reasons.append("lm_drift")
    return row, ";".join(reasons) if reasons else None


def make_sample_contact_sheet(out: Path, rows: list[dict[str, Any]], title: str) -> None:
    selected = rows[:12]
    if not selected:
        return
    cell_w, cell_h = 260, 230
    sheet = Image.new("RGB", (cell_w * 3, cell_h * 4), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        image = Image.open(row["image_path"]).convert("L").resize((180, 180))
        mask = Image.open(row["mask_path"]).convert("L").resize((180, 180))
        source = Image.open(row["source_image_path"]).convert("L").resize((180, 180))
        x = (index % 3) * cell_w
        y = (index // 3) * cell_h
        sheet.paste(source.convert("RGB"), (x, y + 28))
        sheet.paste(image.convert("RGB"), (x + 60, y + 28))
        sheet.paste(mask.convert("RGB"), (x + 120, y + 28))
        draw.text((x + 4, y + 4), f"{title} {index:02d} {row['source_image_id']}", fill="black")
    sheet.save(out / "qc" / f"{title}_sample_overview.png")


def perform_audit(args: argparse.Namespace, out: Path, smoke: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    if "test" in {part.lower() for part in data_root.parts}:
        raise RuntimeError(f"Refusing data root containing test path: {data_root}")
    checkpoint_audit = model_audit(checkpoint)
    train_patients = read_patient_list(args.train_patients)
    validation_patients = read_patient_list(args.validation_patients)
    test_patients = test_patients_from_names(data_root)
    fixed_ids = fixed_ids_from_b3_config(args.b3_config)
    pools, split_audit = make_candidates(data_root, train_patients, validation_patients, test_patients, fixed_ids)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B3 formal generation")
    model, _ = load_model_for_gpu(checkpoint, device)
    smoke_items = []
    for target_class in CLASS_VALUES:
        smoke_items.append((pools[target_class][0], stable_sample_seed(BASE_SEED, target_class, pools[target_class][0].image_id, 0, 0), 0))
    smoke_result = run_smoke(model, smoke_items[: min(args.batch_size, len(smoke_items))], device) if smoke else {"passed": False, "not_run": True}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    split_audit["fold1_test_used"] = False
    split_audit["validation_source_used"] = False
    split_audit["source_manifest_mode"] = "fold1_train_minus_validation_minus_test_minus_fixed100_cases"
    split_audit["candidate_counts"] = {name: len(values) for name, values in pools.items()}
    split_audit["candidate_unique_source_counts"] = {name: len({item.image_id for item in values}) for name, values in pools.items()}
    write_json(out / "audits" / "checkpoint_audit.json", checkpoint_audit)
    write_json(out / "audits" / "source_patient_audit.json", split_audit)
    write_json(out / "audits" / "smoke.json", smoke_result)
    write_code_audit(out, args, checkpoint_audit, split_audit)
    config = {
        "task": TASK,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_audit["sha256"],
        "data_root": str(data_root),
        "train_patients_manifest": str(Path(args.train_patients).resolve()),
        "validation_patients_manifest": str(Path(args.validation_patients).resolve()),
        "b3_config": str(Path(args.b3_config).resolve()),
        "per_class_requested": args.per_class,
        "classes": list(CLASS_VALUES),
        "base_seed": BASE_SEED,
        "ddim_steps": DDIM_STEPS,
        "batch_size": args.batch_size,
        "max_attempt_multiplier": args.max_attempt_multiplier,
        "preprocessing": {"image": "reflect pad 9 pixels per side", "mask": "zero pad 9 pixels per side", "saved": "crop to 750x750", "resize_used": False},
        "tau_rule": {"Background": 0, "LM": 0, "FC": "t", "LC": "t", "VV": "t"},
        "input_channels": ["x_tau", "hard_masked_oct", "segmentation", "tau_normalized"],
        "fold1_test_used": False,
        "validation_source_used": False,
        "segmentation_training_started": False,
        "created_at": now_iso(),
    }
    write_json(out / "configs" / "generation_config.json", config)
    return checkpoint_audit, split_audit, pools


def generate_pool(args: argparse.Namespace, out: Path, checkpoint_audit: dict[str, Any], split_audit: dict[str, Any], pools: dict[str, list[Candidate]]) -> dict[str, Any]:
    from lesionstatediff.region_time_b3_semantic import sample_region_time_ddim_b3

    device = torch.device("cuda")
    model, _ = load_model_for_gpu(Path(args.checkpoint).resolve(), device)
    checkpoint_sha = checkpoint_audit["sha256"]
    overall_rows: list[dict[str, Any]] = []
    class_summary: dict[str, Any] = {}
    for target_class in CLASS_VALUES:
        class_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        plan = make_plan(pools[target_class], target_class, args.per_class, args.max_attempt_multiplier, BASE_SEED)
        accepted = 0
        attempts = 0
        generation_errors = 0
        started = time.time()
        for start in range(0, len(plan), args.batch_size):
            if accepted >= args.per_class:
                break
            batch_plan = plan[start : start + args.batch_size]
            batch, input_errors = build_batch(batch_plan)
            for error in input_errors:
                error.update({"target_class": target_class, "fold1_test_used": False, "validation_source_used": False})
                rejected_rows.append(error)
                generation_errors += 1
            if batch is None:
                continue
            source = batch["images"].to(device, non_blocking=True)
            segmentation = batch["masks"].to(device, non_blocking=True)
            noises = []
            for index, (_, sample_seed, _) in enumerate(batch["valid"]):
                generator = torch.Generator(device=device).manual_seed(int(sample_seed))
                noises.append(torch.randn(source[index:index + 1].shape, device=device, generator=generator, dtype=source.dtype))
            with torch.inference_mode():
                result = sample_region_time_ddim_b3(model, DDIMScheduler(num_train_timesteps=1000), source, segmentation, torch.cat(noises), DDIM_STEPS)
            generated = crop_768(result["generated"])
            for index, (candidate, sample_seed, cycle) in enumerate(batch["valid"]):
                if accepted >= args.per_class:
                    break
                attempts += 1
                generated_u8 = to_uint8(generated[index])
                original_u8 = batch["raw_images"][index]
                mask_u8 = batch["raw_masks"][index]
                synthetic_id = f"ours_{target_class}_{candidate.image_id}_g{accepted:04d}_seed{sample_seed}"
                image_path = out / "images" / target_class / f"{synthetic_id}.png"
                mask_path = out / "masks" / target_class / f"{synthetic_id}.png"
                row, reason = qc_generated(generated_u8, original_u8, mask_u8, target_class, candidate, sample_seed, cycle, Path(args.checkpoint), checkpoint_sha, image_path, mask_path)
                row["generation_index"] = accepted if reason is None else attempts - 1
                if reason is not None:
                    row["rejection_reason"] = reason
                    rejected_rows.append(row)
                    continue
                row["QC_status"] = "accepted"
                save_png(image_path, generated_u8)
                save_png(mask_path, mask_u8)
                row["generation_index"] = accepted
                class_rows.append(row)
                overall_rows.append(row)
                accepted += 1
            del source, segmentation, noises, result, generated
            if attempts and (attempts % 100 == 0 or accepted == args.per_class):
                print(json.dumps({"event": "progress", "target_class": target_class, "accepted": accepted, "attempts": attempts, "rejected": len(rejected_rows), "elapsed_sec": round(time.time() - started, 1)}, ensure_ascii=False), flush=True)
        write_csv(out / "manifests" / f"{target_class}.csv", class_rows)
        write_csv(out / "qc" / f"{target_class}_rejected.csv", rejected_rows)
        make_sample_contact_sheet(out, class_rows, target_class)
        class_summary[target_class] = {
            "requested": args.per_class,
            "candidate_count": len(pools[target_class]),
            "candidate_unique_source_count": len({item.image_id for item in pools[target_class]}),
            "generated_attempts": attempts,
            "input_or_generation_errors": generation_errors,
            "rejected": len(rejected_rows),
            "accepted": len(class_rows),
            "elapsed_sec": round(time.time() - started, 2),
            **duplicate_stats(class_rows),
        }
        if accepted != args.per_class:
            raise RuntimeError(f"{target_class} accepted {accepted}, expected {args.per_class}; max attempts exhausted")
    total_manifest = out / "manifests" / "formal3000.csv"
    write_csv(total_manifest, overall_rows)
    summary = {
        "task": TASK,
        "checkpoint_path": checkpoint_audit["checkpoint"],
        "checkpoint_sha256": checkpoint_sha,
        "generator_name": "OCT-Ours-B3-Spatial-Class-Semantic-Conditioned-Diffusion",
        "seed": BASE_SEED,
        "ddim_steps": DDIM_STEPS,
        "classes": class_summary,
        "final_total": len(overall_rows),
        "total_manifest": str(total_manifest.resolve()),
        "fold1_test_used": False,
        "validation_source_used": False,
        "segmentation_training_started": False,
        "generation_completed_at": now_iso(),
    }
    write_json(out / "reports" / "FORMAL3000_SUMMARY.json", summary)
    return summary


def make_leakage_audit(out: Path, split_audit: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_patients = {str(row["source_patient_id"]) for row in rows}
    validation = set(split_audit["validation_patients"])
    test = set(split_audit["test_patients"])
    result = {
        "source_patient_count": len(source_patients),
        "source_patients": sorted(source_patients),
        "validation_patients": sorted(validation),
        "test_patients": sorted(test),
        "source_validation_overlap": sorted(source_patients & validation),
        "source_test_overlap": sorted(source_patients & test),
        "fold1_test_used": False,
        "validation_source_used": False,
        "passed": not (source_patients & validation or source_patients & test),
    }
    write_json(out / "audits" / "leakage_audit.json", result)
    if not result["passed"]:
        raise RuntimeError(f"Final leakage audit failed: {result}")
    return result


def write_final_reports(out: Path, summary: dict[str, Any], leakage: dict[str, Any], split_audit: dict[str, Any]) -> None:
    rows = []
    for target_class, values in summary["classes"].items():
        rows.append(f"| {target_class} | {values['requested']} | {values['generated_attempts']} | {values['rejected']} | {values['accepted']} | {values['unique_source_basename']} | {values['duplicate_ratio']:.4f} |")
    audit_text = "\n".join([
        "# FORMAL3000 Technical Audit",
        "",
        f"- task: `{TASK}`",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        f"- checkpoint_sha256: `{summary['checkpoint_sha256']}`",
        "- FC/LC/VV final accepted: `1000/1000/1000`",
        "- total: `3000`",
        "- image and mask saved resolution: `750x750`",
        "- source transform: reflect pad 750->768, model operation, crop 768->750",
        "- resize used: `false`",
        "- fold1_test_used: `false`",
        "- validation_source_used: `false`",
        "- leakage_passed: `true`",
        "",
        "| class | requested | generated attempts | rejected | accepted | unique source | duplicate ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"- total manifest: `{summary['total_manifest']}`",
        f"- leakage audit: `{out / 'audits' / 'leakage_audit.json'}`",
    ])
    (out / "reports" / "FORMAL3000_AUDIT.md").write_text(audit_text + "\n", encoding="utf-8")
    report = f"""# {TASK}

## Task

- source: fold1 train only, with existing validation/test patient boundaries and the B3 fixed-100 cases excluded
- target classes: FC=1000, LC=1000, VV=1000
- formal pool definition: counts are technical-QC accepted pairs, not hand-selected visual examples
- generator training: not continued
- downstream segmentation: not started

## Generator

- model: `Conditional DDPM + Hard Region-Time Conditioning + Spatial Class Semantic Conditioning`
- checkpoint: `{summary['checkpoint_path']}`
- checkpoint SHA256: `{summary['checkpoint_sha256']}`
- B3 epoch: `3`; reported global step: `27200`
- input: `[x_tau, hard_masked_oct, segmentation, tau_normalized]`, 4 channels
- output: 1-channel OCT prediction
- tau: `Background=0, LM=0, FC=t, LC=t, VV=t`
- semantic branch: 5-class x 32-d pixel embedding, zero-initialized 1x1 projection, 512 channels before `mid_block`
- DDIM steps: `{DDIM_STEPS}`
- base seed: `{BASE_SEED}`; sample seeds are deterministic SHA256-derived integers
- original image: `750x750 -> reflect pad -> 768x768`
- class-index mask: `750x750 -> zero pad -> 768x768`
- saved synthetic files: `750x750`
- resize used: `false`

## Counts

| class | candidates | generated attempts | rejected | accepted | unique source | duplicate ratio |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## QC and Leakage

- image/mask pair existence and same stem: checked
- mask labels: subset of `{{0,1,2,3,4}}`: checked
- target class foreground: non-empty: checked
- finite image values: checked
- black/white/near-constant technical failures: rejected and recorded only when present
- outside-lesion and LM drift: checked; no post-hoc restoration was applied
- source validation overlap: `{leakage['source_validation_overlap']}`
- source test overlap: `{leakage['source_test_overlap']}`
- fold1_test_used: `false`
- validation_source_used: `false`

## Output

- output directory: `{out.resolve()}`
- total manifest: `{summary['total_manifest']}`
- checkpoint audit: `{out / 'audits' / 'checkpoint_audit.json'}`
- source patient audit: `{out / 'audits' / 'source_patient_audit.json'}`
- leakage audit: `{out / 'audits' / 'leakage_audit.json'}`
- code audit: `{out / 'CODE_AUDIT.md'}`
- formal audit: `{out / 'reports' / 'FORMAL3000_AUDIT.md'}`

## Final Gate

Synthetic pool generation: COMPLETE
Segmentation training: NOT STARTED
Fold1 test used: FALSE
"""
    (out / "reports" / "OURS_FORMAL3000_GENERATION_REPORT.md").write_text(report, encoding="utf-8")


def archive_output(out: Path, checkpoint: Path) -> dict[str, str]:
    archive = out.parent / f"{out.name}.tar"
    if archive.exists():
        raise FileExistsError(f"Archive already exists: {archive}")
    with tarfile.open(archive, mode="w") as handle:
        handle.add(out, arcname=out.name, recursive=True)
    manifest = out / "manifests" / "formal3000.csv"
    checksum = out / "TRANSFER_SHA256.txt"
    text = "\n".join([
        f"archive_path={archive.resolve()}",
        f"archive_sha256={sha256_file(archive)}",
        f"formal3000_manifest_path={manifest.resolve()}",
        f"formal3000_manifest_sha256={sha256_file(manifest)}",
        f"checkpoint_path={checkpoint.resolve()}",
        f"checkpoint_sha256={sha256_file(checkpoint)}",
    ]) + "\n"
    checksum.write_text(text, encoding="utf-8")
    return {"archive": str(archive.resolve()), "archive_sha256": sha256_file(archive), "manifest_sha256": sha256_file(manifest), "checkpoint_sha256": sha256_file(checkpoint)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "generate"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-patients", required=True)
    parser.add_argument("--validation-patients", required=True)
    parser.add_argument("--b3-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-class", type=int, default=PER_CLASS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-attempt-multiplier", type=int, default=5)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    if args.phase == "audit":
        prepare_dirs(out)
        checkpoint_audit, split_audit, _ = perform_audit(args, out, smoke=True)
        print(json.dumps({"phase": "audit", "passed": True, "checkpoint": checkpoint_audit, "candidate_counts": split_audit["candidate_counts"], "smoke": read_json(out / "audits" / "smoke.json")}, ensure_ascii=False, indent=2), flush=True)
        return
    if not (out / "audits" / "checkpoint_audit.json").exists():
        raise RuntimeError("Formal generation requires a completed audit phase")
    checkpoint_audit, split_audit, pools = perform_audit(args, out, smoke=False)
    summary = generate_pool(args, out, checkpoint_audit, split_audit, pools)
    rows = []
    for target_class in CLASS_VALUES:
        with (out / "manifests" / f"{target_class}.csv").open(newline="", encoding="utf-8") as handle:
            rows.extend(list(csv.DictReader(handle)))
    leakage = make_leakage_audit(out, split_audit, rows)
    write_final_reports(out, summary, leakage, split_audit)
    archive_info = archive_output(out, Path(args.checkpoint).resolve())
    summary["archive"] = archive_info
    write_json(out / "reports" / "FORMAL3000_SUMMARY.json", summary)
    print(json.dumps({"phase": "generate", "completed": True, "summary": summary, "archive": archive_info}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

