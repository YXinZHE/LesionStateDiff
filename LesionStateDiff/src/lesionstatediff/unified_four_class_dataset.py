from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

from .constants import CLASS_INDEX
from .io_utils import patient_id_from_stem, read_class_index_mask, require_no_test_path


POOL_FCLC_ANCHOR = "FCLC_ANCHOR"
POOL_VV_TARGET = "VV_TARGET"
POOL_LM_TARGET = "LM_TARGET"
POOL_ORDER = (POOL_FCLC_ANCHOR, POOL_VV_TARGET, POOL_LM_TARGET)

VIRTUAL_EPOCH_SAMPLE_COUNTS = {
    POOL_FCLC_ANCHOR: 2500,
    POOL_VV_TARGET: 1000,
    POOL_LM_TARGET: 500,
}

EXCLUDED_BACKGROUND_IMAGE_IDS = {"088_1_027", "088_1_028"}


@dataclass(frozen=True)
class UnifiedFourClassRecord:
    image_path: str
    mask_path: str
    target_role: str
    target_class: str
    pool_name: str
    image_id: str
    patient_id: str
    fc_pixels: int
    lc_pixels: int
    vv_pixels: int
    lm_pixels: int
    target_pixels: int
    lesion_area_bin: str
    is_coexist: bool
    is_small: bool
    is_boundary_complex: bool
    has_fc: bool
    has_lc: bool
    has_vv: bool
    has_lm: bool
    fold1_test_used: bool = False


def _binary_dilate_np(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    output = F.max_pool2d(tensor, 2 * radius + 1, stride=1, padding=radius)
    return output.squeeze().numpy() > 0


def _boundary_complex(mask: np.ndarray, target_values: tuple[int, ...]) -> bool:
    target = np.zeros(mask.shape, dtype=bool)
    for value in target_values:
        target |= mask == value
    if not target.any():
        return False
    lumen = mask == CLASS_INDEX["LM"]
    near_lumen = bool((_binary_dilate_np(target, 5) & lumen).any())
    ys, xs = np.where(target)
    bbox_area = max(1, (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    fill_ratio = float(target.sum()) / float(bbox_area)
    return near_lumen or fill_ratio < 0.35


def _iter_image_mask_pairs(data_root: str | Path):
    root = Path(data_root).resolve()
    require_no_test_path(root, "data_root")
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


def _compute_tertiles(values: Iterable[int]) -> tuple[float, float]:
    array = np.asarray([value for value in values if value > 0], dtype=np.float64)
    if array.size == 0:
        return (0.0, 0.0)
    return (
        float(np.quantile(array, 1.0 / 3.0)),
        float(np.quantile(array, 2.0 / 3.0)),
    )


def _area_bin(value: int, tertiles: tuple[float, float]) -> str:
    if value <= 0:
        return "none"
    if value <= tertiles[0]:
        return "small"
    if value <= tertiles[1]:
        return "medium"
    return "large"


def build_unified_records(
    data_root: str | Path,
    train_patients: set[str] | None = None,
) -> list[UnifiedFourClassRecord]:
    raw = []
    fclc_areas: list[int] = []
    vv_areas: list[int] = []
    lm_areas: list[int] = []
    for image_path, mask_path in _iter_image_mask_pairs(data_root):
        image_id = image_path.stem
        patient_id = patient_id_from_stem(image_id)
        if train_patients is not None and patient_id not in train_patients:
            continue
        mask = read_class_index_mask(mask_path)
        lm_pixels = int((mask == CLASS_INDEX["LM"]).sum())
        fc_pixels = int((mask == CLASS_INDEX["FC"]).sum())
        lc_pixels = int((mask == CLASS_INDEX["LC"]).sum())
        vv_pixels = int((mask == CLASS_INDEX["VV"]).sum())
        raw.append(
            (
                image_path,
                mask_path,
                image_id,
                patient_id,
                mask,
                lm_pixels,
                fc_pixels,
                lc_pixels,
                vv_pixels,
            )
        )
        if fc_pixels + lc_pixels > 0:
            fclc_areas.append(fc_pixels + lc_pixels)
        if vv_pixels > 0:
            vv_areas.append(vv_pixels)
        if lm_pixels > 0 and fc_pixels == 0 and lc_pixels == 0 and vv_pixels == 0:
            lm_areas.append(lm_pixels)

    fclc_tertiles = _compute_tertiles(fclc_areas)
    vv_tertiles = _compute_tertiles(vv_areas)
    lm_tertiles = _compute_tertiles(lm_areas)
    records: list[UnifiedFourClassRecord] = []
    for row in raw:
        (
            image_path,
            mask_path,
            image_id,
            patient_id,
            mask,
            lm_pixels,
            fc_pixels,
            lc_pixels,
            vv_pixels,
        ) = row
        common = {
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_id": image_id,
            "patient_id": patient_id,
            "fc_pixels": fc_pixels,
            "lc_pixels": lc_pixels,
            "vv_pixels": vv_pixels,
            "lm_pixels": lm_pixels,
            "is_coexist": fc_pixels > 0 and lc_pixels > 0,
            "has_fc": fc_pixels > 0,
            "has_lc": lc_pixels > 0,
            "has_vv": vv_pixels > 0,
            "has_lm": lm_pixels > 0,
            "fold1_test_used": False,
        }
        fclc_pixels = fc_pixels + lc_pixels
        if fclc_pixels > 0:
            bin_name = _area_bin(fclc_pixels, fclc_tertiles)
            records.append(
                UnifiedFourClassRecord(
                    target_role="FCLC",
                    target_class="FCLC",
                    pool_name=POOL_FCLC_ANCHOR,
                    target_pixels=fclc_pixels,
                    lesion_area_bin=bin_name,
                    is_small=bin_name == "small",
                    is_boundary_complex=_boundary_complex(
                        mask, (CLASS_INDEX["FC"], CLASS_INDEX["LC"])
                    ),
                    **common,
                )
            )
        if vv_pixels > 0:
            bin_name = _area_bin(vv_pixels, vv_tertiles)
            records.append(
                UnifiedFourClassRecord(
                    target_role="VV",
                    target_class="VV",
                    pool_name=POOL_VV_TARGET,
                    target_pixels=vv_pixels,
                    lesion_area_bin=bin_name,
                    is_small=bin_name == "small",
                    is_boundary_complex=_boundary_complex(mask, (CLASS_INDEX["VV"],)),
                    **common,
                )
            )
        is_lm_only = lm_pixels > 0 and fc_pixels == 0 and lc_pixels == 0 and vv_pixels == 0
        if is_lm_only and image_id not in EXCLUDED_BACKGROUND_IMAGE_IDS:
            bin_name = _area_bin(lm_pixels, lm_tertiles)
            records.append(
                UnifiedFourClassRecord(
                    target_role="LM",
                    target_class="LM",
                    pool_name=POOL_LM_TARGET,
                    target_pixels=lm_pixels,
                    lesion_area_bin=bin_name,
                    is_small=bin_name == "small",
                    is_boundary_complex=_boundary_complex(mask, (CLASS_INDEX["LM"],)),
                    **common,
                )
            )
    return records


def save_unified_records_json(
    path: str | Path,
    records: list[UnifiedFourClassRecord],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_unified_records_csv(
    path: str | Path,
    records: list[UnifiedFourClassRecord],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(UnifiedFourClassRecord.__dataclass_fields__)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def load_unified_records_json(path: str | Path) -> list[UnifiedFourClassRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = [UnifiedFourClassRecord(**row) for row in payload]
    leaked = [record.image_id for record in records if record.fold1_test_used]
    if leaked:
        raise ValueError(f"Training manifest includes held-out test records: {leaked[:5]}")
    return records


class PatientBalancedCycle:
    """Cycle over patients before reusing records from the same patient."""

    def __init__(
        self,
        records: list[UnifiedFourClassRecord],
        indices: list[int],
        seed: int,
    ) -> None:
        self.records = records
        self.rng = random.Random(seed)
        self.by_patient: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            self.by_patient[records[index].patient_id].append(index)
        self.patient_queue: deque[str] = deque()
        self.item_queues: dict[str, deque[int]] = {}
        self._reshuffle_patients()

    def _reshuffle_patients(self) -> None:
        patients = sorted(self.by_patient)
        self.rng.shuffle(patients)
        self.patient_queue = deque(patients)
        self.item_queues = {}
        for patient, values in self.by_patient.items():
            shuffled = list(values)
            self.rng.shuffle(shuffled)
            self.item_queues[patient] = deque(shuffled)

    def next(self, avoid_image_ids: set[str] | None = None) -> int:
        avoid_image_ids = avoid_image_ids or set()
        for _ in range(max(1, len(self.patient_queue))):
            if not self.patient_queue:
                self._reshuffle_patients()
            patient = self.patient_queue.popleft()
            queue = self.item_queues[patient]
            if not queue:
                shuffled = list(self.by_patient[patient])
                self.rng.shuffle(shuffled)
                queue = self.item_queues[patient] = deque(shuffled)
            index = queue.popleft()
            self.patient_queue.append(patient)
            if self.records[index].image_id not in avoid_image_ids:
                return index
        patient = self.rng.choice(sorted(self.by_patient))
        return self.rng.choice(self.by_patient[patient])


class UnifiedVirtualEpochSampler(Sampler[int]):
    """Reproduce the 4,000-sample patient-balanced virtual epoch."""

    def __init__(
        self,
        records: list[UnifiedFourClassRecord],
        steps_per_virtual_epoch: int = 1000,
        batch_size: int = 4,
        seed: int = 3,
    ) -> None:
        self.records = records
        self.steps_per_virtual_epoch = int(steps_per_virtual_epoch)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.epoch_size = self.steps_per_virtual_epoch * self.batch_size
        expected = sum(VIRTUAL_EPOCH_SAMPLE_COUNTS.values())
        if self.epoch_size != expected:
            raise ValueError(
                f"epoch_size={self.epoch_size} must equal fixed sample count {expected}"
            )
        self.by_pool = {
            pool: [index for index, record in enumerate(records) if record.pool_name == pool]
            for pool in POOL_ORDER
        }
        for pool, values in self.by_pool.items():
            if not values:
                raise ValueError(f"Missing training pool: {pool}")

    def __len__(self) -> int:
        return self.epoch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_cycles(self) -> dict[str, PatientBalancedCycle]:
        base_seed = self.seed + self.epoch * 1000003
        return {
            pool: PatientBalancedCycle(
                self.records,
                self.by_pool[pool],
                base_seed + index * 7919,
            )
            for index, pool in enumerate(POOL_ORDER)
        }

    def plan_indices(self) -> list[int]:
        rng = random.Random(self.seed + self.epoch * 104729)
        tokens: list[str] = []
        for pool in POOL_ORDER:
            tokens.extend([pool] * VIRTUAL_EPOCH_SAMPLE_COUNTS[pool])
        rng.shuffle(tokens)
        cycles = self._build_cycles()
        output: list[int] = []
        for start in range(0, len(tokens), self.batch_size):
            used_images: set[str] = set()
            for pool in tokens[start : start + self.batch_size]:
                index = cycles[pool].next(used_images)
                used_images.add(self.records[index].image_id)
                output.append(index)
        return output

    def __iter__(self):
        return iter(self.plan_indices())

    def dry_run(self, virtual_epochs: int = 10) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for epoch in range(1, virtual_epochs + 1):
            self.set_epoch(epoch)
            indices = self.plan_indices()
            pool_counts = Counter(self.records[index].pool_name for index in indices)
            role_counts = Counter(self.records[index].target_role for index in indices)
            patient_counts = Counter(self.records[index].patient_id for index in indices)
            case_counts = Counter(self.records[index].image_id for index in indices)
            rows.append(
                {
                    "virtual_epoch": epoch,
                    "optimizer_steps": self.steps_per_virtual_epoch,
                    "total_samples": len(indices),
                    "pool_counts": dict(pool_counts),
                    "target_role_counts": dict(role_counts),
                    "unique_patients": len(patient_counts),
                    "max_patient_exposure": max(patient_counts.values(), default=0),
                    "unique_cases": len(case_counts),
                    "max_case_exposure": max(case_counts.values(), default=0),
                }
            )
        return rows
