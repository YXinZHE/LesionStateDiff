from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from .io_utils import (
    image_to_tensor,
    mask_to_tensor_raw_class,
    pad_image_reflect,
    pad_mask_zero,
    read_class_index_mask,
    read_grayscale_image,
    require_no_test_path,
)
from .unified_four_class_dataset import load_unified_records_json


class RegionTimeDataset(Dataset):
    """Minimal OCT/segmentation dataset with no soft-mask construction."""

    def __init__(self, manifest_path: str | Path):
        require_no_test_path(manifest_path, "manifest_path")
        self.records = load_unified_records_json(manifest_path)
        for record in self.records:
            require_no_test_path(record.image_path, "image_path")
            require_no_test_path(record.mask_path, "mask_path")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = image_to_tensor(
            pad_image_reflect(read_grayscale_image(record.image_path))
        )
        segmentation = mask_to_tensor_raw_class(
            pad_mask_zero(read_class_index_mask(record.mask_path))
        )
        return {
            "target_image": image,
            "segmentation_map": segmentation,
            "target_role": record.target_role,
            "target_class": record.target_class,
            "pool_name": record.pool_name,
            "image_id": record.image_id,
            "patient_id": record.patient_id,
        }
