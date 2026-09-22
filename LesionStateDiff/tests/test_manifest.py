import json

import pytest

from lesionstatediff.unified_four_class_dataset import load_unified_records_json


def record(*, fold1_test_used: bool = False) -> dict:
    return {
        "image_path": "/data/train/img/case.png",
        "mask_path": "/data/train/mask/case.png",
        "target_role": "FCLC",
        "target_class": "FCLC",
        "pool_name": "FCLC_ANCHOR",
        "image_id": "case",
        "patient_id": "patient",
        "fc_pixels": 1,
        "lc_pixels": 0,
        "vv_pixels": 0,
        "lm_pixels": 10,
        "target_pixels": 1,
        "lesion_area_bin": "small",
        "is_coexist": False,
        "is_small": True,
        "is_boundary_complex": False,
        "has_fc": True,
        "has_lc": False,
        "has_vv": False,
        "has_lm": True,
        "fold1_test_used": fold1_test_used,
    }


def test_manifest_loads_training_records(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([record()]), encoding="utf-8")
    records = load_unified_records_json(path)
    assert len(records) == 1
    assert records[0].patient_id == "patient"


def test_manifest_rejects_test_records(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([record(fold1_test_used=True)]), encoding="utf-8")
    with pytest.raises(ValueError, match="held-out test"):
        load_unified_records_json(path)
