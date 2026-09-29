#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from lesionstatediff.unified_four_class_dataset import (
    build_unified_records,
    save_unified_records_csv,
    save_unified_records_json,
)


def read_patient_ids(path: str | Path) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("patients") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("Patient file must be a JSON list or contain a 'patients' list")
    return {str(value) for value in values}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the LesionStateDiff training manifest")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-patients", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    records = build_unified_records(
        args.data_root,
        train_patients=read_patient_ids(args.train_patients),
    )
    save_unified_records_json(args.output_json, records)
    save_unified_records_csv(args.output_csv, records)
    summary = {
        "records": len(records),
        "patients": len({record.patient_id for record in records}),
        "pool_counts": dict(Counter(record.pool_name for record in records)),
        "fold1_test_used": any(record.fold1_test_used for record in records),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
