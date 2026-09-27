#!/usr/bin/env python3
"""Collect fixed per-fold backbone exports into seven 941-row OOF CSV files."""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("ASD", "NORMAL", "PDA", "PFO", "VSD")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--grid", type=Path, default=ROOT / "configs" / "backbone7_retraining_grid.tsv"
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=ROOT / "configs" / "splits_stage24_confirm_seed20268020",
    )
    return parser.parse_args()


def read_csv(path, delimiter=","):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def fold_map(split_root):
    result = {}
    for fold in range(5):
        payload = json.loads(
            (split_root / f"cv5_tvt_fold{fold}.json").read_text(encoding="utf-8")
        )
        for item in payload["test_files"]:
            result[Path(item).stem] = (fold, item.split("/", 1)[0])
    if len(result) != 941:
        raise RuntimeError(f"expected 941 fixed outer-test recordings, found {len(result)}")
    return result


def main():
    args = parse_args()
    methods = read_csv(args.grid, delimiter="\t")
    expected = fold_map(args.split_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "recording_id", "fold", "true_label", "pred_label", "sex01", "age_days",
        *[f"prob_{label}" for label in LABELS],
    ]

    for method in methods:
        combined = []
        for fold in range(5):
            source = args.input_root / method["method_id"] / f"fold{fold}" / "test_file_probs.csv"
            for row in read_csv(source):
                recording_id = Path(row["file_id"]).stem
                expected_fold, expected_label = expected[recording_id]
                if fold != expected_fold or row["true_label"] != expected_label:
                    raise RuntimeError(f"split mismatch: {method['method_id']} {recording_id}")
                probabilities = [float(row[f"prob_{label}"]) for label in LABELS]
                pred_label = LABELS[max(range(len(LABELS)), key=probabilities.__getitem__)]
                combined.append(
                    {
                        "recording_id": recording_id,
                        "fold": fold,
                        "true_label": row["true_label"],
                        "pred_label": pred_label,
                        "sex01": row.get("sex01", ""),
                        "age_days": row.get("age_days", ""),
                        **{
                            f"prob_{label}": row[f"prob_{label}"]
                            for label in LABELS
                        },
                    }
                )
        ids = {row["recording_id"] for row in combined}
        if len(combined) != 941 or ids != set(expected):
            raise RuntimeError(
                f"OOF coverage failure for {method['method_id']}: rows={len(combined)} ids={len(ids)}"
            )
        output = args.output_dir / method["output_filename"]
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(combined, key=lambda row: (row["fold"], row["recording_id"])))
        print(f"WROTE: method={method['method_id']} rows=941 path={output}")


if __name__ == "__main__":
    main()
