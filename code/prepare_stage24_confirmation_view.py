#!/usr/bin/env python
"""Build physically separated validation/test views for Stage24 confirmation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


LABELS = ("ASD", "NORMAL", "PDA", "PFO", "VSD")
PROB_FIELDS = tuple(f"prob_{label}" for label in LABELS)
DEFAULT_SPLIT_SEED = 20267020
FAMILIES = {
    "ast_flat_seed2_mean": ("ast_flat", "ast_flat_seed2_source"),
    "beats_flat_seed2_mean": ("beats_flat", "beats_flat_seed2_source"),
    "ast_curriculum_balanced_seed2_mean": (
        "ast_curriculum_balanced",
        "ast_curriculum_balanced_seed2_source",
    ),
    "beats_curriculum_conservative_seed2_mean": (
        "beats_curriculum_conservative",
        "beats_curriculum_conservative_seed2_source",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--view-tag", default="stage24_confirmation")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    if not path.is_file():
        raise RuntimeError(f"missing Stage24 confirmation export: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty Stage24 confirmation export: {path}")
    return rows


def write_mean(sources, destination, split, model_tag):
    grouped = [read_rows(path) for path in sources]
    ids = [row["file_id"] for row in grouped[0]]
    labels = [row["true_label"] for row in grouped[0]]
    for rows in grouped[1:]:
        if [row["file_id"] for row in rows] != ids:
            raise RuntimeError(f"Stage24 seed IDs misaligned for {destination}")
        if [row["true_label"] for row in rows] != labels:
            raise RuntimeError(f"Stage24 seed labels misaligned for {destination}")
    probability = np.mean(
        [
            np.asarray(
                [[float(row[field]) for field in PROB_FIELDS] for row in rows],
                dtype=np.float64,
            )
            for rows in grouped
        ],
        axis=0,
    )
    probability = np.clip(probability, 1e-12, None)
    probability /= probability.sum(axis=1, keepdims=True)
    fields = [
        "split",
        "model_tag",
        "file_id",
        "true_label",
        "pred_label",
        "sex01",
        "age_days",
    ] + list(PROB_FIELDS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, reference in enumerate(grouped[0]):
            row = {
                "split": split,
                "model_tag": model_tag,
                "file_id": reference["file_id"],
                "true_label": reference["true_label"],
                "pred_label": LABELS[int(probability[index].argmax())],
                "sex01": reference.get("sex01", ""),
                "age_days": reference.get("age_days", ""),
            }
            for field, value in zip(PROB_FIELDS, probability[index]):
                row[field] = repr(float(value))
            writer.writerow(row)
    summary = {
        "model_tag": model_tag,
        "ensemble_method": "fixed_arithmetic_two_seed_mean",
        "split": split,
        "test_metrics_computed": False,
        "metadata_used_as_features": False,
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sources
        ],
    }
    with (destination.parent / f"{split}_mean_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    split = "val" if args.phase == "validation" else "test"
    view = (
        root / "data" / f"{args.view_tag}_selection"
        if split == "val"
        else root / "data" / f"{args.view_tag}_evaluation"
    )
    for split_seed in (args.split_seed,):
        for fold in range(5):
            for model_tag, (family, component) in FAMILIES.items():
                sources = [
                    root
                    / "stage24_confirmation"
                    / f"seed{split_seed}"
                    / family
                    / f"replica{replica}"
                    / f"fold{fold}"
                    / "components"
                    / component
                    / f"{split}_file_probs.csv"
                    for replica in (1, 2)
                ]
                destination = (
                    view
                    / f"seed{split_seed}"
                    / f"fold{fold}"
                    / "components"
                    / model_tag
                    / f"{split}_file_probs.csv"
                )
                write_mean(sources, destination, split, model_tag)
    if split == "val" and list(view.glob("**/test_file_probs.csv")):
        raise RuntimeError("Stage24 confirmation selection view contains test files")
    print(f"[DONE] Stage24 confirmation {args.phase} view={view}", flush=True)


if __name__ == "__main__":
    main()
