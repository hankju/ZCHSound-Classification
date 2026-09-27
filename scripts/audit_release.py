#!/usr/bin/env python3
"""Validate fixed splits, formal predictions, and the public weight policy."""

import csv
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("ASD", "NORMAL", "PDA", "PFO", "VSD")
PREDICTION_FILES = (
    "stage25_fixed_curriculum_oof_predictions.csv",
    "stage26_multimodal_oof_predictions.csv",
)
PUBLICATION_PREDICTION_FILES = (
    "convnext_tiny_oof_predictions.csv",
    "densenet121_oof_predictions.csv",
    "efficientnet_b3_oof_predictions.csv",
    "regnety_008_oof_predictions.csv",
    "resnet50_oof_predictions.csv",
    "stage25_fixed_curriculum_oof_predictions.csv",
    "stage26_multimodal_oof_predictions.csv",
    "swin_tiny_oof_predictions.csv",
    "vit_small_16_oof_predictions.csv",
)


def fixed_fold_map():
    split_root = ROOT / "configs" / "splits_stage24_confirm_seed20268020"
    result = {}
    for fold in range(5):
        payload = json.loads(
            (split_root / f"cv5_tvt_fold{fold}.json").read_text(encoding="utf-8")
        )
        for item in payload["test_files"]:
            recording_id = Path(item).stem
            if recording_id in result:
                raise RuntimeError(f"duplicate outer-test ID: {recording_id}")
            result[recording_id] = (fold, Path(item).parts[0])
    if len(result) != 941:
        raise RuntimeError(f"expected 941 outer-test IDs, found {len(result)}")
    return result


def check_predictions(path, fold_map):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["recording_id"] for row in rows]
    if len(rows) != 941 or len(set(ids)) != 941 or set(ids) != set(fold_map):
        raise RuntimeError(f"OOF coverage failure: {path}")
    for row in rows:
        fold, label = fold_map[row["recording_id"]]
        if int(row["fold"]) != fold or row["true_label"] != label:
            raise RuntimeError(f"split mismatch: {row['recording_id']}")
        probabilities = [float(row[f"prob_{name}"]) for name in LABELS]
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities):
            raise RuntimeError(f"invalid probability: {row['recording_id']}")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
            raise RuntimeError(f"probability sum failure: {row['recording_id']}")
        prediction = LABELS[max(range(len(LABELS)), key=probabilities.__getitem__)]
        if prediction != row["pred_label"]:
            raise RuntimeError(f"prediction reconstruction failure: {row['recording_id']}")


def check_weights():
    root = ROOT / "stage24_confirmation" / "seed20268020"
    checkpoints = sorted(root.glob("**/curriculum_checkpoint.pt"))
    if len(checkpoints) != 20:
        raise RuntimeError(f"expected 20 Stage25 checkpoints, found {len(checkpoints)}")
    if any("_flat" in str(path) for path in checkpoints):
        raise RuntimeError("flat-control checkpoint included unexpectedly")
    return checkpoints


def check_source_duplicates(fold_map):
    path = ROOT / "audits" / "source_dataset_duplicate_audit.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        raise RuntimeError(f"expected 6 duplicate-content pairs, found {len(rows)}")
    cross_fold = 0
    label_conflicts = 0
    for row in rows:
        first_id = Path(row["first_path"]).stem
        second_id = Path(row["second_path"]).stem
        first_fold, first_label = fold_map[first_id]
        second_fold, second_label = fold_map[second_id]
        if (
            first_fold != int(row["first_outer_test_fold"])
            or second_fold != int(row["second_outer_test_fold"])
            or first_label != row["first_label"]
            or second_label != row["second_label"]
        ):
            raise RuntimeError(f"duplicate audit split mismatch: {first_id}/{second_id}")
        is_cross_fold = first_fold != second_fold
        is_conflict = first_label != second_label
        if (row["cross_outer_fold"] == "true") != is_cross_fold:
            raise RuntimeError(f"duplicate audit cross-fold mismatch: {first_id}/{second_id}")
        if (row["label_conflict"] == "true") != is_conflict:
            raise RuntimeError(f"duplicate audit label mismatch: {first_id}/{second_id}")
        if row["formal_action"] != "retain_both_original_rows":
            raise RuntimeError(f"unexpected formal duplicate action: {first_id}/{second_id}")
        cross_fold += int(is_cross_fold)
        label_conflicts += int(is_conflict)
    if cross_fold != 4 or label_conflicts != 1:
        raise RuntimeError(
            f"duplicate audit counts changed: cross_fold={cross_fold}, "
            f"label_conflicts={label_conflicts}"
        )
    return len(rows), cross_fold, label_conflicts


def check_publication_assets():
    required = (
        "LICENSE",
        "CITATION.cff",
        "CITATIONS.bib",
        "THIRD_PARTY_NOTICES.md",
        "MODEL_CARD.md",
        "third_party/unilm_beats/LICENSE",
        "docs/DATASET.md",
        "docs/BASELINE_RETRAINING.md",
        ".github/workflows/artifact-audit.yml",
        "configs/backbone7_retraining_grid.tsv",
        "configs/dataset_audio_sha256.csv",
        "code/nostack_filelevel_ablate.py",
        "code/filelevel_mil_common.py",
        "code/collect_backbone_oof.py",
        "scripts/run_backbone_task.py",
        "scripts/verify_dataset.py",
    )
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    if missing:
        raise RuntimeError(f"missing publication assets: {missing}")

    with (ROOT / "configs" / "dataset_audio_sha256.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        dataset_rows = list(csv.DictReader(handle))
    paths = [row["relative_path"] for row in dataset_rows]
    hashes = Counter(row["sha256"] for row in dataset_rows)
    duplicate_groups = sorted(count for count in hashes.values() if count > 1)
    if len(dataset_rows) != 941 or len(set(paths)) != 941:
        raise RuntimeError("dataset hash manifest is not 941 unique paths")
    if duplicate_groups != [2] * 6:
        raise RuntimeError(f"unexpected dataset duplicate hash groups: {duplicate_groups}")

    with (ROOT / "configs" / "backbone7_retraining_grid.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        baseline_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(baseline_rows) != 7 or len({row["method_id"] for row in baseline_rows}) != 7:
        raise RuntimeError("baseline retraining grid is not seven unique methods")
    return len(dataset_rows), len(duplicate_groups), len(baseline_rows)


def main():
    fold_map = fixed_fold_map()
    dataset_hashes, duplicate_hash_groups, baseline_methods = check_publication_assets()
    duplicate_pairs, cross_fold_pairs, label_conflicts = check_source_duplicates(
        fold_map
    )
    for filename in PREDICTION_FILES:
        check_predictions(ROOT / "predictions" / filename, fold_map)
    publication_prediction_root = (
        ROOT / "statistics" / "recording_level_analyses" / "predictions"
    )
    for filename in PUBLICATION_PREDICTION_FILES:
        check_predictions(publication_prediction_root / filename, fold_map)
    checkpoints = check_weights()
    fold_audit = json.loads(
        (ROOT / "audits" / "fold_local_protocol_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if fold_audit.get("status") != "passed":
        raise RuntimeError("fold-local protocol audit did not pass")
    largest = max(path.stat().st_size for path in ROOT.rglob("*") if path.is_file())
    if largest >= 100 * 1024 * 1024:
        raise RuntimeError(f"GitHub object limit exceeded: {largest} bytes")
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoints)
    print(
        "PASSED: splits=941 formal_predictions=2x941 publication_predictions=9x941 "
        f"duplicate_pairs={duplicate_pairs} cross_fold_pairs={cross_fold_pairs} "
        f"label_conflicts={label_conflicts} "
        f"dataset_hashes={dataset_hashes} duplicate_hash_groups={duplicate_hash_groups} "
        f"baseline_methods={baseline_methods} "
        f"checkpoint_MiB={checkpoint_bytes / 1048576:.2f} "
        f"largest_file_MiB={largest / 1048576:.2f}"
    )


if __name__ == "__main__":
    main()
