#!/usr/bin/env python3
"""Audit fold-local Stage25/Stage26 selection and OOF coverage from release files."""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "ast_curriculum_balanced": ("curriculum_summary.json", 0),
    "beats_curriculum_conservative": ("curriculum_summary.json", 100000),
    "ast_flat": ("lora_summary.json", 0),
    "beats_flat": ("beats_summary.json", 100000),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "audits" / "fold_local_protocol_audit.json"),
    )
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_ids(path, field):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row[field] for row in csv.DictReader(handle)]


def audit_splits():
    split_root = ROOT / "configs" / "splits_stage24_confirm_seed20268020"
    folds = {}
    all_test = []
    for fold in range(5):
        payload = read_json(split_root / f"cv5_tvt_fold{fold}.json")
        groups = {
            name: list(payload[f"{name}_files"])
            for name in ("train", "val", "test")
        }
        sets = {name: set(values) for name, values in groups.items()}
        patient_sets = {
            name: {Path(value).stem for value in values}
            for name, values in groups.items()
        }
        disjoint = all(
            not sets[left] & sets[right]
            and not patient_sets[left] & patient_sets[right]
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        )
        if not disjoint:
            raise RuntimeError(f"fold {fold} train/validation/test overlap")
        folds[fold] = groups
        all_test.extend(groups["test"])
    if len(all_test) != 941 or len(set(all_test)) != 941:
        raise RuntimeError("OOF split coverage is not exactly 941 unique recordings")
    return folds


def audit_component_exports(folds):
    root = ROOT / "stage24_confirmation" / "seed20268020"
    checked = []
    for family, (summary_name, encoder_offset) in FAMILIES.items():
        component = f"{family}_seed2_source"
        for replica in (1, 2):
            for fold in range(5):
                current = (
                    root
                    / family
                    / f"replica{replica}"
                    / f"fold{fold}"
                    / "components"
                    / component
                )
                summary = read_json(current / summary_name)
                validation_ids = csv_ids(current / "val_file_probs.csv", "file_id")
                test_ids = csv_ids(current / "test_file_probs.csv", "file_id")
                expected_seed = 20332100 + encoder_offset + replica * 1000 + fold
                checks = {
                    "patient_overlap_zero": summary.get("patient_overlap") == 0,
                    "test_metrics_blinded": summary.get("test_metrics_computed") is False,
                    "selection_scope_same_fold_validation": summary.get(
                        "hyperparameter_selection_data"
                    )
                    == "same-fold validation patients only",
                    "seed_matches_protocol": int(summary.get("seed", -1)) == expected_seed,
                    "validation_ids_match_split": set(validation_ids)
                    == set(folds[fold]["val"]),
                    "test_ids_match_split": set(test_ids) == set(folds[fold]["test"]),
                }
                if not all(checks.values()):
                    raise RuntimeError(
                        f"component protocol failure family={family} replica={replica} "
                        f"fold={fold}: {checks}"
                    )
                checked.append(
                    {
                        "family": family,
                        "replica": replica,
                        "fold": fold,
                        "seed": expected_seed,
                        "validation_n": len(validation_ids),
                        "test_n": len(test_ids),
                        "checks": checks,
                    }
                )
    return checked


def audit_lock(lock_path, summary_path, summary_hash_field):
    lock = read_json(lock_path)
    summary = read_json(summary_path)
    checks = {
        "selection_only": lock.get("selection_only") is True,
        "test_used_for_selection_false": lock.get("test_used_for_selection") is False,
        "outer_test_probabilities_not_loaded": lock.get(
            "outer_test_probabilities_loaded"
        )
        is False,
        "selection_root_contains_no_test": lock.get(
            "selection_root_contains_test_probabilities"
        )
        is False,
        "summary_test_used_for_selection_false": summary.get(
            "test_used_for_selection"
        )
        is False,
        "lock_sha_matches_evaluation": summary.get(summary_hash_field)
        == lock.get("selection_sha256"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"lock/evaluation audit failure: {lock_path}: {checks}")
    return {
        "selection_sha256": lock["selection_sha256"],
        "fold_decisions": len(lock.get("decisions", lock.get("folds", []))),
        "checks": checks,
    }


def audit_oof_predictions(folds):
    expected = {
        Path(item).stem: (fold, Path(item).parts[0])
        for fold, groups in folds.items()
        for item in groups["test"]
    }
    results = {}
    for name in (
        "stage25_fixed_curriculum_oof_predictions.csv",
        "stage26_multimodal_oof_predictions.csv",
    ):
        path = ROOT / "predictions" / name
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        ids = [row["recording_id"] for row in rows]
        aligned = all(
            row["recording_id"] in expected
            and int(row["fold"]) == expected[row["recording_id"]][0]
            and row["true_label"] == expected[row["recording_id"]][1]
            for row in rows
        )
        checks = {
            "rows_941": len(rows) == 941,
            "recording_ids_unique": len(set(ids)) == 941,
            "matches_fixed_fold_and_label": aligned,
        }
        if not all(checks.values()):
            raise RuntimeError(f"OOF audit failure: {name}: {checks}")
        results[name] = checks
    return results


def main():
    args = parse_args()
    folds = audit_splits()
    components = audit_component_exports(folds)
    stage25 = audit_lock(
        ROOT / "results" / "stage25" / "selection_lock.json",
        ROOT / "results" / "stage25" / "stage24_conditional_fusion_summary.json",
        "selection_sha256",
    )
    stage26 = audit_lock(
        ROOT / "results" / "stage26" / "selection_lock.json",
        ROOT / "results" / "stage26" / "stage26_multimodal_locked_summary.json",
        "stage26_selection_sha256",
    )
    payload = {
        "status": "passed",
        "dataset_variant": "original_public_941_recordings",
        "split_seed": 20268020,
        "same_fold_train_validation_test_patient_overlap": 0,
        "outer_test_recordings_once": 941,
        "component_runs_checked": len(components),
        "component_runs": components,
        "stage25_lock": stage25,
        "stage26_lock": stage26,
        "oof_predictions": audit_oof_predictions(folds),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "PASSED: folds=5 overlap=0 component_runs=40 "
        "OOF=2x941 lock_hashes=2"
    )


if __name__ == "__main__":
    main()
