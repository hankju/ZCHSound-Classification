#!/usr/bin/env python
"""Evaluate the validation-locked Stage26 demographic correction once."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from select_nested12_multiscale_gate import verify_lock
from select_stage26_multimodal_demographic_prior import (
    LABELS,
    apply_prior,
    canonical_hash,
    fit_prior,
    load_metadata,
    metadata_arrays,
    read_acoustic,
    score,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--selection-lock", required=True)
    parser.add_argument("--acoustic-reference-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output_dir).resolve()
    summary_path = output / "stage26_multimodal_locked_summary.json"
    if summary_path.exists():
        raise RuntimeError(f"Stage26 test already evaluated: {summary_path}")
    lock = verify_lock(args.selection_lock)
    if not (
        lock.get("tag") == "stage26_train_fitted_demographic_prior_selection"
        and lock.get("selection_only") is True
        and lock.get("selection_root_contains_test_probabilities") is False
        and lock.get("outer_test_probabilities_loaded") is False
        and lock.get("test_used_for_selection") is False
    ):
        raise RuntimeError("Stage26 selection lock is not test-blind")
    metadata_path = Path(args.metadata_csv).resolve()
    if str(metadata_path) != lock["metadata_csv"]:
        raise RuntimeError("Stage26 metadata path changed after lock")
    metadata = load_metadata(metadata_path)
    split_seed = int(lock["split_seed"])
    split_dir = (
        Path(args.split_root).resolve()
        / f"splits_stage24_confirm_seed{split_seed}"
    )
    decision_map = {item["fold"]: item for item in lock["folds"]}
    rows = []
    fold_results = []
    audits = []
    for fold in range(5):
        with (split_dir / f"cv5_tvt_fold{fold}.json").open(
            "r", encoding="utf-8"
        ) as handle:
            split = json.load(handle)
        train_ids = split["train_files"]
        val_ids = split["val_files"]
        test_ids, test_y, acoustic_test = read_acoustic(
            args.test_root, split_seed, fold, "test"
        )
        if set(test_ids) != set(split["test_files"]):
            raise RuntimeError(f"Stage26 test manifest mismatch fold={fold}")
        train_set, val_set, test_set = map(
            set, (train_ids, val_ids, split["test_files"])
        )
        disjoint = not (
            train_set & val_set or train_set & test_set or val_set & test_set
        )
        if not disjoint:
            raise RuntimeError(f"Stage26 patient overlap fold={fold}")

        fit_ids = train_ids + val_ids
        fit_sex, fit_age, fit_y = metadata_arrays(fit_ids, metadata)
        test_sex, test_age, metadata_test_y = metadata_arrays(test_ids, metadata)
        if not np.array_equal(test_y, metadata_test_y):
            raise RuntimeError(f"Stage26 test metadata label mismatch fold={fold}")
        decision = decision_map[fold]
        if decision["train_file_ids_sha256"] != canonical_hash(train_ids):
            raise RuntimeError(f"Stage26 train identity changed fold={fold}")
        if decision["validation_file_ids_sha256"] != canonical_hash(val_ids):
            raise RuntimeError(f"Stage26 validation identity changed fold={fold}")
        config = decision["locked_candidate"]
        if config is None:
            final_test = acoustic_test.copy()
            effective_weight = np.zeros(len(test_y), dtype=np.float64)
        else:
            fitted = fit_prior(
                config["mode"], config["C"], fit_sex, fit_age, fit_y
            )
            final_test, effective_weight = apply_prior(
                config, fitted, acoustic_test, test_sex, test_age
            )
        acoustic_metrics = score(test_y, acoustic_test)
        multimodal_metrics = score(test_y, final_test)
        fold_results.append({
            "fold": fold,
            "locked_branch": decision["locked_branch"],
            "locked_candidate": config,
            "test_count": len(test_y),
            "acoustic_test_metrics": acoustic_metrics,
            "multimodal_test_metrics": multimodal_metrics,
            "mean_effective_weight": float(effective_weight.mean()),
        })
        audits.append({
            "fold": fold,
            "train_validation_test_patient_disjoint": disjoint,
            "metadata_fit_patient_count": len(fit_ids),
            "metadata_fit_test_patient_overlap": len(set(fit_ids) & test_set),
            "test_files_match_fixed_split": True,
            "test_used_for_selection": False,
        })
        acoustic_pred = acoustic_test.argmax(axis=1)
        multimodal_pred = final_test.argmax(axis=1)
        for index, file_id in enumerate(test_ids):
            row = {
                "fold": fold,
                "file_id": file_id,
                "true_label": LABELS[int(test_y[index])],
                "acoustic_pred_label": LABELS[int(acoustic_pred[index])],
                "multimodal_pred_label": LABELS[int(multimodal_pred[index])],
                "sex01": float(test_sex[index]),
                "age_days": float(test_age[index]),
                "locked_branch": decision["locked_branch"],
                "effective_prior_weight": float(effective_weight[index]),
            }
            for family, probabilities in (
                ("acoustic", acoustic_test),
                ("multimodal", final_test),
            ):
                for class_index, label in enumerate(LABELS):
                    row[f"{family}_prob_{label}"] = float(
                        probabilities[index, class_index]
                    )
            rows.append(row)
        print(
            f"[FOLD] fold={fold} branch={decision['locked_branch']} "
            f"acoustic_acc={acoustic_metrics['accuracy']:.4f} "
            f"multimodal_acc={multimodal_metrics['accuracy']:.4f}",
            flush=True,
        )

    y = np.asarray(
        [LABELS.index(row["true_label"]) for row in rows], dtype=np.int64
    )
    pooled_probabilities = {
        family: np.asarray([
            [row[f"{family}_prob_{label}"] for label in LABELS]
            for row in rows
        ], dtype=np.float64)
        for family in ("acoustic", "multimodal")
    }
    pooled = {
        family: score(y, probabilities)
        for family, probabilities in pooled_probabilities.items()
    }
    for family, probabilities in pooled_probabilities.items():
        pooled[family]["classification_report"] = classification_report(
            y,
            probabilities.argmax(axis=1),
            labels=np.arange(len(LABELS)),
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )
    with Path(args.acoustic_reference_summary).open(
        "r", encoding="utf-8"
    ) as handle:
        reference = json.load(handle)["mean_across_repeats"]["curriculum"]
    for metric in (
        "accuracy", "macro_f1", "minority_recall", "minority_f1",
        "normal_recall", "nll",
    ):
        if abs(pooled["acoustic"][metric] - reference[metric]) > 1e-12:
            raise RuntimeError(
                f"Stage26 acoustic baseline mismatch {metric}: "
                f"{pooled['acoustic'][metric]} != {reference[metric]}"
            )

    acoustic_pred = pooled_probabilities["acoustic"].argmax(axis=1)
    multimodal_pred = pooled_probabilities["multimodal"].argmax(axis=1)
    changed = acoustic_pred != multimodal_pred
    acoustic_correct = acoustic_pred == y
    multimodal_correct = multimodal_pred == y
    disagreement = {
        "prediction_changed_count": int(changed.sum()),
        "multimodal_only_correct_count": int(
            np.sum(changed & multimodal_correct & ~acoustic_correct)
        ),
        "acoustic_only_correct_count": int(
            np.sum(changed & acoustic_correct & ~multimodal_correct)
        ),
        "both_wrong_after_changed_count": int(
            np.sum(changed & ~acoustic_correct & ~multimodal_correct)
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "stage26_multimodal_locked_predictions.csv"
    fields = [
        "fold", "file_id", "true_label", "acoustic_pred_label",
        "multimodal_pred_label", "sex01", "age_days", "locked_branch",
        "effective_prior_weight",
    ]
    for family in ("acoustic", "multimodal"):
        fields.extend(f"{family}_prob_{label}" for label in LABELS)
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metric_names = (
        "accuracy", "macro_f1", "minority_recall", "minority_f1",
        "normal_recall", "nll",
    )
    summary = {
        "tag": "stage26_multimodal_demographic_prior_locked_evaluation",
        "stage26_selection_sha256": lock["selection_sha256"],
        "test_used_for_selection": False,
        "outer_test_evaluated_once_after_lock": True,
        "split_seed": split_seed,
        "label_names": list(LABELS),
        "acoustic_pooled": pooled["acoustic"],
        "multimodal_pooled": pooled["multimodal"],
        "delta_multimodal_minus_acoustic": {
            metric: pooled["multimodal"][metric] - pooled["acoustic"][metric]
            for metric in metric_names
        },
        "disagreement": disagreement,
        "fold_results": fold_results,
        "audits": audits,
        "all_audits_passed": all(
            item["train_validation_test_patient_disjoint"]
            and item["metadata_fit_test_patient_overlap"] == 0
            and item["test_files_match_fixed_split"]
            and item["test_used_for_selection"] is False
            for item in audits
        ),
        "prediction_path": str(prediction_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"[DONE] acoustic_acc={pooled['acoustic']['accuracy']:.6f} "
        f"acoustic_macro={pooled['acoustic']['macro_f1']:.6f} "
        f"multimodal_acc={pooled['multimodal']['accuracy']:.6f} "
        f"multimodal_macro={pooled['multimodal']['macro_f1']:.6f} "
        f"multimodal_minf1={pooled['multimodal']['minority_f1']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
