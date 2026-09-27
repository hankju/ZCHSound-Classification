#!/usr/bin/env python
"""Fit strict outer-fold fusion from newly trained inner OOF component predictions."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import classification_report

from nested_crossfit_sparse_stacking import load_fold_data, patient_id, score_probs
from optimize_legacy_fusion_cv import load_component_names, load_prob_csv
from strict_samefold_sparse_stacking import fit_locked


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-output-root", required=True)
    parser.add_argument("--outer-prob-root", required=True)
    parser.add_argument("--component-grid", required=True)
    parser.add_argument("--outer-split-dir", required=True)
    parser.add_argument("--inner-split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--fusion-cv-folds", type=int, default=5)
    parser.add_argument("--fusion-cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--tag", default="nested5_strict_fusion")
    parser.add_argument("--reference-summary")
    return parser.parse_args()


def load_inner_holdout(inner_root, component_names, outer_fold, inner_fold, label_names=None):
    ids = None
    labels = None
    names = label_names
    component_probs = []
    for component in component_names:
        path = (
            Path(inner_root)
            / f"outer{outer_fold}"
            / f"inner{inner_fold}"
            / "components"
            / component
            / "test_file_probs.csv"
        )
        rows, current_names = load_prob_csv(str(path))
        current_ids = [row["file_id"] for row in rows]
        current_labels = [row["true_label"] for row in rows]
        if names is None:
            names = current_names
        elif names != current_names:
            raise RuntimeError(f"label order mismatch: {path}")
        if ids is None:
            ids = current_ids
            labels = current_labels
        elif ids != current_ids or labels != current_labels:
            raise RuntimeError(f"inner component row mismatch: {path}")
        component_probs.append(np.stack([row["probs"] for row in rows]))
    label_to_idx = {name: idx for idx, name in enumerate(names)}
    return {
        "file_ids": ids,
        "true_labels": labels,
        "y": np.asarray([label_to_idx[item] for item in labels], dtype=np.int64),
        "probs": np.stack(component_probs, axis=1),
        "label_names": names,
    }


def audit_probability_artifact(path, expected_split, expected_files, expected_tag_fragment):
    """Validate artifact identity before its probabilities enter the fusion pipeline."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        prob_fields = [field for field in fields if field.startswith("prob_")]
        required = {"split", "model_tag", "file_id", "true_label"}
        if not required.issubset(fields) or not prob_fields:
            raise RuntimeError(f"invalid probability artifact schema: {path}")
        rows = list(reader)

    ids = [row["file_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_files):
        raise RuntimeError(f"probability artifact file mismatch: {path}")
    tags = {row["model_tag"] for row in rows}
    if len(tags) != 1 or expected_tag_fragment not in next(iter(tags)):
        raise RuntimeError(f"probability artifact provenance mismatch: {path}")
    for row in rows:
        if row["split"] != expected_split:
            raise RuntimeError(f"probability artifact split mismatch: {path}")
        if row["true_label"] != Path(row["file_id"]).parts[0]:
            raise RuntimeError(f"probability artifact label mismatch: {path}")
        probs = np.asarray([float(row[field]) for field in prob_fields], dtype=np.float64)
        if (
            not np.all(np.isfinite(probs))
            or np.any(probs < 0.0)
            or not np.isclose(probs.sum(), 1.0, atol=2e-5)
        ):
            raise RuntimeError(f"invalid probability row: {path}")
    return tags


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    component_names = load_component_names(args.component_grid)
    all_rows = []
    fold_results = []
    audits = []
    label_names = None

    for outer_fold in range(args.outer_folds):
        outer_val, label_names = load_fold_data(
            args.outer_prob_root,
            component_names,
            outer_fold,
            "val",
            label_names,
        )
        outer_test, label_names = load_fold_data(
            args.outer_prob_root,
            component_names,
            outer_fold,
            "test",
            label_names,
        )
        with open(
            Path(args.outer_split_dir) / f"cv5_tvt_fold{outer_fold}.json",
            "r",
            encoding="utf-8",
        ) as handle:
            outer_split = json.load(handle)
        outer_file_sets = {
            name: set(outer_split[f"{name}_files"])
            for name in ("train", "val", "test")
        }
        outer_patient_sets = {
            name: {patient_id(item) for item in files}
            for name, files in outer_file_sets.items()
        }
        if (
            outer_patient_sets["train"] & outer_patient_sets["val"]
            or outer_patient_sets["train"] & outer_patient_sets["test"]
            or outer_patient_sets["val"] & outer_patient_sets["test"]
        ):
            raise RuntimeError(f"outer fold {outer_fold} has patient overlap")
        for component in component_names:
            val_path = (
                Path(args.outer_prob_root)
                / f"fold{outer_fold}"
                / "components"
                / component
                / "val_file_probs.csv"
            )
            test_path = val_path.with_name("test_file_probs.csv")
            val_tags = audit_probability_artifact(
                val_path,
                "val",
                outer_split["val_files"],
                f"fold{outer_fold}",
            )
            test_tags = audit_probability_artifact(
                test_path,
                "test",
                outer_split["test_files"],
                f"fold{outer_fold}",
            )
            if val_tags != test_tags:
                raise RuntimeError(
                    f"outer fold {outer_fold} component {component} val/test model mismatch"
                )
        outer_test_set = set(outer_split["test_files"])
        meta_ids = []
        meta_labels = []
        meta_y_parts = []
        meta_prob_parts = []
        inner_holdout_sets = []
        inner_manifests_match = True
        inner_splits_are_disjoint = True
        inner_splits_exclude_outer_test = True
        for inner_fold in range(args.inner_folds):
            inner = load_inner_holdout(
                args.inner_output_root,
                component_names,
                outer_fold,
                inner_fold,
                label_names,
            )
            with open(
                Path(args.inner_split_dir) / f"outer{outer_fold}_inner{inner_fold}.json",
                "r",
                encoding="utf-8",
            ) as handle:
                inner_split = json.load(handle)
            inner_train = set(inner_split["train_files"])
            inner_val = set(inner_split["val_files"])
            inner_test = set(inner_split["test_files"])
            inner_partitions = (inner_train, inner_val, inner_test)
            inner_patient_partitions = tuple(
                {patient_id(item) for item in partition}
                for partition in inner_partitions
            )
            if (
                inner_patient_partitions[0] & inner_patient_partitions[1]
                or inner_patient_partitions[0] & inner_patient_partitions[2]
                or inner_patient_partitions[1] & inner_patient_partitions[2]
            ):
                raise RuntimeError(
                    f"outer {outer_fold} inner {inner_fold} has patient overlap"
                )
            if set().union(*inner_partitions) != outer_file_sets["train"]:
                raise RuntimeError(
                    f"outer {outer_fold} inner {inner_fold} does not partition outer train"
                )
            for component in component_names:
                inner_path = (
                    Path(args.inner_output_root)
                    / f"outer{outer_fold}"
                    / f"inner{inner_fold}"
                    / "components"
                    / component
                    / "test_file_probs.csv"
                )
                audit_probability_artifact(
                    inner_path,
                    "test",
                    inner_split["test_files"],
                    f"outer{outer_fold}_inner{inner_fold}_{component}",
                )
            inner_manifests_match &= set(inner["file_ids"]) == inner_test
            inner_splits_are_disjoint &= not (
                inner_train & inner_val or inner_train & inner_test or inner_val & inner_test
            )
            inner_splits_exclude_outer_test &= not (
                (inner_train | inner_val | inner_test) & outer_test_set
            )
            meta_ids.extend(inner["file_ids"])
            meta_labels.extend(inner["true_labels"])
            meta_y_parts.append(inner["y"])
            meta_prob_parts.append(inner["probs"])
            inner_holdout_sets.append(set(inner["file_ids"]))
        meta_ids.extend(outer_val.file_ids)
        meta_labels.extend(outer_val.true_labels)
        meta_y_parts.append(outer_val.y)
        meta_prob_parts.append(outer_val.probs)
        meta_y = np.concatenate(meta_y_parts)
        meta_probs = np.concatenate(meta_prob_parts, axis=0)
        if len(meta_ids) != len(set(meta_ids)):
            raise RuntimeError(f"duplicate meta patient/file in outer fold {outer_fold}")
        meta_patients = [patient_id(item) for item in meta_ids]
        if len(meta_patients) != len(set(meta_patients)):
            raise RuntimeError(f"duplicate meta patient in outer fold {outer_fold}")

        expected_meta = set(outer_split["train_files"]) | set(outer_split["val_files"])
        expected_test = set(outer_split["test_files"])
        meta_matches = set(meta_ids) == expected_meta
        test_matches = set(outer_test.file_ids) == expected_test
        meta_test_overlap = {patient_id(item) for item in meta_ids} & {
            patient_id(item) for item in outer_test.file_ids
        }
        inner_coverage = set().union(*inner_holdout_sets) == set(outer_split["train_files"])
        inner_disjoint = sum(len(item) for item in inner_holdout_sets) == len(
            set().union(*inner_holdout_sets)
        )
        if (
            not meta_matches
            or not test_matches
            or meta_test_overlap
            or not inner_coverage
            or not inner_disjoint
            or not inner_manifests_match
            or not inner_splits_are_disjoint
            or not inner_splits_exclude_outer_test
        ):
            raise RuntimeError(f"outer fold {outer_fold} provenance audit failed")

        fusion_args = SimpleNamespace(
            inner_folds=args.fusion_cv_folds,
            inner_repeats=args.fusion_cv_repeats,
            seed=args.seed,
        )
        final_probs, details = fit_locked(
            meta_probs,
            meta_y,
            outer_test.probs,
            component_names,
            label_names,
            fusion_args,
            outer_fold,
        )
        test_metrics = score_probs(outer_test.y, final_probs)
        pred = final_probs.argmax(axis=1)
        details.update({
            "outer_fold": outer_fold,
            "meta_oof_count": len(meta_ids),
            "test_count": len(outer_test.y),
            "test_metrics": test_metrics,
        })
        fold_results.append(details)
        audits.append({
            "outer_fold": outer_fold,
            "inner_holdouts_are_disjoint": inner_disjoint,
            "inner_holdouts_cover_outer_train": inner_coverage,
            "inner_output_files_match_fixed_manifests": inner_manifests_match,
            "inner_train_validation_holdout_are_disjoint": inner_splits_are_disjoint,
            "all_inner_splits_exclude_outer_test": inner_splits_exclude_outer_test,
            "meta_files_equal_outer_train_plus_validation": meta_matches,
            "meta_test_patient_overlap": len(meta_test_overlap),
            "test_files_match_fixed_outer_split": test_matches,
            "cross_outer_fold_component_predictions_used": False,
        })
        for idx, file_id in enumerate(outer_test.file_ids):
            row = {
                "fold": outer_fold,
                "file_id": file_id,
                "true_label": outer_test.true_labels[idx],
                "pred_label": label_names[int(pred[idx])],
            }
            for class_idx, name in enumerate(label_names):
                row[f"prob_{name}"] = float(final_probs[idx, class_idx])
            all_rows.append(row)
        print(
            f"[FOLD] {outer_fold} meta={len(meta_ids)} test_acc={test_metrics['accuracy']:.4f} "
            f"components={details['component_count']}",
            flush=True,
        )

    y_true = np.asarray([label_names.index(row["true_label"]) for row in all_rows])
    pooled_probs = np.asarray(
        [[float(row[f"prob_{name}"]) for name in label_names] for row in all_rows]
    )
    pooled = score_probs(y_true, pooled_probs)
    pooled_pred = pooled_probs.argmax(axis=1)
    pooled["classification_report"] = classification_report(
        y_true,
        pooled_pred,
        labels=np.arange(len(label_names)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    reference = None
    if args.reference_summary:
        with open(args.reference_summary, "r", encoding="utf-8") as handle:
            ref = json.load(handle)["pooled_oof"]
        reference = {
            "accuracy": ref["accuracy"],
            "macro_f1": ref["macro_f1"],
            "nll": ref["nll"],
            "accuracy_delta": pooled["accuracy"] - ref["accuracy"],
            "macro_f1_delta": pooled["macro_f1"] - ref["macro_f1"],
        }

    prediction_path = output_dir / f"{args.tag}_oof_predictions.csv"
    fields = ["fold", "file_id", "true_label", "pred_label"] + [
        f"prob_{name}" for name in label_names
    ]
    with open(prediction_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "tag": args.tag,
        "protocol": (
            "Inner component models generate OOF probabilities for outer train patients; the existing "
            "same-fold outer model supplies clean probabilities for outer validation and test. Fusion "
            "is selected only from outer-train-plus-validation OOF features and evaluated once on the "
            "disjoint outer test fold."
        ),
        "test_used_for_selection": False,
        "cross_outer_fold_component_predictions_used": False,
        "component_names": component_names,
        "pooled_oof": pooled,
        "reference_strict_stage5": reference,
        "fold_results": fold_results,
        "provenance_audit_passed": all(
            item["inner_holdouts_are_disjoint"]
            and item["inner_holdouts_cover_outer_train"]
            and item["inner_output_files_match_fixed_manifests"]
            and item["inner_train_validation_holdout_are_disjoint"]
            and item["all_inner_splits_exclude_outer_test"]
            and item["meta_files_equal_outer_train_plus_validation"]
            and item["meta_test_patient_overlap"] == 0
            and item["test_files_match_fixed_outer_split"]
            and not item["cross_outer_fold_component_predictions_used"]
            for item in audits
        ),
        "audit": audits,
        "predictions_csv": str(prediction_path),
    }
    summary_path = output_dir / f"{args.tag}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(
        f"[POOLED] n={len(all_rows)} accuracy={pooled['accuracy']:.6f} "
        f"macro_f1={pooled['macro_f1']:.6f} nll={pooled['nll']:.6f}",
        flush=True,
    )
    print(f"[SAVED] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
