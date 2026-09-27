#!/usr/bin/env python
"""Strict same-fold sparse stacking over existing component exports."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold

from nested_crossfit_sparse_stacking import (
    aligned_predict_proba,
    build_features,
    component_order,
    fit_stacker,
    load_fold_data,
    patient_id,
    score_probs,
)
from optimize_legacy_fusion_cv import load_component_names


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--component-grid", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--inner-repeats", type=int, default=3)
    parser.add_argument("--tag", default="strict_samefold_sparse_stacking")
    parser.add_argument("--reference-summary", required=True)
    return parser.parse_args()


def normalize(probs):
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def select_config(val_probs, val_y, normal_idx, inner_folds, inner_repeats, seed):
    component_sizes = tuple(
        size for size in (1, 2, 3, 4, 5, 6, 8, 10) if size <= val_probs.shape[1]
    )
    selectors = ("greedy_mean", "top_accuracy")
    c_values = (0.01, 0.03, 0.10, 0.30, 1.00)
    class_weights = (None, "balanced")
    alphas = (0.0, 0.10, 0.25, 0.50, 0.75)
    max_components = max(component_sizes)
    sample_count, _, class_count = val_probs.shape
    splitter = RepeatedStratifiedKFold(
        n_splits=inner_folds,
        n_repeats=inner_repeats,
        random_state=seed,
    )
    prediction_sums = {}
    prediction_counts = {}
    split_accuracies = defaultdict(list)

    for train_idx, valid_idx in splitter.split(np.zeros(sample_count), val_y):
        for selector in selectors:
            order = component_order(val_probs[train_idx], val_y[train_idx], selector, max_components)
            for size in component_sizes:
                selected = order[:size]
                x_train, _ = build_features(val_probs[train_idx], selected, normal_idx)
                x_valid, base_valid = build_features(val_probs[valid_idx], selected, normal_idx)
                for c_value in c_values:
                    for class_weight in class_weights:
                        scaler, model = fit_stacker(x_train, val_y[train_idx], c_value, class_weight)
                        stack_valid = aligned_predict_proba(scaler, model, x_valid, class_count)
                        for alpha in alphas:
                            config = (selector, size, c_value, class_weight, alpha)
                            probs = normalize(alpha * stack_valid + (1.0 - alpha) * base_valid)
                            if config not in prediction_sums:
                                prediction_sums[config] = np.zeros(
                                    (sample_count, class_count), dtype=np.float64
                                )
                                prediction_counts[config] = np.zeros(sample_count, dtype=np.int16)
                            prediction_sums[config][valid_idx] += probs
                            prediction_counts[config][valid_idx] += 1
                            split_accuracies[config].append(
                                float(accuracy_score(val_y[valid_idx], probs.argmax(axis=1)))
                            )

    ranked = []
    for config, summed in prediction_sums.items():
        counts = prediction_counts[config]
        if not np.all(counts == inner_repeats):
            raise RuntimeError("incomplete repeated inner OOF predictions")
        probs = summed / counts[:, None]
        current = score_probs(val_y, probs)
        std_acc = float(np.std(split_accuracies[config]))
        size = config[1]
        key = (
            current["accuracy"]
            - 0.02 * std_acc
            + 0.001 * current["macro_f1"]
            - 0.000001 * current["nll"]
            - 0.00000001 * size
        )
        ranked.append((key, config, current, std_acc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def fit_locked(val_probs, val_y, test_probs, component_names, label_names, args, fold):
    normal_idx = label_names.index("NORMAL")
    ranked = select_config(
        val_probs,
        val_y,
        normal_idx,
        args.inner_folds,
        args.inner_repeats,
        args.seed + fold,
    )
    _, config, inner_metrics, inner_std = ranked[0]
    selector, size, c_value, class_weight, alpha = config
    order = component_order(val_probs, val_y, selector, size)
    selected = order[:size]
    x_val, _ = build_features(val_probs, selected, normal_idx)
    x_test, base_test = build_features(test_probs, selected, normal_idx)
    scaler, model = fit_stacker(x_val, val_y, c_value, class_weight)
    stack_test = aligned_predict_proba(scaler, model, x_test, len(label_names))
    final_test = normalize(alpha * stack_test + (1.0 - alpha) * base_test)
    details = {
        "selector": selector,
        "component_count": size,
        "components": [component_names[idx] for idx in selected],
        "C": c_value,
        "class_weight": class_weight,
        "stacking_blend_alpha": alpha,
        "inner_oof_accuracy": inner_metrics["accuracy"],
        "inner_oof_macro_f1": inner_metrics["macro_f1"],
        "inner_oof_nll": inner_metrics["nll"],
        "inner_split_accuracy_std": inner_std,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "model_classes": [int(item) for item in model.classes_],
        "model_intercept": model.intercept_.tolist(),
        "model_coef": model.coef_.tolist(),
        "top_inner_candidates": [
            {
                "selector": item[1][0],
                "component_count": item[1][1],
                "C": item[1][2],
                "class_weight": item[1][3],
                "stacking_blend_alpha": item[1][4],
                **item[2],
                "inner_split_accuracy_std": item[3],
            }
            for item in ranked[:20]
        ],
    }
    return final_test, details


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    component_names = load_component_names(args.component_grid)
    all_rows = []
    fold_results = []
    audits = []
    label_names = None

    for fold in range(args.folds):
        val, label_names = load_fold_data(
            args.run_root, component_names, fold, "val", label_names
        )
        test, label_names = load_fold_data(
            args.run_root, component_names, fold, "test", label_names
        )
        with open(Path(args.split_dir) / f"cv5_tvt_fold{fold}.json", "r", encoding="utf-8") as handle:
            split = json.load(handle)
        train_patients = {patient_id(item) for item in split["train_files"]}
        val_patients = {patient_id(item) for item in split["val_files"]}
        test_patients = {patient_id(item) for item in split["test_files"]}
        disjoint = not (
            train_patients & val_patients
            or train_patients & test_patients
            or val_patients & test_patients
        )
        val_match = set(val.file_ids) == set(split["val_files"])
        test_match = set(test.file_ids) == set(split["test_files"])
        if not disjoint or not val_match or not test_match:
            raise RuntimeError(f"fold {fold} split audit failed")

        print(
            f"[OUTER] fold={fold} validation={len(val.y)} test={len(test.y)} "
            f"components={len(component_names)}",
            flush=True,
        )
        final_test, details = fit_locked(
            val.probs,
            val.y,
            test.probs,
            component_names,
            label_names,
            args,
            fold,
        )
        test_metrics = score_probs(test.y, final_test)
        pred = final_test.argmax(axis=1)
        details.update({
            "fold": fold,
            "validation_count": len(val.y),
            "test_count": len(test.y),
            "test_metrics": test_metrics,
        })
        fold_results.append(details)
        audits.append({
            "fold": fold,
            "component_model_source_fold": fold,
            "train_validation_test_patient_disjoint": disjoint,
            "validation_files_match_fixed_split": val_match,
            "test_files_match_fixed_split": test_match,
            "cross_fold_component_predictions_used": False,
        })
        for idx, file_id in enumerate(test.file_ids):
            row = {
                "fold": fold,
                "file_id": file_id,
                "true_label": test.true_labels[idx],
                "pred_label": label_names[int(pred[idx])],
            }
            for class_idx, name in enumerate(label_names):
                row[f"prob_{name}"] = float(final_test[idx, class_idx])
            all_rows.append(row)
        print(
            f"[OUTER_RESULT] fold={fold} inner_acc={details['inner_oof_accuracy']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} test_macro={test_metrics['macro_f1']:.4f} "
            f"components={details['component_count']} alpha={details['stacking_blend_alpha']}",
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

    with open(args.reference_summary, "r", encoding="utf-8") as handle:
        reference = json.load(handle)["pooled_oof"]
    comparison = {
        "accuracy": reference["accuracy"],
        "macro_f1": reference["macro_f1"],
        "nll": reference["nll"],
        "accuracy_delta": pooled["accuracy"] - reference["accuracy"],
        "macro_f1_delta": pooled["macro_f1"] - reference["macro_f1"],
        "nll_delta": pooled["nll"] - reference["nll"],
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
            "Each outer fold uses only component models trained for that same fold. Sparse component "
            "selection and stacking hyperparameters are selected by repeated inner CV on that fold's "
            "validation set, fitted on the full same-fold validation set, and evaluated once on the "
            "disjoint same-fold test set."
        ),
        "test_used_for_selection": False,
        "cross_fold_component_predictions_used": False,
        "component_count_available": len(component_names),
        "inner_folds": args.inner_folds,
        "inner_repeats": args.inner_repeats,
        "pooled_oof": pooled,
        "reference_strict_stage5": comparison,
        "fold_results": fold_results,
        "patient_and_provenance_audit_passed": all(
            item["component_model_source_fold"] == item["fold"]
            and item["train_validation_test_patient_disjoint"]
            and item["validation_files_match_fixed_split"]
            and item["test_files_match_fixed_split"]
            and not item["cross_fold_component_predictions_used"]
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
