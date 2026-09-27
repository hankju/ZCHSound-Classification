#!/usr/bin/env python
"""Patient-safe nested stacking over existing component probability exports."""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from optimize_legacy_fusion_cv import load_component_names, load_prob_csv


@dataclass
class FoldData:
    file_ids: list[str]
    patient_ids: list[str]
    true_labels: list[str]
    y: np.ndarray
    probs: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--component-grid", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--tag", default="nested_crossfit_sparse_stacking")
    parser.add_argument("--reference-summary")
    return parser.parse_args()


def patient_id(file_id):
    return Path(file_id).stem


def load_fold_data(run_root, component_names, fold, split_name, label_names=None):
    ids = None
    true_labels = None
    component_probs = []
    labels = label_names
    for name in component_names:
        path = Path(run_root) / f"fold{fold}" / "components" / name / f"{split_name}_file_probs.csv"
        rows, current_labels = load_prob_csv(str(path))
        current_ids = [row["file_id"] for row in rows]
        current_true = [row["true_label"] for row in rows]
        if labels is None:
            labels = current_labels
        elif labels != current_labels:
            raise RuntimeError(f"label order mismatch: {path}")
        if ids is None:
            ids = current_ids
            true_labels = current_true
        elif ids != current_ids or true_labels != current_true:
            raise RuntimeError(f"row mismatch: {path}")
        component_probs.append(np.stack([row["probs"] for row in rows]))
    label_to_idx = {name: idx for idx, name in enumerate(labels)}
    return FoldData(
        file_ids=ids,
        patient_ids=[patient_id(item) for item in ids],
        true_labels=true_labels,
        y=np.asarray([label_to_idx[item] for item in true_labels], dtype=np.int64),
        probs=np.stack(component_probs, axis=1),
    ), labels


def aggregate_safe_meta(validation_folds, excluded_patients):
    probs_by_patient = defaultdict(list)
    label_by_patient = {}
    source_folds = defaultdict(list)
    for fold, data in validation_folds.items():
        for idx, pid in enumerate(data.patient_ids):
            if pid in excluded_patients:
                continue
            if pid in label_by_patient and label_by_patient[pid] != int(data.y[idx]):
                raise RuntimeError(f"inconsistent label for patient {pid}")
            label_by_patient[pid] = int(data.y[idx])
            probs_by_patient[pid].append(data.probs[idx])
            source_folds[pid].append(fold)
    patients = sorted(probs_by_patient)
    probs = np.stack([np.mean(probs_by_patient[pid], axis=0) for pid in patients])
    y = np.asarray([label_by_patient[pid] for pid in patients], dtype=np.int64)
    return patients, y, probs, {pid: source_folds[pid] for pid in patients}


def score_probs(y, probs):
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "nll": float(log_loss(y, probs, labels=np.arange(probs.shape[1]))),
    }


def component_order(probs, y, selector, max_components):
    component_count = probs.shape[1]
    individual = []
    for idx in range(component_count):
        metrics = score_probs(y, probs[:, idx, :])
        individual.append((metrics["accuracy"], metrics["macro_f1"], -metrics["nll"], -idx, idx))
    ranked = [item[-1] for item in sorted(individual, reverse=True)]
    if selector == "top_accuracy":
        return ranked[:max_components]
    if selector != "greedy_mean":
        raise ValueError(selector)

    selected = [ranked[0]]
    running_sum = probs[:, selected[0], :].copy()
    remaining = set(range(component_count)) - set(selected)
    while len(selected) < min(max_components, component_count):
        best = None
        for idx in remaining:
            candidate_probs = (running_sum + probs[:, idx, :]) / (len(selected) + 1)
            metrics = score_probs(y, candidate_probs)
            key = (metrics["accuracy"], metrics["macro_f1"], -metrics["nll"], -idx)
            if best is None or key > best[0]:
                best = (key, idx)
        selected.append(best[1])
        running_sum += probs[:, best[1], :]
        remaining.remove(best[1])
    return selected


def build_features(probs, selected, normal_idx):
    chosen = np.clip(probs[:, selected, :], 1e-8, 1.0)
    abnormal_idx = [idx for idx in range(chosen.shape[2]) if idx != normal_idx]
    log_ratio = np.log(chosen[:, :, abnormal_idx]) - np.log(chosen[:, :, normal_idx, None])
    mean_probs = chosen.mean(axis=1)
    std_probs = chosen.std(axis=1)
    features = np.concatenate([log_ratio.reshape(len(chosen), -1), mean_probs, std_probs], axis=1)
    return features, mean_probs


def fit_stacker(train_features, train_y, c_value, class_weight):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(train_features)
    model = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )
    model.fit(scaled, train_y)
    return scaler, model


def aligned_predict_proba(scaler, model, features, class_count):
    raw = model.predict_proba(scaler.transform(features))
    out = np.zeros((len(features), class_count), dtype=np.float64)
    for col, class_idx in enumerate(model.classes_):
        out[:, int(class_idx)] = raw[:, col]
    return out


def select_config(meta_probs, meta_y, normal_idx, inner_folds, seed):
    component_sizes = (1, 3, 5, 8, 12)
    selectors = ("greedy_mean", "top_accuracy")
    c_values = (0.03, 0.10, 0.30, 1.00)
    class_weights = (None, "balanced")
    blend_alphas = (0.0, 0.25, 0.50, 0.75, 1.0)
    max_components = max(component_sizes)

    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    predictions = defaultdict(list)
    truths = defaultdict(list)
    fold_accuracies = defaultdict(list)

    for train_idx, valid_idx in splitter.split(np.zeros(len(meta_y)), meta_y):
        for selector in selectors:
            order = component_order(meta_probs[train_idx], meta_y[train_idx], selector, max_components)
            for size in component_sizes:
                selected = order[:size]
                x_train, _ = build_features(meta_probs[train_idx], selected, normal_idx)
                x_valid, base_valid = build_features(meta_probs[valid_idx], selected, normal_idx)
                for c_value in c_values:
                    for class_weight in class_weights:
                        scaler, model = fit_stacker(x_train, meta_y[train_idx], c_value, class_weight)
                        stack_valid = aligned_predict_proba(
                            scaler, model, x_valid, meta_probs.shape[2]
                        )
                        for alpha in blend_alphas:
                            config = (selector, size, c_value, class_weight, alpha)
                            probs = alpha * stack_valid + (1.0 - alpha) * base_valid
                            probs /= probs.sum(axis=1, keepdims=True)
                            predictions[config].append(probs)
                            truths[config].append(meta_y[valid_idx])
                            fold_accuracies[config].append(
                                float(accuracy_score(meta_y[valid_idx], probs.argmax(axis=1)))
                            )

    ranked = []
    for config in predictions:
        y = np.concatenate(truths[config])
        probs = np.concatenate(predictions[config])
        metrics = score_probs(y, probs)
        std_acc = float(np.std(fold_accuracies[config]))
        selector, size, c_value, class_weight, alpha = config
        key = (
            metrics["accuracy"]
            - 0.01 * std_acc
            + 0.001 * metrics["macro_f1"]
            - 0.000001 * metrics["nll"]
            - 0.00000001 * size
        )
        ranked.append((key, config, metrics, std_acc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def fit_outer(meta_probs, meta_y, test_probs, component_names, label_names, inner_folds, seed):
    normal_idx = label_names.index("NORMAL")
    ranked = select_config(meta_probs, meta_y, normal_idx, inner_folds, seed)
    _, config, inner_metrics, inner_std = ranked[0]
    selector, size, c_value, class_weight, alpha = config
    order = component_order(meta_probs, meta_y, selector, max_components=size)
    selected = order[:size]
    x_meta, _ = build_features(meta_probs, selected, normal_idx)
    x_test, base_test = build_features(test_probs, selected, normal_idx)
    scaler, model = fit_stacker(x_meta, meta_y, c_value, class_weight)
    stack_test = aligned_predict_proba(scaler, model, x_test, len(label_names))
    final_test = alpha * stack_test + (1.0 - alpha) * base_test
    final_test /= final_test.sum(axis=1, keepdims=True)
    details = {
        "selected_by_inner_cv_only": True,
        "selector": selector,
        "component_count": size,
        "components": [component_names[idx] for idx in selected],
        "C": c_value,
        "class_weight": class_weight,
        "stacking_blend_alpha": alpha,
        "inner_oof_accuracy": inner_metrics["accuracy"],
        "inner_oof_macro_f1": inner_metrics["macro_f1"],
        "inner_oof_nll": inner_metrics["nll"],
        "inner_fold_accuracy_std": inner_std,
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
                "fold_accuracy_std": item[3],
            }
            for item in ranked[:10]
        ],
    }
    return final_test, details


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    component_names = load_component_names(args.component_grid)

    validation_folds = {}
    test_folds = {}
    label_names = None
    for fold in range(args.folds):
        validation_folds[fold], label_names = load_fold_data(
            args.run_root, component_names, fold, "val", label_names
        )
        test_folds[fold], label_names = load_fold_data(
            args.run_root, component_names, fold, "test", label_names
        )

    all_rows = []
    fold_results = []
    audits = []
    for outer_fold in range(args.folds):
        test = test_folds[outer_fold]
        excluded = set(test.patient_ids)
        split_path = Path(args.split_dir) / f"cv5_tvt_fold{outer_fold}.json"
        with open(split_path, "r", encoding="utf-8") as handle:
            split = json.load(handle)
        train_patients = {patient_id(item) for item in split["train_files"]}
        validation_patients = {patient_id(item) for item in split["val_files"]}
        expected_test_patients = {patient_id(item) for item in split["test_files"]}
        expected_test_files = set(split["test_files"])
        test_files_match = set(test.file_ids) == expected_test_files
        meta_patients, meta_y, meta_probs, sources = aggregate_safe_meta(validation_folds, excluded)
        overlap = set(meta_patients) & excluded
        split_disjoint = not (
            train_patients & validation_patients
            or train_patients & expected_test_patients
            or validation_patients & expected_test_patients
        )
        if overlap or not split_disjoint or not test_files_match:
            raise RuntimeError(f"outer fold {outer_fold} failed patient/split audit")

        print(
            f"[OUTER] fold={outer_fold} meta_patients={len(meta_patients)} "
            f"test_patients={len(excluded)} components={len(component_names)}",
            flush=True,
        )
        test_probs, details = fit_outer(
            meta_probs,
            meta_y,
            test.probs,
            component_names,
            label_names,
            args.inner_folds,
            args.seed + outer_fold,
        )
        test_metrics = score_probs(test.y, test_probs)
        pred = test_probs.argmax(axis=1)
        details["outer_fold"] = outer_fold
        details["meta_patient_count"] = len(meta_patients)
        details["meta_prediction_occurrences"] = sum(len(sources[pid]) for pid in meta_patients)
        details["test_metrics"] = test_metrics
        fold_results.append(details)
        audits.append({
            "outer_fold": outer_fold,
            "meta_test_patient_overlap": len(overlap),
            "test_patient_count": len(excluded),
            "meta_patient_count": len(meta_patients),
            "fixed_split_train_validation_test_disjoint": split_disjoint,
            "prediction_input_files_match_fixed_test_split": test_files_match,
        })
        for row_idx, file_id in enumerate(test.file_ids):
            row = {
                "fold": outer_fold,
                "file_id": file_id,
                "true_label": test.true_labels[row_idx],
                "pred_label": label_names[int(pred[row_idx])],
            }
            for class_idx, name in enumerate(label_names):
                row[f"prob_{name}"] = float(test_probs[row_idx, class_idx])
            all_rows.append(row)
        print(
            f"[OUTER_RESULT] fold={outer_fold} inner_acc={details['inner_oof_accuracy']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} test_macro={test_metrics['macro_f1']:.4f} "
            f"components={details['component_count']} alpha={details['stacking_blend_alpha']}",
            flush=True,
        )

    y_true = np.asarray([label_names.index(row["true_label"]) for row in all_rows])
    pooled_probs = np.asarray(
        [[float(row[f"prob_{name}"]) for name in label_names] for row in all_rows]
    )
    pooled_metrics = score_probs(y_true, pooled_probs)
    y_pred = pooled_probs.argmax(axis=1)
    pooled_metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
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
            "accuracy_delta": pooled_metrics["accuracy"] - ref["accuracy"],
            "macro_f1_delta": pooled_metrics["macro_f1"] - ref["macro_f1"],
        }

    prediction_path = output_dir / f"{args.tag}_oof_predictions.csv"
    fields = ["fold", "file_id", "true_label", "pred_label"] + [f"prob_{name}" for name in label_names]
    with open(prediction_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "tag": args.tag,
        "protocol": (
            "For each outer test fold, remove every outer-test patient from all validation OOF exports; "
            "average repeated cross-fitted validation predictions per remaining patient; select the "
            "component subset and regularized stacking configuration by inner stratified CV only; "
            "fit on all safe meta-training patients; evaluate the outer test fold once."
        ),
        "test_used_for_selection": False,
        "component_grid": str(Path(args.component_grid).resolve()),
        "component_count_available": len(component_names),
        "seed": args.seed,
        "inner_folds": args.inner_folds,
        "pooled_oof": pooled_metrics,
        "fold_results": fold_results,
        "patient_audit_passed": all(
            item["meta_test_patient_overlap"] == 0
            and item["fixed_split_train_validation_test_disjoint"]
            and item["prediction_input_files_match_fixed_test_split"]
            for item in audits
        ),
        "patient_audit": audits,
        "reference_stage5_foldwise": reference,
        "predictions_csv": str(prediction_path),
    }
    summary_path = output_dir / f"{args.tag}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(
        f"[POOLED] n={len(all_rows)} accuracy={pooled_metrics['accuracy']:.6f} "
        f"macro_f1={pooled_metrics['macro_f1']:.6f} nll={pooled_metrics['nll']:.6f}",
        flush=True,
    )
    print(f"[SAVED] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
