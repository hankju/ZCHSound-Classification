#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import classification_report


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize legacy CV fusion on validation OOF and report held-out test OOF.")
    parser.add_argument("--run_root", required=True, help="Root with fold*/components/* probability CSVs.")
    parser.add_argument("--component_grid", required=True, help="TSV grid; first column is component name.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="optimized_legacy_fusion")
    parser.add_argument("--num_trials", type=int, default=80000)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--modes", default="arith,geom")
    parser.add_argument("--temperature_range", default="0.65,2.75")
    parser.add_argument("--bias_range", default="-1.25,1.25")
    parser.add_argument("--normal_bias_range", default="-1.00,1.00")
    parser.add_argument("--power_range", default="0.55,1.75")
    parser.add_argument("--score_macro_weight", type=float, default=0.001, help="Small tie-breaker added to accuracy.")
    return parser.parse_args()


@dataclass
class DatasetBundle:
    file_ids: list[str]
    true_labels: list[str]
    y_idx: np.ndarray
    label_names: list[str]
    probs_by_component: dict[str, np.ndarray]


def parse_range(spec: str):
    lo, hi = [float(token.strip()) for token in spec.split(",", 1)]
    if hi < lo:
        raise ValueError(f"invalid range: {spec}")
    return lo, hi


def load_component_names(path: str):
    names = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            names.append(row["name"])
    if not names:
        raise RuntimeError(f"no components found in {path}")
    return names


def load_prob_csv(path: str):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        prob_cols = [name for name in (reader.fieldnames or []) if name.startswith("prob_")]
        if not prob_cols:
            raise RuntimeError(f"no probability columns in {path}")
        label_names = [name[5:] for name in prob_cols]
        rows = []
        for row in reader:
            rows.append(
                {
                    "file_id": row["file_id"],
                    "true_label": row["true_label"],
                    "probs": np.asarray([float(row[col]) for col in prob_cols], dtype=np.float64),
                }
            )
    rows.sort(key=lambda item: item["file_id"])
    return rows, label_names


def load_cv_bundle(run_root: str, component_names: Sequence[str], folds: int, split_name: str):
    all_file_ids = []
    all_true_labels = []
    label_names = None
    parts_by_component = {name: [] for name in component_names}

    for fold in range(folds):
        fold_ids = None
        fold_true = None
        for component in component_names:
            path = os.path.join(run_root, f"fold{fold}", "components", component, f"{split_name}_file_probs.csv")
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            rows, current_labels = load_prob_csv(path)
            current_ids = [row["file_id"] for row in rows]
            current_true = [row["true_label"] for row in rows]
            current_probs = np.stack([row["probs"] for row in rows], axis=0)

            if label_names is None:
                label_names = current_labels
            elif current_labels != label_names:
                raise RuntimeError(f"label mismatch: {path}")

            if fold_ids is None:
                fold_ids = current_ids
                fold_true = current_true
            else:
                if current_ids != fold_ids:
                    raise RuntimeError(f"file_id mismatch in fold {fold}: {component}")
                if current_true != fold_true:
                    raise RuntimeError(f"true_label mismatch in fold {fold}: {component}")

            parts_by_component[component].append(current_probs)

        all_file_ids.extend([f"fold{fold}/{file_id}" for file_id in (fold_ids or [])])
        all_true_labels.extend(fold_true or [])

    label_to_idx = {label: idx for idx, label in enumerate(label_names or [])}
    y_idx = np.asarray([label_to_idx[label] for label in all_true_labels], dtype=np.int64)
    probs_by_component = {name: np.concatenate(parts, axis=0) for name, parts in parts_by_component.items()}
    return DatasetBundle(all_file_ids, all_true_labels, y_idx, label_names or [], probs_by_component)


def normalize_rows(arr: np.ndarray):
    sums = arr.sum(axis=1, keepdims=True)
    fallback = np.full_like(arr, 1.0 / arr.shape[1])
    return np.divide(arr, sums, out=fallback, where=sums > 0.0)


def softmax(logits: np.ndarray):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def temp_scale_probs(probs: np.ndarray, temperatures: np.ndarray):
    out = []
    for idx, temp in enumerate(temperatures):
        out.append(softmax(np.log(np.clip(probs[idx], 1e-10, 1.0)) / max(float(temp), 1e-6)))
    return np.stack(out, axis=0)


def sigmoid(x: np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))


def logit(x: np.ndarray):
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def apply_subtype_calibration(probs: np.ndarray, label_names: Sequence[str], normal_bias: float, powers: np.ndarray):
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    normal_idx = label_to_idx["NORMAL"]
    abnormal_indices = [idx for idx, name in enumerate(label_names) if name != "NORMAL"]

    calibrated = np.zeros_like(probs)
    final_normal = sigmoid(logit(probs[:, normal_idx]) + normal_bias)
    abnormal_mass = np.clip(1.0 - final_normal, 1e-6, 1.0)
    calibrated[:, normal_idx] = 1.0 - abnormal_mass

    abnormal_cond = normalize_rows(np.clip(probs[:, abnormal_indices], 1e-10, 1.0))
    powered = abnormal_cond.copy()
    for col_idx, power in enumerate(powers):
        powered[:, col_idx] = np.power(powered[:, col_idx], power)
    powered = normalize_rows(powered)
    calibrated[:, abnormal_indices] = abnormal_mass[:, None] * powered
    return normalize_rows(calibrated)


def combine(stacked_probs: np.ndarray, label_names: Sequence[str], mode: str, weights: np.ndarray, temperatures: np.ndarray, bias: np.ndarray, normal_bias: float, powers: np.ndarray):
    scaled = temp_scale_probs(stacked_probs, temperatures)
    if mode == "arith":
        probs = np.tensordot(weights, scaled, axes=(0, 0))
        probs = normalize_rows(probs)
        logits = np.log(np.clip(probs, 1e-10, 1.0)) + bias
        probs = softmax(logits)
    elif mode == "geom":
        logits = np.tensordot(weights, np.log(np.clip(scaled, 1e-10, 1.0)), axes=(0, 0)) + bias
        probs = softmax(logits)
    else:
        raise ValueError(mode)
    return apply_subtype_calibration(probs, label_names, normal_bias, powers)


def score_probs(probs: np.ndarray, y_idx: np.ndarray):
    pred = probs.argmax(axis=1)
    acc = float(np.mean(pred == y_idx))
    f1_terms = []
    for class_idx in range(probs.shape[1]):
        pred_mask = pred == class_idx
        true_mask = y_idx == class_idx
        tp = float(np.sum(pred_mask & true_mask))
        fp = float(np.sum(pred_mask & ~true_mask))
        fn = float(np.sum(~pred_mask & true_mask))
        denom = 2.0 * tp + fp + fn
        f1_terms.append(0.0 if denom <= 0.0 else (2.0 * tp) / denom)
    macro_f1 = float(np.mean(f1_terms))
    chosen = probs[np.arange(len(y_idx)), y_idx]
    nll = float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0))))
    return acc, macro_f1, nll, pred


def scalar_key(acc: float, macro_f1: float, nll: float, macro_weight: float):
    return acc + macro_weight * macro_f1 - 1e-6 * nll


def make_solution(component_names, label_names, mode, weights, temperatures, bias, normal_bias, powers, select_metrics, eval_metrics):
    abnormal_names = [name for name in label_names if name != "NORMAL"]
    return {
        "mode": mode,
        "weights": {name: float(weights[idx]) for idx, name in enumerate(component_names)},
        "temperatures": {name: float(temperatures[idx]) for idx, name in enumerate(component_names)},
        "bias": {name: float(bias[idx]) for idx, name in enumerate(label_names)},
        "normal_bias": float(normal_bias),
        "powers": {name: float(powers[idx]) for idx, name in enumerate(abnormal_names)},
        "select_accuracy": select_metrics[0],
        "select_macro_f1": select_metrics[1],
        "select_nll": select_metrics[2],
        "eval_accuracy": eval_metrics[0],
        "eval_macro_f1": eval_metrics[1],
        "eval_nll": eval_metrics[2],
    }


def random_solution(rng, component_count: int, class_count: int, abnormal_count: int, modes, temp_range, bias_range, normal_bias_range, power_range):
    weights = rng.dirichlet(np.ones(component_count))
    temperatures = np.exp(rng.uniform(math.log(temp_range[0]), math.log(temp_range[1]), size=component_count))
    bias = rng.uniform(bias_range[0], bias_range[1], size=class_count)
    bias = bias - bias.mean()
    powers = np.exp(rng.uniform(math.log(power_range[0]), math.log(power_range[1]), size=abnormal_count))
    normal_bias = float(rng.uniform(normal_bias_range[0], normal_bias_range[1]))
    mode = str(rng.choice(modes))
    return mode, weights, temperatures, bias, normal_bias, powers


def baseline_candidates(component_names, label_names):
    n = len(component_names)
    class_count = len(label_names)
    abnormal_names = [name for name in label_names if name != "NORMAL"]
    class_to_idx = {name: idx for idx, name in enumerate(label_names)}
    candidates = []

    weights = np.ones(n, dtype=np.float64) / n
    candidates.append(("arith", weights, np.ones(n), np.zeros(class_count), 0.0, np.ones(len(abnormal_names))))

    legacy = {"legacy42": 0.34, "mel62": 0.31, "log52": 0.35}
    if all(name in component_names for name in legacy):
        weights = np.zeros(n, dtype=np.float64)
        for idx, name in enumerate(component_names):
            weights[idx] = legacy.get(name, 0.0)
        weights = weights / weights.sum()
        bias = np.zeros(class_count, dtype=np.float64)
        for label, value in {"NORMAL": -0.2, "PDA": -0.6, "VSD": 0.25}.items():
            bias[class_to_idx[label]] = value
        powers = np.ones(len(abnormal_names), dtype=np.float64)
        for idx, label in enumerate(abnormal_names):
            if label in {"PDA", "VSD"}:
                powers[idx] = 0.9
        candidates.append(("arith", weights, np.ones(n), bias, 0.0, powers))

    for idx in range(n):
        weights = np.zeros(n, dtype=np.float64)
        weights[idx] = 1.0
        candidates.append(("arith", weights, np.ones(n), np.zeros(class_count), 0.0, np.ones(len(abnormal_names))))
    return candidates


def write_predictions(path: str, bundle: DatasetBundle, pred_idx: np.ndarray, probs: np.ndarray):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = ["file_id", "true_label", "pred_label"] + [f"prob_{name}" for name in bundle.label_names]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, file_id in enumerate(bundle.file_ids):
            row = {
                "file_id": file_id,
                "true_label": bundle.true_labels[row_idx],
                "pred_label": bundle.label_names[int(pred_idx[row_idx])],
            }
            for class_idx, class_name in enumerate(bundle.label_names):
                row[f"prob_{class_name}"] = float(probs[row_idx, class_idx])
            writer.writerow(row)


def main():
    args = parse_args()
    component_names = load_component_names(args.component_grid)
    modes = [token.strip() for token in args.modes.split(",") if token.strip()]
    temp_range = parse_range(args.temperature_range)
    bias_range = parse_range(args.bias_range)
    normal_bias_range = parse_range(args.normal_bias_range)
    power_range = parse_range(args.power_range)

    val = load_cv_bundle(args.run_root, component_names, args.folds, "val")
    test = load_cv_bundle(args.run_root, component_names, args.folds, "test")
    stacked_val = np.stack([val.probs_by_component[name] for name in component_names], axis=0)
    stacked_test = np.stack([test.probs_by_component[name] for name in component_names], axis=0)

    rng = np.random.default_rng(args.seed)
    top = []

    def consider(candidate):
        mode, weights, temperatures, bias, normal_bias, powers = candidate
        val_probs = combine(stacked_val, val.label_names, mode, weights, temperatures, bias, normal_bias, powers)
        select_metrics = score_probs(val_probs, val.y_idx)[:3]
        key = scalar_key(*select_metrics, args.score_macro_weight)
        top.append((key, candidate, select_metrics))
        top.sort(key=lambda item: item[0], reverse=True)
        del top[args.top_k :]

    for candidate in baseline_candidates(component_names, val.label_names):
        consider(candidate)

    abnormal_count = len([name for name in val.label_names if name != "NORMAL"])
    for trial in range(max(0, args.num_trials)):
        candidate = random_solution(
            rng,
            component_count=len(component_names),
            class_count=len(val.label_names),
            abnormal_count=abnormal_count,
            modes=modes,
            temp_range=temp_range,
            bias_range=bias_range,
            normal_bias_range=normal_bias_range,
            power_range=power_range,
        )
        consider(candidate)
        if trial and trial % 10000 == 0:
            best = top[0]
            print(f"[SEARCH] trial={trial} best_select_acc={best[2][0]:.4f} best_select_macro_f1={best[2][1]:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    solutions = []
    for rank, (_key, candidate, select_metrics) in enumerate(top, start=1):
        mode, weights, temperatures, bias, normal_bias, powers = candidate
        test_probs = combine(stacked_test, test.label_names, mode, weights, temperatures, bias, normal_bias, powers)
        eval_acc, eval_macro_f1, eval_nll, pred_idx = score_probs(test_probs, test.y_idx)
        solution = make_solution(
            component_names,
            test.label_names,
            mode,
            weights,
            temperatures,
            bias,
            normal_bias,
            powers,
            select_metrics,
            (eval_acc, eval_macro_f1, eval_nll),
        )
        solution["rank"] = rank
        solutions.append(solution)
        if rank == 1:
            write_predictions(os.path.join(args.output_dir, f"{args.tag}_oof_predictions.csv"), test, pred_idx, test_probs)
            report = classification_report(test.true_labels, [test.label_names[int(i)] for i in pred_idx], digits=4, zero_division=0, output_dict=True)
            solution["classification_report"] = report
            print(f"[BEST] rank=1 select_acc={select_metrics[0]:.4f} select_macro_f1={select_metrics[1]:.4f} eval_acc={eval_acc:.4f} eval_macro_f1={eval_macro_f1:.4f}")

    summary = {
        "tag": args.tag,
        "run_root": args.run_root,
        "component_grid": args.component_grid,
        "folds": args.folds,
        "num_trials": args.num_trials,
        "component_names": component_names,
        "best": solutions[0],
        "top_solutions": solutions,
        "oof_predictions_csv": os.path.join(args.output_dir, f"{args.tag}_oof_predictions.csv"),
    }
    summary_path = os.path.join(args.output_dir, f"{args.tag}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"[SAVED] {summary_path}")


if __name__ == "__main__":
    main()
