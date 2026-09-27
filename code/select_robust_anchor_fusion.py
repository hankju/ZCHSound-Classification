#!/usr/bin/env python
"""Select a conservative Nested6-anchored convex fusion without reading outer test probabilities."""

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

from fit_nested_inner_oof_fusion import load_inner_holdout
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
from strict_samefold_sparse_stacking import normalize, select_config


@dataclass(frozen=True)
class ConvexCandidate:
    name: str
    component_count: int
    l2: Optional[float]
    prior: str
    complexity: int


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-output-root", required=True)
    parser.add_argument("--outer-prob-root", required=True)
    parser.add_argument("--component-grid", required=True)
    parser.add_argument("--outer-split-dir", required=True)
    parser.add_argument("--inner-split-dir", required=True)
    parser.add_argument("--output-lock", required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--anchor-components", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--minimum-accuracy-gain", type=float, default=0.003)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.60)
    parser.add_argument("--maximum-nll-regression", type=float, default=0.02)
    return parser.parse_args()


def canonical_hash(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def candidate_prior(candidate, total_components):
    if candidate.prior == "uniform":
        return np.full(candidate.component_count, 1.0 / candidate.component_count)
    if candidate.prior == "anchor":
        prior = np.zeros(total_components, dtype=np.float64)
        prior[:6] = 1.0 / 6.0
        return prior
    raise ValueError(candidate.prior)


def fit_simplex_weights(probs, y, prior, l2):
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-10, 1.0)
    prior = np.asarray(prior, dtype=np.float64)
    row = np.arange(len(y))

    def objective(weights):
        mixed = np.einsum("nmc,m->nc", probs, weights)
        selected = np.clip(mixed[row, y], 1e-12, 1.0)
        penalty = l2 * np.square(weights - prior).sum()
        return float(-np.log(selected).mean() + penalty)

    def gradient(weights):
        mixed = np.einsum("nmc,m->nc", probs, weights)
        selected = np.clip(mixed[row, y], 1e-12, 1.0)
        true_component_probs = probs[row, :, y]
        grad = -(true_component_probs / selected[:, None]).mean(axis=0)
        return grad + 2.0 * l2 * (weights - prior)

    result = minimize(
        objective,
        prior,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(prior),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"simplex optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def predict_convex(probs, weights):
    return normalize(np.einsum("nmc,m->nc", probs, weights))


def make_candidates(total_components, anchor_components):
    if anchor_components != 6 or total_components != 10:
        raise ValueError("the locked robust search expects six anchor and ten total components")
    candidates = [
        ConvexCandidate("uniform6", 6, None, "uniform", 0),
        ConvexCandidate("simplex6_l2_0.1", 6, 0.1, "uniform", 1),
        ConvexCandidate("simplex6_l2_1", 6, 1.0, "uniform", 1),
        ConvexCandidate("simplex6_l2_10", 6, 10.0, "uniform", 1),
        ConvexCandidate("uniform10", 10, None, "uniform", 2),
        ConvexCandidate("anchor10_l2_0", 10, 0.0, "anchor", 3),
        ConvexCandidate("anchor10_l2_0.1", 10, 0.1, "anchor", 3),
        ConvexCandidate("anchor10_l2_1", 10, 1.0, "anchor", 3),
        ConvexCandidate("anchor10_l2_10", 10, 10.0, "anchor", 3),
        ConvexCandidate("anchor10_l2_100", 10, 100.0, "anchor", 3),
    ]
    return candidates


def evaluate_convex_candidates(probs, y, candidates, splits):
    sample_count, total_components, class_count = probs.shape
    sums = {candidate.name: np.zeros((sample_count, class_count)) for candidate in candidates}
    counts = {candidate.name: np.zeros(sample_count, dtype=np.int16) for candidate in candidates}
    split_accuracies = {candidate.name: [] for candidate in candidates}
    weights_by_split = {candidate.name: [] for candidate in candidates}

    for train_idx, valid_idx in splits:
        for candidate in candidates:
            current_train = probs[train_idx, : candidate.component_count]
            current_valid = probs[valid_idx, : candidate.component_count]
            prior = candidate_prior(candidate, total_components)
            if candidate.l2 is None:
                weights = prior
            else:
                weights = fit_simplex_weights(current_train, y[train_idx], prior, candidate.l2)
            predicted = predict_convex(current_valid, weights)
            sums[candidate.name][valid_idx] += predicted
            counts[candidate.name][valid_idx] += 1
            split_accuracies[candidate.name].append(
                float(accuracy_score(y[valid_idx], predicted.argmax(axis=1)))
            )
            weights_by_split[candidate.name].append(weights.tolist())

    results = []
    for candidate in candidates:
        current_counts = counts[candidate.name]
        if not np.all(current_counts == current_counts[0]):
            raise RuntimeError(f"incomplete OOF predictions for {candidate.name}")
        oof_probs = sums[candidate.name] / current_counts[:, None]
        metrics = score_probs(y, oof_probs)
        split_acc = np.asarray(split_accuracies[candidate.name], dtype=np.float64)
        results.append({
            "name": candidate.name,
            "component_count": candidate.component_count,
            "l2": candidate.l2,
            "prior": candidate.prior,
            "complexity": candidate.complexity,
            **metrics,
            "split_accuracy_std": float(split_acc.std(ddof=1)),
            "split_accuracy_se": float(split_acc.std(ddof=1) / math.sqrt(len(split_acc))),
            "split_accuracies": split_acc.tolist(),
            "mean_weights": np.mean(weights_by_split[candidate.name], axis=0).tolist(),
        })
    return results


def evaluate_sparse_anchor(probs, y, normal_idx, splits, seed, fold):
    ranked = select_config(probs, y, normal_idx, 5, 5, seed + fold)
    _, config, _, _ = ranked[0]
    selector, size, c_value, class_weight, alpha = config
    sample_count, _, class_count = probs.shape
    summed = np.zeros((sample_count, class_count), dtype=np.float64)
    counts = np.zeros(sample_count, dtype=np.int16)
    split_accuracies = []

    for train_idx, valid_idx in splits:
        order = component_order(probs[train_idx], y[train_idx], selector, size)
        selected = order[:size]
        x_train, _ = build_features(probs[train_idx], selected, normal_idx)
        x_valid, base_valid = build_features(probs[valid_idx], selected, normal_idx)
        scaler, model = fit_stacker(x_train, y[train_idx], c_value, class_weight)
        stacked = aligned_predict_proba(scaler, model, x_valid, class_count)
        predicted = normalize(alpha * stacked + (1.0 - alpha) * base_valid)
        summed[valid_idx] += predicted
        counts[valid_idx] += 1
        split_accuracies.append(float(accuracy_score(y[valid_idx], predicted.argmax(axis=1))))

    if not np.all(counts == counts[0]):
        raise RuntimeError("incomplete sparse-anchor OOF predictions")
    metrics = score_probs(y, summed / counts[:, None])
    split_acc = np.asarray(split_accuracies, dtype=np.float64)
    return {
        "name": "nested6_sparse_anchor",
        "selector": selector,
        "component_count": size,
        "C": c_value,
        "class_weight": class_weight,
        "stacking_blend_alpha": alpha,
        **metrics,
        "split_accuracy_std": float(split_acc.std(ddof=1)),
        "split_accuracy_se": float(split_acc.std(ddof=1) / math.sqrt(len(split_acc))),
        "split_accuracies": split_acc.tolist(),
    }


def select_one_se(results):
    best = sorted(results, key=lambda item: (-item["accuracy"], item["nll"]))[0]
    cutoff = best["accuracy"] - best["split_accuracy_se"]
    eligible = [item for item in results if item["accuracy"] >= cutoff]
    selected = sorted(
        eligible,
        key=lambda item: (
            item["complexity"],
            item["component_count"],
            item["nll"],
            item["split_accuracy_std"],
            -item["accuracy"],
        ),
    )[0]
    return selected, best, cutoff


def load_meta_oof(args, component_names, outer_fold, label_names=None):
    outer_val, label_names = load_fold_data(
        args.outer_prob_root, component_names, outer_fold, "val", label_names
    )
    with open(
        Path(args.outer_split_dir) / f"cv5_tvt_fold{outer_fold}.json",
        "r",
        encoding="utf-8",
    ) as handle:
        outer_split = json.load(handle)

    ids = []
    y_parts = []
    prob_parts = []
    for inner_fold in range(args.inner_folds):
        current = load_inner_holdout(
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
        if set(current["file_ids"]) != set(inner_split["test_files"]):
            raise RuntimeError(f"outer {outer_fold} inner {inner_fold} manifest mismatch")
        ids.extend(current["file_ids"])
        y_parts.append(current["y"])
        prob_parts.append(current["probs"])

    ids.extend(outer_val.file_ids)
    y_parts.append(outer_val.y)
    prob_parts.append(outer_val.probs)
    meta_patients = [patient_id(item) for item in ids]
    test_patients = {patient_id(item) for item in outer_split["test_files"]}
    expected = set(outer_split["train_files"]) | set(outer_split["val_files"])
    if (
        len(meta_patients) != len(set(meta_patients))
        or set(ids) != expected
        or set(meta_patients) & test_patients
    ):
        raise RuntimeError(f"outer fold {outer_fold} meta provenance failed")
    return np.concatenate(prob_parts), np.concatenate(y_parts), label_names


def main():
    args = parse_args()
    component_names = load_component_names(args.component_grid)
    candidates = make_candidates(len(component_names), args.anchor_components)
    folds = []
    label_names = None

    for outer_fold in range(args.outer_folds):
        probs, y, label_names = load_meta_oof(args, component_names, outer_fold, label_names)
        splitter = RepeatedStratifiedKFold(
            n_splits=args.cv_folds,
            n_repeats=args.cv_repeats,
            random_state=args.seed + outer_fold,
        )
        splits = list(splitter.split(np.zeros(len(y)), y))
        anchor = evaluate_sparse_anchor(
            probs[:, : args.anchor_components],
            y,
            label_names.index("NORMAL"),
            splits,
            args.seed,
            outer_fold,
        )
        convex_results = evaluate_convex_candidates(probs, y, candidates, splits)
        selected, best, cutoff = select_one_se(convex_results)
        combined_se = math.sqrt(
            anchor["split_accuracy_se"] ** 2 + selected["split_accuracy_se"] ** 2
        )
        required_gain = max(args.minimum_accuracy_gain, combined_se)
        paired = np.asarray(selected["split_accuracies"]) - np.asarray(
            anchor["split_accuracies"]
        )
        win_fraction = float(np.mean(paired > 0.0))
        accuracy_gain = selected["accuracy"] - anchor["accuracy"]
        gate_passed = bool(
            accuracy_gain > required_gain
            and win_fraction >= args.minimum_win_fraction
            and selected["nll"] <= anchor["nll"] + args.maximum_nll_regression
        )
        fold_result = {
            "outer_fold": outer_fold,
            "meta_oof_count": len(y),
            "anchor": anchor,
            "convex_selected_by_one_se": selected,
            "convex_best_accuracy_candidate": best,
            "one_se_accuracy_cutoff": cutoff,
            "accuracy_gain_over_anchor": accuracy_gain,
            "combined_accuracy_se": combined_se,
            "required_accuracy_gain": required_gain,
            "paired_split_win_fraction": win_fraction,
            "gate_passed": gate_passed,
            "locked_family": "robust_convex" if gate_passed else "nested6_sparse_anchor",
            "all_convex_candidates": convex_results,
        }
        folds.append(fold_result)
        print(
            f"[FOLD] {outer_fold} anchor={anchor['accuracy']:.4f} "
            f"convex={selected['accuracy']:.4f} gain={accuracy_gain:+.4f} "
            f"wins={win_fraction:.2f} gate={gate_passed}",
            flush=True,
        )

    payload = {
        "tag": "nested10_robust_anchor_selection",
        "selection_only": True,
        "outer_test_probabilities_loaded": False,
        "test_used_for_selection": False,
        "component_names": component_names,
        "anchor_component_count": args.anchor_components,
        "selection_protocol": (
            "Repeated five-fold, five-repeat meta-OOF CV. Convex component weights are "
            "nonnegative and sum to one. A one-standard-error rule favors the simplest "
            "convex candidate. It may replace the Nested6 sparse anchor only when its "
            "accuracy gain exceeds both 0.003 and the combined CV standard error, wins "
            "at least 60% of paired CV splits, and regresses NLL by no more than 0.02."
        ),
        "seed": args.seed,
        "minimum_accuracy_gain": args.minimum_accuracy_gain,
        "minimum_win_fraction": args.minimum_win_fraction,
        "maximum_nll_regression": args.maximum_nll_regression,
        "folds": folds,
        "robust_fold_count": sum(item["gate_passed"] for item in folds),
    }
    payload["lock_sha256"] = canonical_hash(payload)
    output = Path(args.output_lock)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[LOCK] {output} sha256={payload['lock_sha256']}", flush=True)


if __name__ == "__main__":
    main()
