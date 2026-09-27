#!/usr/bin/env python
"""Select multi-objective demographic fusion without loading outer-test artifacts."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from optimize_legacy_fusion_cv import load_component_names
from select_nested7_multimodal_gate import (
    canonical_hash,
    crossfit_acoustic,
    load_meta_with_ids,
    load_metadata,
)
from strict_samefold_sparse_stacking import normalize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-output-root", required=True)
    parser.add_argument("--outer-prob-root", required=True)
    parser.add_argument("--component-grid", required=True)
    parser.add_argument("--outer-split-dir", required=True)
    parser.add_argument("--inner-split-dir", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--acoustic-lock", required=True)
    parser.add_argument("--output-lock", required=True)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def aligned_probabilities(model, features, class_count):
    raw = model.predict_proba(features)
    result = np.zeros((len(features), class_count), dtype=np.float64)
    for column, class_index in enumerate(model.classes_):
        result[:, int(class_index)] = raw[:, column]
    return normalize(result)


def metadata_features(sex, age_days):
    age_missing = np.isnan(age_days).astype(np.float64)
    age_years = np.maximum(np.nan_to_num(age_days, nan=0.0), 0.0) / 365.25
    age_log = np.log1p(age_years)
    thresholds = np.column_stack([
        age_years <= threshold for threshold in (0.25, 0.50, 1.0, 2.0, 5.0)
    ]).astype(np.float64)
    return np.column_stack([
        sex,
        np.minimum(age_years, 15.0),
        age_log,
        np.sqrt(age_years),
        age_missing,
        thresholds,
        sex * age_log,
    ])


def acoustic_features(acoustic, normal_idx):
    acoustic = normalize(acoustic)
    abnormal = [index for index in range(acoustic.shape[1]) if index != normal_idx]
    return np.log(acoustic[:, abnormal]) - np.log(acoustic[:, normal_idx, None])


def combined_features(acoustic, sex, age, normal_idx):
    return np.column_stack([
        acoustic_features(acoustic, normal_idx),
        metadata_features(sex, age),
    ])


def candidate_library():
    candidates = []
    for c_value in (0.001, 0.01, 0.10):
        for class_weight in (None, "balanced"):
            for alpha in (0.10, 0.25, 0.50):
                candidates.append({
                    "family": "residual_lr",
                    "C": c_value,
                    "class_weight": class_weight,
                    "blend_alpha": alpha,
                    "complexity": 1,
                })
    for c_value in (0.01, 0.10, 1.0):
        for strength in (0.10, 0.25, 0.50, 1.0):
            candidates.append({
                "family": "bayesian_age_sex_prior",
                "C": c_value,
                "prior_strength": strength,
                "complexity": 1,
            })
    for l2 in (1.0, 10.0):
        for alpha in (0.10, 0.25, 0.50):
            candidates.append({
                "family": "hist_gradient_boosting",
                "l2_regularization": l2,
                "blend_alpha": alpha,
                "complexity": 2,
            })
    for class_weight in (None, "balanced"):
        for alpha in (0.10, 0.25, 0.50):
            candidates.append({
                "family": "extra_trees",
                "class_weight": class_weight,
                "blend_alpha": alpha,
                "complexity": 2,
            })
    return candidates


def fit_candidate(candidate, acoustic, sex, age, y, normal_idx, seed):
    family = candidate["family"]
    if family == "bayesian_age_sex_prior":
        features = metadata_features(sex, age)
        scaler = StandardScaler()
        model = LogisticRegression(
            C=float(candidate["C"]),
            solver="lbfgs",
            max_iter=2000,
            random_state=0,
        ).fit(scaler.fit_transform(features), y)
        counts = np.bincount(y, minlength=acoustic.shape[1]).astype(np.float64) + 1.0
        prior = counts / counts.sum()
        return {"scaler": scaler, "model": model, "class_prior": prior}

    features = combined_features(acoustic, sex, age, normal_idx)
    if family == "residual_lr":
        scaler = StandardScaler()
        model = LogisticRegression(
            C=float(candidate["C"]),
            class_weight=candidate["class_weight"],
            solver="lbfgs",
            max_iter=2000,
            random_state=0,
        ).fit(scaler.fit_transform(features), y)
        return {"scaler": scaler, "model": model}
    if family == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=float(candidate["l2_regularization"]),
            random_state=seed,
        ).fit(features, y)
        return {"model": model}
    if family == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=8,
            max_features=1.0,
            class_weight=candidate["class_weight"],
            random_state=seed,
            n_jobs=1,
        ).fit(features, y)
        return {"model": model}
    raise ValueError(family)


def predict_candidate(candidate, fitted, acoustic, sex, age, normal_idx):
    family = candidate["family"]
    class_count = acoustic.shape[1]
    if family == "bayesian_age_sex_prior":
        features = fitted["scaler"].transform(metadata_features(sex, age))
        metadata_probs = aligned_probabilities(fitted["model"], features, class_count)
        strength = float(candidate["prior_strength"])
        corrected = (
            np.log(np.clip(acoustic, 1e-12, 1.0))
            + strength
            * (
                np.log(np.clip(metadata_probs, 1e-12, 1.0))
                - np.log(fitted["class_prior"])[None, :]
            )
        )
        return softmax(corrected, axis=1)

    features = combined_features(acoustic, sex, age, normal_idx)
    if family == "residual_lr":
        features = fitted["scaler"].transform(features)
    post = aligned_probabilities(fitted["model"], features, class_count)
    alpha = float(candidate["blend_alpha"])
    return normalize(alpha * post + (1.0 - alpha) * acoustic)


def detailed_metrics(y, probs, minority_indices, normal_idx):
    probs = normalize(probs)
    pred = probs.argmax(axis=1)
    recalls = recall_score(
        y,
        pred,
        labels=np.arange(probs.shape[1]),
        average=None,
        zero_division=0,
    )
    class_f1 = f1_score(
        y,
        pred,
        labels=np.arange(probs.shape[1]),
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(class_f1.mean()),
        "minority_recall": float(recalls[minority_indices].mean()),
        "minority_f1": float(class_f1[minority_indices].mean()),
        "normal_recall": float(recalls[normal_idx]),
        "nll": float(log_loss(y, probs, labels=np.arange(probs.shape[1]))),
        "per_class_recall": recalls.tolist(),
        "per_class_f1": class_f1.tolist(),
    }


def candidate_key(candidate):
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"))


def evaluate_library(acoustic, y, sex, age, label_names, seed):
    normal_idx = label_names.index("NORMAL")
    minority_indices = np.asarray(
        [label_names.index(name) for name in ("ASD", "PDA", "PFO")],
        dtype=np.int64,
    )
    candidates = candidate_library()
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y))
    sums = {
        candidate_key(candidate): np.zeros_like(acoustic)
        for candidate in candidates
    }
    counts = {
        candidate_key(candidate): np.zeros(len(y), dtype=np.int16)
        for candidate in candidates
    }
    split_metrics = defaultdict(lambda: defaultdict(list))
    baseline_splits = defaultdict(list)

    for split_index, (train_idx, valid_idx) in enumerate(splits):
        baseline_current = detailed_metrics(
            y[valid_idx],
            acoustic[valid_idx],
            minority_indices,
            normal_idx,
        )
        for metric in ("accuracy", "macro_f1", "minority_recall"):
            baseline_splits[metric].append(baseline_current[metric])

        fitted_cache = {}
        for candidate in candidates:
            fit_key = tuple(
                (key, value)
                for key, value in sorted(candidate.items())
                if key not in ("blend_alpha", "prior_strength", "complexity")
            )
            if fit_key not in fitted_cache:
                fitted_cache[fit_key] = fit_candidate(
                    candidate,
                    acoustic[train_idx],
                    sex[train_idx],
                    age[train_idx],
                    y[train_idx],
                    normal_idx,
                    seed + split_index,
                )
            predicted = predict_candidate(
                candidate,
                fitted_cache[fit_key],
                acoustic[valid_idx],
                sex[valid_idx],
                age[valid_idx],
                normal_idx,
            )
            key = candidate_key(candidate)
            sums[key][valid_idx] += predicted
            counts[key][valid_idx] += 1
            current = detailed_metrics(
                y[valid_idx],
                predicted,
                minority_indices,
                normal_idx,
            )
            for metric in ("accuracy", "macro_f1", "minority_recall"):
                split_metrics[key][metric].append(current[metric])

    baseline = detailed_metrics(y, acoustic, minority_indices, normal_idx)
    baseline["split_metrics"] = dict(baseline_splits)
    results = []
    for candidate in candidates:
        key = candidate_key(candidate)
        if not np.all(counts[key] == 5):
            raise RuntimeError(f"incomplete OOF predictions: {key}")
        current = detailed_metrics(
            y,
            sums[key] / counts[key][:, None],
            minority_indices,
            normal_idx,
        )
        current["candidate"] = candidate
        current["split_metrics"] = dict(split_metrics[key])
        for metric in ("accuracy", "macro_f1", "minority_recall"):
            values = np.asarray(current["split_metrics"][metric], dtype=np.float64)
            current[f"{metric}_split_se"] = float(
                values.std(ddof=1) / math.sqrt(len(values))
            )
        results.append(current)
    return baseline, results


def select_one_se(candidates, metric):
    best = sorted(candidates, key=lambda item: (-item[metric], item["nll"]))[0]
    cutoff = best[metric] - best[f"{metric}_split_se"]
    eligible = [item for item in candidates if item[metric] >= cutoff]
    selected = sorted(
        eligible,
        key=lambda item: (
            item["candidate"]["complexity"],
            item["candidate"].get("blend_alpha", item["candidate"].get("prior_strength", 0.0)),
            item["nll"],
            -item[metric],
        ),
    )[0]
    return selected, best, cutoff


def gate_branch(name, metric, selected, baseline, minimum_gain, minimum_wins):
    selected_splits = np.asarray(selected["split_metrics"][metric], dtype=np.float64)
    baseline_splits = np.asarray(baseline["split_metrics"][metric], dtype=np.float64)
    paired = selected_splits - baseline_splits
    paired_se = float(paired.std(ddof=1) / math.sqrt(len(paired)))
    gain = float(selected[metric] - baseline[metric])
    win_fraction = float(np.mean(paired > 0.0))
    common = (
        gain > max(minimum_gain, paired_se)
        and win_fraction >= minimum_wins
        and selected["nll"] <= baseline["nll"] + 0.03
    )
    if name == "accuracy":
        guardrail = selected["macro_f1"] >= baseline["macro_f1"] - 0.005
    elif name == "macro_f1":
        guardrail = (
            selected["accuracy"] >= baseline["accuracy"] - 0.005
            and selected["normal_recall"] >= baseline["normal_recall"] - 0.01
        )
    elif name == "minority_recall":
        guardrail = (
            selected["accuracy"] >= baseline["accuracy"] - 0.005
            and selected["macro_f1"] >= baseline["macro_f1"] - 0.002
            and selected["normal_recall"] >= baseline["normal_recall"] - 0.01
        )
    else:
        raise ValueError(name)
    return {
        "branch": name,
        "metric": metric,
        "gain": gain,
        "paired_delta_se": paired_se,
        "required_gain": max(minimum_gain, paired_se),
        "paired_split_win_fraction": win_fraction,
        "guardrail_passed": bool(guardrail),
        "gate_passed": bool(common and guardrail),
    }


def main():
    args = parse_args()
    with open(args.acoustic_lock, "r", encoding="utf-8") as handle:
        acoustic_lock = json.load(handle)
    acoustic_hash = acoustic_lock.pop("selection_sha256")
    if canonical_hash(acoustic_lock) != acoustic_hash:
        raise RuntimeError("acoustic selection lock hash mismatch")
    acoustic_lock["selection_sha256"] = acoustic_hash
    if not acoustic_lock.get("selection_only") or acoustic_lock.get("test_used_for_selection"):
        raise RuntimeError("invalid acoustic lock provenance")

    component_names = load_component_names(args.component_grid)
    if component_names != acoustic_lock["component_names"]:
        raise RuntimeError("component mismatch")
    metadata = load_metadata(args.metadata_csv)
    folds = []
    label_names = None
    branch_specs = (
        ("accuracy", "accuracy", 0.002, 0.60),
        ("macro_f1", "macro_f1", 0.005, 0.60),
        ("minority_recall", "minority_recall", 0.01, 0.60),
    )

    for fold in range(5):
        ids, probs, y, label_names = load_meta_with_ids(
            args,
            component_names,
            fold,
            label_names,
        )
        sex = np.asarray([metadata[file_id][0] for file_id in ids], dtype=np.float64)
        age = np.asarray([metadata[file_id][1] for file_id in ids], dtype=np.float64)
        acoustic, _ = crossfit_acoustic(
            probs,
            y,
            label_names.index("NORMAL"),
            acoustic_lock["folds"][fold],
            args.seed + fold,
        )
        baseline, candidates = evaluate_library(
            acoustic,
            y,
            sex,
            age,
            label_names,
            args.seed + 100 + fold,
        )
        branches = []
        selected_by_branch = {}
        for name, metric, minimum_gain, minimum_wins in branch_specs:
            selected, best, cutoff = select_one_se(candidates, metric)
            gate = gate_branch(
                name,
                metric,
                selected,
                baseline,
                minimum_gain,
                minimum_wins,
            )
            gate.update({
                "selected_candidate": selected,
                "best_candidate": best,
                "one_se_cutoff": cutoff,
            })
            branches.append(gate)
            selected_by_branch[name] = gate

        locked_branch = "acoustic_only"
        locked_candidate = None
        for name in ("accuracy", "macro_f1", "minority_recall"):
            if selected_by_branch[name]["gate_passed"]:
                locked_branch = name
                locked_candidate = selected_by_branch[name]["selected_candidate"]["candidate"]
                break
        folds.append({
            "outer_fold": fold,
            "meta_oof_count": len(y),
            "baseline": baseline,
            "branch_results": branches,
            "locked_branch": locked_branch,
            "locked_candidate": locked_candidate,
            "all_candidates": candidates,
        })
        print(
            f"[FOLD] {fold} baseline_acc={baseline['accuracy']:.4f} "
            f"baseline_macro={baseline['macro_f1']:.4f} "
            f"baseline_minrec={baseline['minority_recall']:.4f} "
            f"locked={locked_branch} candidate={locked_candidate}",
            flush=True,
        )

    payload = {
        "tag": "nested7_multimodal_multiobjective_selection",
        "selection_only": True,
        "outer_test_probabilities_loaded": False,
        "test_used_for_selection": False,
        "component_names": component_names,
        "acoustic_selection_sha256": acoustic_lock["selection_sha256"],
        "metadata_used": ["sex01", "age_days"],
        "minority_classes": ["ASD", "PDA", "PFO"],
        "branch_priority": ["accuracy", "macro_f1", "minority_recall"],
        "accuracy_guardrail_for_f1_branches": -0.005,
        "normal_recall_guardrail_for_f1_branches": -0.01,
        "seed": args.seed,
        "selection_protocol": (
            "Each outer fold uses cross-fitted locked-acoustic probabilities for outer "
            "train+validation patients only. Residual LR, Bayesian age/sex prior correction, "
            "regularized histogram gradient boosting, and ExtraTrees are evaluated by repeated "
            "5-fold x 5 CV. Separate accuracy, macro-F1, and ASD/PDA/PFO recall gates use "
            "one-SE candidate selection, paired split stability, NLL limits, and accuracy/normal "
            "recall guardrails. Branch priority is fixed before outer-test evaluation."
        ),
        "folds": folds,
    }
    payload["selection_sha256"] = canonical_hash(payload)
    output = Path(args.output_lock).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[LOCKED] {output} sha256={payload['selection_sha256']}", flush=True)


if __name__ == "__main__":
    main()
