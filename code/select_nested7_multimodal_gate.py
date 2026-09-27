#!/usr/bin/env python
"""Lock patient-safe demographic post-fusion without loading outer-test artifacts."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from fit_nested_inner_oof_fusion import load_inner_holdout
from nested_crossfit_sparse_stacking import load_fold_data, score_probs
from optimize_legacy_fusion_cv import load_component_names
from select_ast_anchor_gate import anchor_predict
from select_robust_anchor_fusion import load_meta_oof
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
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--minimum-accuracy-gain", type=float, default=0.002)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.60)
    parser.add_argument("--maximum-nll-regression", type=float, default=0.01)
    return parser.parse_args()


def canonical_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_metadata(path):
    result = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row.get("label") or row.get("class_from_folder")
            filename = row.get("filename") or row.get("file_name")
            if not label or not filename:
                continue
            result[f"{label}/{filename}"] = (
                float(row.get("sex01") or 0.0),
                float(row.get("age_days") or "nan"),
            )
    return result


def load_meta_with_ids(args, component_names, fold, label_names=None):
    probs, y, label_names = load_meta_oof(args, component_names, fold, label_names)
    ids = []
    for inner in range(args.inner_folds):
        current = load_inner_holdout(
            args.inner_output_root,
            component_names,
            fold,
            inner,
            label_names,
        )
        ids.extend(current["file_ids"])
    outer_val, label_names = load_fold_data(
        args.outer_prob_root,
        component_names,
        fold,
        "val",
        label_names,
    )
    ids.extend(outer_val.file_ids)
    if len(ids) != len(y) or len(ids) != len(set(ids)):
        raise RuntimeError(f"invalid meta file identity in fold {fold}")
    return ids, probs, y, label_names


def crossfit_acoustic(probs, y, normal_idx, fold_lock, seed):
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    summed = np.zeros((len(y), probs.shape[2]), dtype=np.float64)
    counts = np.zeros(len(y), dtype=np.int16)
    split_accuracies = []
    blend_alpha = float(fold_lock["locked_blend_alpha"])
    for train_idx, valid_idx in splitter.split(np.zeros(len(y)), y):
        anchor, _ = anchor_predict(
            probs[train_idx, :6],
            y[train_idx],
            probs[valid_idx, :6],
            normal_idx,
            fold_lock["anchor_config"],
        )
        predicted = normalize(
            (1.0 - blend_alpha) * anchor + blend_alpha * probs[valid_idx, 6]
        )
        summed[valid_idx] += predicted
        counts[valid_idx] += 1
        split_accuracies.append(
            float(accuracy_score(y[valid_idx], predicted.argmax(axis=1)))
        )
    if not np.all(counts == 5):
        raise RuntimeError("incomplete cross-fitted acoustic predictions")
    return summed / counts[:, None], split_accuracies


def build_features(acoustic, sex, age_days, normal_idx, mode):
    acoustic = normalize(acoustic)
    abnormal = [index for index in range(acoustic.shape[1]) if index != normal_idx]
    log_ratios = np.log(acoustic[:, abnormal]) - np.log(acoustic[:, normal_idx, None])
    age_missing = np.isnan(age_days).astype(np.float64)
    age_years = np.maximum(np.nan_to_num(age_days, nan=0.0), 0.0) / 365.25
    age_log = np.log1p(age_years)
    if mode == "sex":
        metadata = sex[:, None]
    elif mode == "age":
        metadata = np.column_stack([age_years, age_log, age_missing])
    elif mode == "sex_age":
        metadata = np.column_stack([sex, age_years, age_log, age_missing, sex * age_log])
    else:
        raise ValueError(mode)
    return np.column_stack([log_ratios, metadata])


def aligned_predict(scaler, model, features, class_count):
    raw = model.predict_proba(scaler.transform(features))
    result = np.zeros((len(features), class_count), dtype=np.float64)
    for column, class_index in enumerate(model.classes_):
        result[:, int(class_index)] = raw[:, column]
    return normalize(result)


def evaluate_candidates(acoustic, y, sex, age, normal_idx, seed):
    modes = (("sex", 1), ("age", 1), ("sex_age", 2))
    c_values = (0.001, 0.01, 0.10)
    alphas = (0.05, 0.10, 0.20, 0.35, 0.50)
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y))
    baseline_split_acc = [
        float(accuracy_score(y[valid], acoustic[valid].argmax(axis=1)))
        for _, valid in splits
    ]
    baseline = score_probs(y, acoustic)
    baseline["split_accuracies"] = baseline_split_acc
    sums = {}
    counts = {}
    split_acc = defaultdict(list)

    feature_cache = {
        mode: build_features(acoustic, sex, age, normal_idx, mode)
        for mode, _ in modes
    }
    for train_idx, valid_idx in splits:
        for mode, complexity in modes:
            features = feature_cache[mode]
            for c_value in c_values:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(features[train_idx])
                model = LogisticRegression(
                    C=c_value,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=0,
                ).fit(x_train, y[train_idx])
                post = aligned_predict(
                    scaler,
                    model,
                    features[valid_idx],
                    acoustic.shape[1],
                )
                for alpha in alphas:
                    config = (mode, c_value, alpha, complexity)
                    predicted = normalize(
                        alpha * post + (1.0 - alpha) * acoustic[valid_idx]
                    )
                    if config not in sums:
                        sums[config] = np.zeros_like(acoustic)
                        counts[config] = np.zeros(len(y), dtype=np.int16)
                    sums[config][valid_idx] += predicted
                    counts[config][valid_idx] += 1
                    split_acc[config].append(
                        float(accuracy_score(y[valid_idx], predicted.argmax(axis=1)))
                    )

    candidates = []
    for config, summed in sums.items():
        if not np.all(counts[config] == 5):
            raise RuntimeError("incomplete demographic OOF predictions")
        current = score_probs(y, summed / counts[config][:, None])
        values = np.asarray(split_acc[config], dtype=np.float64)
        current.update({
            "mode": config[0],
            "C": config[1],
            "blend_alpha": config[2],
            "complexity": config[3],
            "split_accuracy_std": float(values.std(ddof=1)),
            "split_accuracy_se": float(values.std(ddof=1) / math.sqrt(len(values))),
            "split_accuracies": values.tolist(),
        })
        candidates.append(current)
    return baseline, candidates


def select_positive_one_se(candidates):
    best = sorted(candidates, key=lambda item: (-item["accuracy"], item["nll"]))[0]
    cutoff = best["accuracy"] - best["split_accuracy_se"]
    eligible = [item for item in candidates if item["accuracy"] >= cutoff]
    selected = sorted(
        eligible,
        key=lambda item: (
            item["complexity"],
            item["blend_alpha"],
            item["C"],
            item["nll"],
        ),
    )[0]
    return selected, best, cutoff


def main():
    args = parse_args()
    with open(args.acoustic_lock, "r", encoding="utf-8") as handle:
        acoustic_lock = json.load(handle)
    acoustic_hash = acoustic_lock.pop("selection_sha256")
    if canonical_hash(acoustic_lock) != acoustic_hash:
        raise RuntimeError("acoustic selection lock hash mismatch")
    acoustic_lock["selection_sha256"] = acoustic_hash
    if not acoustic_lock.get("selection_only") or acoustic_lock.get("test_used_for_selection"):
        raise RuntimeError("invalid acoustic selection lock provenance")
    component_names = load_component_names(args.component_grid)
    if component_names != acoustic_lock["component_names"]:
        raise RuntimeError("acoustic lock component mismatch")
    metadata = load_metadata(args.metadata_csv)
    folds = []
    label_names = None

    for fold in range(5):
        ids, probs, y, label_names = load_meta_with_ids(
            args,
            component_names,
            fold,
            label_names,
        )
        sex = np.asarray([metadata[file_id][0] for file_id in ids], dtype=np.float64)
        age = np.asarray([metadata[file_id][1] for file_id in ids], dtype=np.float64)
        normal_idx = label_names.index("NORMAL")
        fold_acoustic_lock = acoustic_lock["folds"][fold]
        acoustic, acoustic_split_acc = crossfit_acoustic(
            probs,
            y,
            normal_idx,
            fold_acoustic_lock,
            args.seed + fold,
        )
        baseline, candidates = evaluate_candidates(
            acoustic,
            y,
            sex,
            age,
            normal_idx,
            args.seed + 100 + fold,
        )
        selected, best, cutoff = select_positive_one_se(candidates)
        paired = np.asarray(selected["split_accuracies"]) - np.asarray(
            baseline["split_accuracies"]
        )
        paired_se = float(paired.std(ddof=1) / math.sqrt(len(paired)))
        required_gain = max(args.minimum_accuracy_gain, paired_se)
        gain = float(selected["accuracy"] - baseline["accuracy"])
        win_fraction = float(np.mean(paired > 0.0))
        gate_passed = bool(
            gain > required_gain
            and win_fraction >= args.minimum_win_fraction
            and selected["nll"] <= baseline["nll"] + args.maximum_nll_regression
        )
        folds.append({
            "outer_fold": fold,
            "meta_oof_count": len(y),
            "acoustic_lock_sha256": acoustic_lock["selection_sha256"],
            "acoustic_crossfit_metrics": baseline,
            "acoustic_crossfit_split_accuracies": acoustic_split_acc,
            "selected_positive_demographic": selected,
            "best_positive_demographic": best,
            "positive_one_se_cutoff": cutoff,
            "accuracy_gain_over_acoustic": gain,
            "paired_accuracy_delta_se": paired_se,
            "required_accuracy_gain": required_gain,
            "paired_split_win_fraction": win_fraction,
            "gate_passed": gate_passed,
            "locked_family": "demographic_postfusion" if gate_passed else "acoustic_only",
            "locked_config": (
                {"mode": selected["mode"], "C": selected["C"], "blend_alpha": selected["blend_alpha"]}
                if gate_passed else None
            ),
            "all_positive_demographic_candidates": candidates,
        })
        print(
            f"[FOLD] {fold} acoustic={baseline['accuracy']:.4f} "
            f"post={selected['accuracy']:.4f} mode={selected['mode']} "
            f"alpha={selected['blend_alpha']:.2f} gain={gain:+.4f} "
            f"wins={win_fraction:.2f} gate={gate_passed}",
            flush=True,
        )

    payload = {
        "tag": "nested7_strict_multimodal_selection",
        "selection_only": True,
        "outer_test_probabilities_loaded": False,
        "test_used_for_selection": False,
        "component_names": component_names,
        "acoustic_selection_lock": str(Path(args.acoustic_lock).resolve()),
        "acoustic_selection_sha256": acoustic_lock["selection_sha256"],
        "metadata_used": ["sex01", "age_days"],
        "seed": args.seed,
        "selection_protocol": (
            "Within each outer fold, acoustic probabilities for train+validation patients are "
            "cross-fitted with repeated 5-fold x 5 CV under the locked acoustic method. "
            "Sex/age Logistic Regression candidates are evaluated by a separate repeated "
            "5-fold x 5 CV. A one-SE rule favors low-complexity and small-blend candidates. "
            "Metadata is admitted only when gain exceeds both 0.002 and paired-difference SE, "
            "wins at least 60% of splits, and NLL regresses by at most 0.01. Outer test is not loaded."
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
