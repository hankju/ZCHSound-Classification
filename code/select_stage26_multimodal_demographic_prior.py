#!/usr/bin/env python
"""Lock train-fitted demographic-prior correction using validation only."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from select_nested7_multimodal_multiobjective import detailed_metrics
from strict_samefold_sparse_stacking import normalize


LABELS = ("ASD", "NORMAL", "PDA", "PFO", "VSD")
CURRICULUM = (
    "ast_curriculum_balanced_seed2_mean",
    "beats_curriculum_conservative_seed2_mean",
)
MINORITY = ("ASD", "PDA", "PFO")
C_VALUES = (0.01, 0.1, 1.0)
STRENGTHS = (0.1, 0.25, 0.5)
MODES = ("age_only", "age_sex")
SCOPES = ("abnormal_conditional", "all_classes")
ADAPTATIONS = ("fixed", "entropy")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-lock", required=True)
    parser.add_argument("--split-seed", type=int, default=20268020)
    parser.add_argument("--cv-seed", type=int, default=20262600)
    return parser.parse_args()


def canonical_hash(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(path):
    result = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row.get("label") or row.get("class_from_folder")
            filename = row.get("filename") or row.get("file_name")
            if not label or not filename:
                continue
            result[f"{label}/{filename}"] = (
                float(row["sex01"]),
                float(row["age_days"]),
                LABELS.index(label),
            )
    return result


def read_acoustic(root, split_seed, fold, split):
    loaded = []
    reference_ids = None
    reference_y = None
    for name in CURRICULUM:
        path = (
            Path(root)
            / f"seed{split_seed}"
            / f"fold{fold}"
            / "components"
            / name
            / f"{split}_file_probs.csv"
        )
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError(f"empty Stage26 acoustic export: {path}")
        ids = [row["file_id"] for row in rows]
        y = np.asarray(
            [LABELS.index(row["true_label"]) for row in rows], dtype=np.int64
        )
        probs = normalize(np.asarray([
            [float(row[f"prob_{label}"]) for label in LABELS]
            for row in rows
        ], dtype=np.float64))
        if reference_ids is None:
            reference_ids, reference_y = ids, y
        elif ids != reference_ids or not np.array_equal(y, reference_y):
            raise RuntimeError(f"Stage26 acoustic alignment failed fold={fold}")
        loaded.append(probs)
    return reference_ids, reference_y, normalize(np.mean(loaded, axis=0))


def metadata_arrays(file_ids, metadata):
    missing = [file_id for file_id in file_ids if file_id not in metadata]
    if missing:
        raise RuntimeError(f"missing metadata for {len(missing)} patients")
    sex = np.asarray([metadata[file_id][0] for file_id in file_ids], dtype=np.float64)
    age = np.asarray([metadata[file_id][1] for file_id in file_ids], dtype=np.float64)
    y = np.asarray([metadata[file_id][2] for file_id in file_ids], dtype=np.int64)
    return sex, age, y


def metadata_features(sex, age_days, mode):
    age_missing = np.isnan(age_days).astype(np.float64)
    age_years = np.maximum(np.nan_to_num(age_days, nan=0.0), 0.0) / 365.25
    clipped_age = np.minimum(age_years, 15.0)
    age_log = np.log1p(age_years)
    thresholds = np.column_stack([
        age_years <= threshold for threshold in (0.25, 0.5, 1.0, 2.0, 5.0)
    ]).astype(np.float64)
    features = np.column_stack([
        clipped_age,
        age_log,
        np.sqrt(age_years),
        age_missing,
        thresholds,
    ])
    if mode == "age_sex":
        features = np.column_stack([features, sex, sex * age_log])
    elif mode != "age_only":
        raise ValueError(mode)
    return features


def aligned_probabilities(scaler, model, features, class_count):
    raw = model.predict_proba(scaler.transform(features))
    result = np.zeros((len(features), class_count), dtype=np.float64)
    for column, class_index in enumerate(model.classes_):
        result[:, int(class_index)] = raw[:, column]
    return normalize(result)


def fit_prior(mode, c_value, sex, age, y):
    features = metadata_features(sex, age, mode)
    scaler = StandardScaler()
    model = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    ).fit(scaler.fit_transform(features), y)
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float64) + 1.0
    return scaler, model, counts / counts.sum()


def select_c_train_only(mode, sex, age, y, seed):
    features = metadata_features(sex, age, mode)
    splitter = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=3, random_state=seed
    )
    records = []
    for c_value in C_VALUES:
        total = 0.0
        split_losses = []
        for train_index, valid_index in splitter.split(features, y):
            scaler = StandardScaler()
            model = LogisticRegression(
                C=c_value,
                solver="lbfgs",
                max_iter=2000,
                random_state=0,
            ).fit(scaler.fit_transform(features[train_index]), y[train_index])
            probabilities = aligned_probabilities(
                scaler, model, features[valid_index], len(LABELS)
            )
            loss = float(log_loss(
                y[valid_index], probabilities, labels=np.arange(len(LABELS))
            ))
            split_losses.append(loss)
            total += loss
        records.append({
            "C": c_value,
            "mean_nll": total / len(split_losses),
            "split_nll_std": float(np.std(split_losses, ddof=1)),
            "splits": len(split_losses),
        })
    selected = min(records, key=lambda item: (item["mean_nll"], item["C"]))
    return float(selected["C"]), records


def effective_weight(acoustic, strength, adaptation):
    if adaptation == "fixed":
        return np.full(len(acoustic), float(strength), dtype=np.float64)
    if adaptation == "entropy":
        entropy = -np.sum(
            acoustic * np.log(np.clip(acoustic, 1e-12, 1.0)), axis=1
        )
        return float(strength) * entropy / math.log(acoustic.shape[1])
    raise ValueError(adaptation)


def apply_prior(config, fitted, acoustic, sex, age):
    acoustic = normalize(acoustic)
    scaler, model, class_prior = fitted
    metadata_prob = aligned_probabilities(
        scaler,
        model,
        metadata_features(sex, age, config["mode"]),
        acoustic.shape[1],
    )
    weight = effective_weight(acoustic, config["strength"], config["adaptation"])
    if config["scope"] == "all_classes":
        logits = (
            np.log(np.clip(acoustic, 1e-12, 1.0))
            + weight[:, None]
            * (
                np.log(np.clip(metadata_prob, 1e-12, 1.0))
                - np.log(np.clip(class_prior, 1e-12, 1.0))[None, :]
            )
        )
        return normalize(softmax(logits, axis=1)), weight
    if config["scope"] != "abnormal_conditional":
        raise ValueError(config["scope"])
    normal_index = LABELS.index("NORMAL")
    abnormal = [index for index in range(len(LABELS)) if index != normal_index]
    acoustic_conditional = normalize(acoustic[:, abnormal])
    metadata_conditional = normalize(metadata_prob[:, abnormal])
    prior_conditional = normalize(class_prior[abnormal][None, :])[0]
    conditional_logits = (
        np.log(np.clip(acoustic_conditional, 1e-12, 1.0))
        + weight[:, None]
        * (
            np.log(np.clip(metadata_conditional, 1e-12, 1.0))
            - np.log(np.clip(prior_conditional, 1e-12, 1.0))[None, :]
        )
    )
    corrected_conditional = softmax(conditional_logits, axis=1)
    result = np.zeros_like(acoustic)
    result[:, normal_index] = acoustic[:, normal_index]
    result[:, abnormal] = (
        1.0 - acoustic[:, normal_index, None]
    ) * corrected_conditional
    result = normalize(result)
    if not np.allclose(
        result[:, normal_index], acoustic[:, normal_index], atol=1e-12
    ):
        raise RuntimeError("Stage26 abnormal correction changed P(NORMAL)")
    return result, weight


def score(y, probabilities):
    minority = np.asarray([LABELS.index(name) for name in MINORITY])
    return detailed_metrics(
        y, normalize(probabilities), minority, LABELS.index("NORMAL")
    )


def candidate_library(selected_c):
    candidates = []
    for mode in MODES:
        for scope in SCOPES:
            for adaptation in ADAPTATIONS:
                for strength in STRENGTHS:
                    candidates.append({
                        "mode": mode,
                        "C": selected_c[mode],
                        "scope": scope,
                        "adaptation": adaptation,
                        "strength": strength,
                    })
    return candidates


def choose_candidate(baseline, records, validation_count):
    one_patient = 1.0 / validation_count
    for item in records:
        current = item["validation_metrics"]
        item["balanced_score"] = 0.5 * (
            current["macro_f1"] + current["minority_f1"]
        )
        item["delta"] = {
            metric: current[metric] - baseline[metric]
            for metric in (
                "accuracy", "macro_f1", "minority_recall", "minority_f1",
                "normal_recall", "nll",
            )
        }
        item["accuracy_gate"] = bool(
            item["delta"]["accuracy"] >= 0.005
            and item["delta"]["macro_f1"] >= -0.005
            and item["delta"]["minority_f1"] >= -0.01
            and item["delta"]["normal_recall"] >= -0.02
            and item["delta"]["nll"] <= 0.02
        )
        baseline_balanced = 0.5 * (
            baseline["macro_f1"] + baseline["minority_f1"]
        )
        item["balanced_gate"] = bool(
            item["balanced_score"] - baseline_balanced >= 0.005
            and item["delta"]["accuracy"] >= -one_patient - 1e-12
            and item["delta"]["normal_recall"] >= -0.02
            and item["delta"]["nll"] <= 0.02
        )
        item["calibration_gate"] = bool(
            item["delta"]["nll"] <= -0.005
            and item["delta"]["accuracy"] >= -one_patient - 1e-12
            and item["delta"]["macro_f1"] >= -0.005
            and item["delta"]["normal_recall"] >= -0.02
        )

    accuracy = [item for item in records if item["accuracy_gate"]]
    if accuracy:
        selected = max(accuracy, key=lambda item: (
            item["validation_metrics"]["accuracy"],
            item["validation_metrics"]["macro_f1"],
            item["validation_metrics"]["minority_f1"],
            -item["validation_metrics"]["nll"],
            -item["complexity"],
        ))
        return "accuracy", selected
    balanced = [item for item in records if item["balanced_gate"]]
    if balanced:
        selected = max(balanced, key=lambda item: (
            item["balanced_score"],
            item["validation_metrics"]["accuracy"],
            -item["validation_metrics"]["nll"],
            -item["complexity"],
        ))
        return "balanced_f1", selected
    calibration = [item for item in records if item["calibration_gate"]]
    if calibration:
        selected = min(calibration, key=lambda item: (
            item["validation_metrics"]["nll"],
            -item["validation_metrics"]["accuracy"],
            item["complexity"],
        ))
        return "calibration", selected
    return "acoustic_only", None


def main():
    args = parse_args()
    validation_root = Path(args.validation_root).resolve()
    if list(validation_root.glob("**/test_file_probs.csv")):
        raise RuntimeError("Stage26 selection root contains test probabilities")
    metadata_path = Path(args.metadata_csv).resolve()
    metadata = load_metadata(metadata_path)
    split_dir = (
        Path(args.split_root).resolve()
        / f"splits_stage24_confirm_seed{args.split_seed}"
    )
    folds = []
    for fold in range(5):
        split_path = split_dir / f"cv5_tvt_fold{fold}.json"
        with split_path.open("r", encoding="utf-8") as handle:
            split = json.load(handle)
        train_ids = split["train_files"]
        val_ids, val_y, acoustic_val = read_acoustic(
            validation_root, args.split_seed, fold, "val"
        )
        if set(val_ids) != set(split["val_files"]):
            raise RuntimeError(f"Stage26 validation manifest mismatch fold={fold}")
        train_sex, train_age, train_y = metadata_arrays(train_ids, metadata)
        val_sex, val_age, metadata_val_y = metadata_arrays(val_ids, metadata)
        if not np.array_equal(val_y, metadata_val_y):
            raise RuntimeError(f"Stage26 metadata label mismatch fold={fold}")

        selected_c = {}
        c_search = {}
        fitted = {}
        for mode_index, mode in enumerate(MODES):
            selected_c[mode], c_search[mode] = select_c_train_only(
                mode,
                train_sex,
                train_age,
                train_y,
                args.cv_seed + fold * 10 + mode_index,
            )
            fitted[mode] = fit_prior(
                mode, selected_c[mode], train_sex, train_age, train_y
            )

        records = []
        for config in candidate_library(selected_c):
            probabilities, weight = apply_prior(
                config,
                fitted[config["mode"]],
                acoustic_val,
                val_sex,
                val_age,
            )
            complexity = (
                int(config["mode"] == "age_sex")
                + int(config["scope"] == "all_classes")
                + int(config["adaptation"] == "entropy")
            )
            records.append({
                "config": config,
                "complexity": complexity,
                "mean_effective_weight": float(weight.mean()),
                "validation_metrics": score(val_y, probabilities),
            })
        baseline = score(val_y, acoustic_val)
        branch, selected = choose_candidate(baseline, records, len(val_y))
        selected_config = None if selected is None else selected["config"]
        folds.append({
            "fold": fold,
            "train_count": len(train_ids),
            "validation_count": len(val_ids),
            "train_validation_patient_overlap": len(set(train_ids) & set(val_ids)),
            "validation_files_match_fixed_split": True,
            "train_file_ids_sha256": canonical_hash(train_ids),
            "validation_file_ids_sha256": canonical_hash(val_ids),
            "metadata_c_selection": c_search,
            "selected_c": selected_c,
            "acoustic_validation_metrics": baseline,
            "locked_branch": branch,
            "locked_candidate": selected_config,
            "locked_validation_metrics": (
                baseline if selected is None else selected["validation_metrics"]
            ),
            "top_candidates": sorted(
                records,
                key=lambda item: (
                    item["balanced_score"],
                    item["validation_metrics"]["accuracy"],
                    -item["validation_metrics"]["nll"],
                ),
                reverse=True,
            )[:10],
        })
        print(
            f"[LOCK] fold={fold} branch={branch} candidate={selected_config} "
            f"acoustic_acc={baseline['accuracy']:.4f} "
            f"locked_acc={folds[-1]['locked_validation_metrics']['accuracy']:.4f}",
            flush=True,
        )

    payload = {
        "tag": "stage26_train_fitted_demographic_prior_selection",
        "selection_only": True,
        "selection_root": str(validation_root),
        "selection_root_contains_test_probabilities": False,
        "outer_test_probabilities_loaded": False,
        "test_used_for_selection": False,
        "split_seed": args.split_seed,
        "metadata_csv": str(metadata_path),
        "metadata_csv_sha256": file_sha256(metadata_path),
        "metadata_features": ["age_days", "sex01"],
        "acoustic_components": list(CURRICULUM),
        "candidate_grid": {
            "modes": list(MODES),
            "C_train_only_cv": list(C_VALUES),
            "scopes": list(SCOPES),
            "adaptations": list(ADAPTATIONS),
            "strengths": list(STRENGTHS),
        },
        "selection_rule": (
            "Choose demographic LR C by repeated five-fold CV on outer-train only. "
            "Fit each prior on outer-train and select correction on same-fold validation "
            "with branch priority accuracy, balanced F1, calibration, then acoustic "
            "fallback. Accuracy/normal-recall/NLL guardrails are fixed in code. Refit "
            "the selected prior on train+validation only after this lock."
        ),
        "folds": folds,
    }
    payload["selection_sha256"] = canonical_hash(payload)
    output = Path(args.output_lock).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[LOCKED] {output} sha256={payload['selection_sha256']}", flush=True)


if __name__ == "__main__":
    main()
