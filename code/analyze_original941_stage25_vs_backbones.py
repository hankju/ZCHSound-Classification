#!/usr/bin/env python
"""Run matched-fold statistics for Stage25 and seven single backbones."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import (
    binomtest,
    f as f_distribution,
    friedmanchisquare,
    shapiro,
    studentized_range,
)
from sklearn.metrics import accuracy_score, f1_score


LABELS = ("ASD", "NORMAL", "PDA", "PFO", "VSD")
METHOD_FILES = (
    ("ConvNeXt-Tiny", "convnext_oof_predictions.csv"),
    ("Swin-Tiny", "swin_log42_oof_predictions.csv"),
    ("DenseNet121", "densenet121_oof_predictions.csv"),
    ("EfficientNet-B3", "effb3_oof_predictions.csv"),
    ("ViT-Small/16", "vit_small_oof_predictions.csv"),
    ("RegNetY-008", "regnety008_oof_predictions.csv"),
    ("ResNet50", "resnet50_oof_predictions.csv"),
)
NORMALIZED_METHOD_FILES = (
    ("ConvNeXt-Tiny", "convnext_tiny_oof_predictions.csv"),
    ("Swin-Tiny", "swin_tiny_oof_predictions.csv"),
    ("DenseNet121", "densenet121_oof_predictions.csv"),
    ("EfficientNet-B3", "efficientnet_b3_oof_predictions.csv"),
    ("ViT-Small/16", "vit_small_16_oof_predictions.csv"),
    ("RegNetY-008", "regnety_008_oof_predictions.csv"),
    ("ResNet50", "resnet50_oof_predictions.csv"),
)
FORMAL_METHOD = "Stage25 fixed curriculum"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage25-predictions")
    parser.add_argument("--backbone-predictions-dir")
    parser.add_argument(
        "--normalized-predictions-dir",
        help="Use the nine publication CSVs instead of historical raw exports.",
    )
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def expected_fold_map(split_dir):
    mapping = {}
    for fold in range(5):
        payload = json.loads(
            (split_dir / f"cv5_tvt_fold{fold}.json").read_text(encoding="utf-8")
        )
        for item in payload["test_files"]:
            recording_id = Path(item).stem
            if recording_id in mapping:
                raise RuntimeError(f"duplicate outer-test ID: {recording_id}")
            mapping[recording_id] = (fold, item.split("/", 1)[0])
    if len(mapping) != 941:
        raise RuntimeError(f"expected 941 OOF IDs, found {len(mapping)}")
    return mapping


def build_stage25_rows(path, fold_map):
    rows = read_csv(path)
    normalized = []
    for row in rows:
        recording_id = Path(row["file_id"]).stem
        if recording_id not in fold_map:
            raise RuntimeError(f"Stage25 ID not in fixed split: {recording_id}")
        fold, expected_label = fold_map[recording_id]
        probabilities = np.asarray(
            [float(row[f"curriculum_prob_{label}"]) for label in LABELS],
            dtype=np.float64,
        )
        if not np.isfinite(probabilities).all() or not np.isclose(
            probabilities.sum(), 1.0, atol=1e-5
        ):
            raise RuntimeError(f"invalid Stage25 probabilities: {recording_id}")
        true_label = row["true_label"]
        if int(row["fold"]) != fold or true_label != expected_label:
            raise RuntimeError(f"Stage25 split/label mismatch: {recording_id}")
        pred_label = LABELS[int(probabilities.argmax())]
        normalized.append(
            {
                "recording_id": recording_id,
                "fold": fold,
                "true_label": true_label,
                "pred_label": pred_label,
                **{
                    f"prob_{label}": repr(float(value))
                    for label, value in zip(LABELS, probabilities)
                },
            }
        )
    if len(normalized) != 941 or len({row["recording_id"] for row in normalized}) != 941:
        raise RuntimeError("Stage25 OOF predictions are not 941 unique recordings")
    return normalized


def normalize_backbone_rows(path, fold_map):
    rows = read_csv(path)
    normalized = []
    for row in rows:
        recording_id = row["recording_id"]
        if recording_id not in fold_map:
            raise RuntimeError(f"backbone ID not in fixed split: {recording_id}")
        fold, expected_label = fold_map[recording_id]
        if int(row["fold"]) != fold or row["true_label"] != expected_label:
            raise RuntimeError(f"backbone split/label mismatch: {recording_id}")
        probabilities = np.asarray(
            [float(row[f"prob_{label}"]) for label in LABELS], dtype=np.float64
        )
        if LABELS[int(probabilities.argmax())] != row["pred_label"]:
            raise RuntimeError(f"backbone pred/probability mismatch: {recording_id}")
        normalized.append(row)
    if len(normalized) != 941 or len({row["recording_id"] for row in normalized}) != 941:
        raise RuntimeError(f"backbone OOF coverage failure: {path}")
    return normalized


def metric_rows(method_rows):
    fold_results = []
    pooled_results = []
    for method, rows in method_rows.items():
        ordered = sorted(rows, key=lambda row: row["recording_id"])
        y_true = [row["true_label"] for row in ordered]
        y_pred = [row["pred_label"] for row in ordered]
        pooled_results.append(
            {
                "method": method,
                "n": len(rows),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(
                    y_true, y_pred, labels=LABELS, average="macro", zero_division=0
                ),
            }
        )
        for fold in range(5):
            subset = [row for row in rows if int(row["fold"]) == fold]
            fold_true = [row["true_label"] for row in subset]
            fold_pred = [row["pred_label"] for row in subset]
            fold_results.append(
                {
                    "method": method,
                    "fold": fold,
                    "n": len(subset),
                    "accuracy": accuracy_score(fold_true, fold_pred),
                    "macro_f1": f1_score(
                        fold_true,
                        fold_pred,
                        labels=LABELS,
                        average="macro",
                        zero_division=0,
                    ),
                }
            )
    return fold_results, pooled_results


def matrix_from_folds(fold_rows, methods, metric):
    lookup = {(row["fold"], row["method"]): float(row[metric]) for row in fold_rows}
    return np.asarray(
        [[lookup[(fold, method)] for method in methods] for fold in range(5)],
        dtype=np.float64,
    )


def rcbd_anova(matrix):
    blocks, treatments = matrix.shape
    grand = matrix.mean()
    treatment_means = matrix.mean(axis=0)
    block_means = matrix.mean(axis=1)
    ss_total = float(((matrix - grand) ** 2).sum())
    ss_method = float(blocks * ((treatment_means - grand) ** 2).sum())
    ss_fold = float(treatments * ((block_means - grand) ** 2).sum())
    ss_error = max(0.0, ss_total - ss_method - ss_fold)
    df_method = treatments - 1
    df_fold = blocks - 1
    df_error = df_method * df_fold
    ms_method = ss_method / df_method
    ms_fold = ss_fold / df_fold
    ms_error = ss_error / df_error
    f_method = ms_method / ms_error
    f_fold = ms_fold / ms_error
    rows = [
        {
            "Source": "Fold (block)",
            "df": df_fold,
            "SS": ss_fold,
            "MS": ms_fold,
            "F": f_fold,
            "p": f_distribution.sf(f_fold, df_fold, df_error),
        },
        {
            "Source": "Method",
            "df": df_method,
            "SS": ss_method,
            "MS": ms_method,
            "F": f_method,
            "p": f_distribution.sf(f_method, df_method, df_error),
        },
        {
            "Source": "Residual",
            "df": df_error,
            "SS": ss_error,
            "MS": ms_error,
            "F": "",
            "p": "",
        },
    ]
    return rows, ms_error, df_error


def compact_letters(methods, means, significant):
    columns = [set(methods)]
    for index, left in enumerate(methods):
        for right in methods[index + 1 :]:
            if not significant.get(frozenset((left, right)), False):
                continue
            updated = []
            for column in columns:
                if left in column and right in column:
                    updated.extend((column - {left}, column - {right}))
                else:
                    updated.append(column)
            unique = []
            for column in updated:
                if column and column not in unique:
                    unique.append(column)
            columns = [
                column
                for column in unique
                if not any(column < other for other in unique)
            ]
    rank = {method: index for index, method in enumerate(methods)}
    columns.sort(key=lambda column: (min(rank[item] for item in column), -len(column)))
    alphabet = [chr(ord("a") + index) for index in range(26)]
    if len(columns) > len(alphabet):
        raise RuntimeError("too many compact-letter groups")
    return {
        method: "".join(alphabet[index] for index, column in enumerate(columns) if method in column)
        for method in methods
    }


def duncan_test(matrix, methods, mse, df_error, alpha):
    blocks = matrix.shape[0]
    means = {method: float(matrix[:, index].mean()) for index, method in enumerate(methods)}
    ordered = sorted(methods, key=lambda method: means[method], reverse=True)
    standard_error = math.sqrt(mse / blocks)
    pairs = []
    significant = {}
    for high_index, high in enumerate(ordered):
        for low_index in range(high_index + 1, len(ordered)):
            low = ordered[low_index]
            range_size = low_index - high_index + 1
            alpha_range = 1.0 - (1.0 - alpha) ** (range_size - 1)
            critical_q = studentized_range.ppf(
                1.0 - alpha_range, range_size, df_error
            )
            critical_range = critical_q * standard_error
            difference = means[high] - means[low]
            reject = bool(difference > critical_range)
            significant[frozenset((high, low))] = reject
            pairs.append(
                {
                    "higher_method": high,
                    "lower_method": low,
                    "ordered_range": range_size,
                    "mean_difference": difference,
                    "alpha_for_range": alpha_range,
                    "critical_q": critical_q,
                    "critical_range": critical_range,
                    "reject": reject,
                }
            )
    letters = compact_letters(ordered, means, significant)
    groups = [
        {"Method": method, "Mean (%)": means[method] * 100.0, "Group": letters[method]}
        for method in ordered
    ]
    return groups, pairs


def tukey_test(matrix, methods, mse, df_error, alpha):
    blocks, treatment_count = matrix.shape
    means = {method: float(matrix[:, index].mean()) for index, method in enumerate(methods)}
    standard_error = math.sqrt(mse / blocks)
    critical_q = studentized_range.ppf(1.0 - alpha, treatment_count, df_error)
    rows = []
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1 :]:
            difference = means[left] - means[right]
            q_statistic = abs(difference) / standard_error
            half_width = critical_q * standard_error
            rows.append(
                {
                    "method_1": left,
                    "method_2": right,
                    "mean_difference": difference,
                    "ci_low": difference - half_width,
                    "ci_high": difference + half_width,
                    "q": q_statistic,
                    "p_adjusted": studentized_range.sf(
                        q_statistic, treatment_count, df_error
                    ),
                    "reject": bool(abs(difference) > half_width),
                }
            )
    return rows


def shapiro_tests(matrix, methods, metric):
    rows = []
    for index, method in enumerate(methods):
        statistic, p_value = shapiro(matrix[:, index])
        rows.append(
            {
                "metric": metric,
                "method": method,
                "n_folds": matrix.shape[0],
                "statistic": statistic,
                "p": p_value,
            }
        )
    return rows


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def mcnemar_tests(method_rows):
    formal = {row["recording_id"]: row for row in method_rows[FORMAL_METHOD]}
    results = []
    for method, _ in METHOD_FILES:
        candidate = {row["recording_id"]: row for row in method_rows[method]}
        if set(candidate) != set(formal):
            raise RuntimeError(f"paired ID mismatch for {method}")
        formal_only = 0
        backbone_only = 0
        for recording_id, formal_row in formal.items():
            candidate_row = candidate[recording_id]
            formal_correct = formal_row["pred_label"] == formal_row["true_label"]
            candidate_correct = candidate_row["pred_label"] == candidate_row["true_label"]
            formal_only += int(formal_correct and not candidate_correct)
            backbone_only += int(candidate_correct and not formal_correct)
        discordant = formal_only + backbone_only
        p_value = (
            binomtest(formal_only, discordant, p=0.5, alternative="two-sided").pvalue
            if discordant
            else 1.0
        )
        results.append(
            {
                "formal_method": FORMAL_METHOD,
                "backbone": method,
                "formal_only_correct": formal_only,
                "backbone_only_correct": backbone_only,
                "discordant": discordant,
                "exact_two_sided_p": p_value,
            }
        )
    adjusted = holm_adjust([row["exact_two_sided_p"] for row in results])
    for row, adjusted_p in zip(results, adjusted):
        row["holm_adjusted_p"] = adjusted_p
        row["reject_holm_0_05"] = bool(adjusted_p < 0.05)
    return results


def main():
    args = parse_args()
    normalized_dir = (
        Path(args.normalized_predictions_dir).resolve()
        if args.normalized_predictions_dir
        else None
    )
    if normalized_dir is None and (
        not args.stage25_predictions or not args.backbone_predictions_dir
    ):
        raise ValueError(
            "provide --normalized-predictions-dir, or both historical source arguments"
        )
    stage25_path = (
        normalized_dir / "stage25_fixed_curriculum_oof_predictions.csv"
        if normalized_dir is not None
        else Path(args.stage25_predictions).resolve()
    )
    backbone_dir = (
        normalized_dir
        if normalized_dir is not None
        else Path(args.backbone_predictions_dir).resolve()
    )
    split_dir = Path(args.split_dir).resolve()
    output = Path(args.output_dir).resolve()
    predictions_dir = output / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    fold_map = expected_fold_map(split_dir)
    stage25_rows = (
        normalize_backbone_rows(stage25_path, fold_map)
        if normalized_dir is not None
        else build_stage25_rows(stage25_path, fold_map)
    )
    prediction_fields = [
        "recording_id",
        "fold",
        "true_label",
        "pred_label",
        *[f"prob_{label}" for label in LABELS],
    ]
    stage25_output = predictions_dir / "stage25_fixed_curriculum_oof_predictions.csv"
    write_csv(stage25_output, stage25_rows, prediction_fields)

    method_rows = {FORMAL_METHOD: stage25_rows}
    source_paths = {FORMAL_METHOD: stage25_path}
    method_files = NORMALIZED_METHOD_FILES if normalized_dir is not None else METHOD_FILES
    for method, filename in method_files:
        source = backbone_dir / filename
        method_rows[method] = normalize_backbone_rows(source, fold_map)
        source_paths[method] = source
    methods = [FORMAL_METHOD, *[method for method, _ in method_files]]

    reference_ids = {
        method: {row["recording_id"] for row in rows}
        for method, rows in method_rows.items()
    }
    if any(ids != reference_ids[FORMAL_METHOD] for ids in reference_ids.values()):
        raise RuntimeError("the eight methods do not contain identical recording IDs")

    fold_rows, pooled_rows = metric_rows(method_rows)
    write_csv(output / "eight_method_fold_metrics.csv", fold_rows)
    write_csv(output / "eight_method_pooled_metrics.csv", pooled_rows)

    shapiro_rows = []
    summary = {"alpha": args.alpha, "methods": methods, "metrics": {}}
    for metric in ("accuracy", "macro_f1"):
        matrix = matrix_from_folds(fold_rows, methods, metric)
        matrix_rows = []
        for fold in range(5):
            matrix_rows.append(
                {"fold": fold, **{method: matrix[fold, index] for index, method in enumerate(methods)}}
            )
        write_csv(output / f"{metric}_fold_matrix.csv", matrix_rows)

        anova_rows, mse, df_error = rcbd_anova(matrix)
        write_csv(output / f"anova_{metric}.csv", anova_rows)
        duncan_groups, duncan_pairs = duncan_test(
            matrix, methods, mse, df_error, args.alpha
        )
        write_csv(output / f"duncan_{metric}_groups.csv", duncan_groups)
        write_csv(output / f"duncan_{metric}_pairwise.csv", duncan_pairs)
        tukey_rows = tukey_test(matrix, methods, mse, df_error, args.alpha)
        write_csv(output / f"tukey_hsd_{metric}.csv", tukey_rows)
        friedman_stat, friedman_p = friedmanchisquare(
            *[matrix[:, index] for index in range(matrix.shape[1])]
        )
        friedman = {
            "metric": metric,
            "n_blocks": matrix.shape[0],
            "n_methods": matrix.shape[1],
            "df": matrix.shape[1] - 1,
            "statistic": float(friedman_stat),
            "p": float(friedman_p),
            "kendalls_w": float(
                friedman_stat / (matrix.shape[0] * (matrix.shape[1] - 1))
            ),
        }
        write_csv(output / f"friedman_{metric}.csv", [friedman])
        shapiro_rows.extend(shapiro_tests(matrix, methods, metric))
        summary["metrics"][metric] = {
            "anova": anova_rows,
            "duncan_groups": duncan_groups,
            "friedman": friedman,
        }

    write_csv(output / "shapiro_wilk_by_method.csv", shapiro_rows)
    mcnemar_rows = mcnemar_tests(method_rows)
    write_csv(output / "mcnemar_stage25_vs_backbones.csv", mcnemar_rows)

    summary["mcnemar"] = mcnemar_rows
    summary["input_provenance"] = {
        method: {"path": str(path), "sha256": sha256(path)}
        for method, path in source_paths.items()
    }
    summary["fixed_split_manifest"] = {
        "path": str(split_dir / "cv5_manifest.json"),
        "sha256": sha256(split_dir / "cv5_manifest.json"),
    }
    summary["formal_prediction_output"] = {
        "path": str(stage25_output),
        "sha256": sha256(stage25_output),
    }
    (output / "statistical_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DONE] matched methods={len(methods)} folds=5 recordings=941 output={output}")


if __name__ == "__main__":
    main()
