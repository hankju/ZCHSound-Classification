#!/usr/bin/env python3
"""Generate recording-level confusion, bootstrap, and McNemar analyses."""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path


LABELS = ("ASD", "NORMAL", "PDA", "PFO", "VSD")
METHOD_SOURCES = (
    ("Stage25 fixed curriculum", "stage25_fixed_curriculum", "stage25", None),
    ("Stage26 multimodal", "stage26_multimodal", "stage26", None),
    ("ConvNeXt-Tiny", "convnext_tiny", "backbone", "convnext_oof_predictions.csv"),
    ("Swin-Tiny", "swin_tiny", "backbone", "swin_log42_oof_predictions.csv"),
    ("DenseNet121", "densenet121", "backbone", "densenet121_oof_predictions.csv"),
    ("EfficientNet-B3", "efficientnet_b3", "backbone", "effb3_oof_predictions.csv"),
    ("ViT-Small/16", "vit_small_16", "backbone", "vit_small_oof_predictions.csv"),
    ("RegNetY-008", "regnety_008", "backbone", "regnety008_oof_predictions.csv"),
    ("ResNet50", "resnet50", "backbone", "resnet50_oof_predictions.csv"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage25-predictions", required=True)
    parser.add_argument("--stage26-predictions", required=True)
    parser.add_argument("--backbone-predictions-dir", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_fold_map(split_dir):
    mapping = {}
    for fold in range(5):
        payload = json.loads(
            (split_dir / f"cv5_tvt_fold{fold}.json").read_text(encoding="utf-8")
        )
        for item in payload["test_files"]:
            recording_id = Path(item).stem
            if recording_id in mapping:
                raise RuntimeError(f"duplicate outer-test recording ID: {recording_id}")
            mapping[recording_id] = (fold, item.split("/", 1)[0])
    if len(mapping) != 941:
        raise RuntimeError(f"expected 941 fixed-split recordings, found {len(mapping)}")
    return mapping


def source_columns(source_type):
    if source_type == "stage26":
        return (
            "file_id",
            "multimodal_pred_label",
            tuple(f"multimodal_prob_{label}" for label in LABELS),
        )
    return (
        "recording_id",
        "pred_label",
        tuple(f"prob_{label}" for label in LABELS),
    )


def normalize_rows(path, source_type, fold_map):
    rows = read_csv(path)
    id_field, pred_field, probability_fields = source_columns(source_type)
    normalized = []
    for row in rows:
        recording_id = Path(row[id_field]).stem
        if recording_id not in fold_map:
            raise RuntimeError(f"recording ID not in fixed split: {recording_id}")
        fold, expected_label = fold_map[recording_id]
        true_label = row["true_label"]
        probabilities = [float(row[field]) for field in probability_fields]
        if int(row["fold"]) != fold or true_label != expected_label:
            raise RuntimeError(f"split or label mismatch: {recording_id}")
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities):
            raise RuntimeError(f"invalid probability value: {recording_id}")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
            raise RuntimeError(f"probabilities do not sum to one: {recording_id}")
        pred_label = LABELS[max(range(len(LABELS)), key=probabilities.__getitem__)]
        if pred_label != row[pred_field]:
            raise RuntimeError(f"prediction does not match probabilities: {recording_id}")
        normalized.append(
            {
                "recording_id": recording_id,
                "fold": fold,
                "true_label": true_label,
                "pred_label": pred_label,
                **{
                    f"prob_{label}": repr(value)
                    for label, value in zip(LABELS, probabilities)
                },
            }
        )
    normalized.sort(key=lambda row: row["recording_id"])
    ids = [row["recording_id"] for row in normalized]
    if len(normalized) != 941 or len(set(ids)) != 941:
        raise RuntimeError(f"OOF coverage failure: {path}")
    return normalized


def confusion_matrix(y_true, y_pred):
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[label_to_index[true_label]][label_to_index[pred_label]] += 1
    return matrix


def indexed_confusion_matrix(y_true, y_pred):
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for true_index, pred_index in zip(y_true, y_pred):
        matrix[true_index][pred_index] += 1
    return matrix


def metrics_from_confusion(matrix):
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(len(LABELS)))
    class_f1 = []
    for index in range(len(LABELS)):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(LABELS)) if row != index)
        false_negative = sum(matrix[index][column] for column in range(len(LABELS)) if column != index)
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1.append(2 * true_positive / denominator if denominator else 0.0)
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(class_f1) / len(class_f1),
        **{f"f1_{label}": class_f1[index] for index, label in enumerate(LABELS)},
    }


def percentile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_intervals(method_arrays, resamples, seed, alpha):
    methods = list(method_arrays)
    n = len(next(iter(method_arrays.values()))[0])
    rng = random.Random(seed)
    metric_names = ("accuracy", "macro_f1", *[f"f1_{label}" for label in LABELS])
    distributions = {
        method: {metric: [] for metric in metric_names} for method in methods
    }
    population = range(n)
    for iteration in range(resamples):
        sample = rng.choices(population, k=n)
        for method in methods:
            y_true, y_pred = method_arrays[method]
            matrix = [[0 for _ in LABELS] for _ in LABELS]
            for index in sample:
                matrix[y_true[index]][y_pred[index]] += 1
            metrics = metrics_from_confusion(matrix)
            for metric in metric_names:
                distributions[method][metric].append(metrics[metric])
        if (iteration + 1) % 1000 == 0:
            print(f"[BOOTSTRAP] {iteration + 1}/{resamples}", flush=True)

    rows = []
    for method in methods:
        y_true, y_pred = method_arrays[method]
        point = metrics_from_confusion(indexed_confusion_matrix(y_true, y_pred))
        for metric in metric_names:
            values = distributions[method][metric]
            rows.append(
                {
                    "method": method,
                    "n": n,
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_level": 1.0 - alpha,
                    "ci_method": "recording-level percentile bootstrap",
                    "n_resamples": resamples,
                    "seed": seed,
                    "ci_lower": percentile(values, alpha / 2.0),
                    "ci_upper": percentile(values, 1.0 - alpha / 2.0),
                }
            )
    return rows


def exact_mcnemar_p(method_1_only, method_2_only):
    discordant = method_1_only + method_2_only
    if not discordant:
        return 1.0
    tail = min(method_1_only, method_2_only)
    cumulative = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * cumulative / (2 ** discordant))


def holm_adjust(rows, alpha):
    order = sorted(range(len(rows)), key=lambda index: rows[index]["exact_two_sided_p"])
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * rows[index]["exact_two_sided_p"])
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running
        rows[index][f"reject_holm_{alpha:g}"] = running < alpha


def pairwise_mcnemar(method_rows, alpha):
    methods = list(method_rows)
    correct = {
        method: [row["pred_label"] == row["true_label"] for row in rows]
        for method, rows in method_rows.items()
    }
    results = []
    for first_index, method_1 in enumerate(methods):
        for method_2 in methods[first_index + 1 :]:
            method_1_only = sum(
                first and not second
                for first, second in zip(correct[method_1], correct[method_2])
            )
            method_2_only = sum(
                second and not first
                for first, second in zip(correct[method_1], correct[method_2])
            )
            results.append(
                {
                    "method_1": method_1,
                    "method_2": method_2,
                    "method_1_only_correct": method_1_only,
                    "method_2_only_correct": method_2_only,
                    "discordant": method_1_only + method_2_only,
                    "exact_two_sided_p": exact_mcnemar_p(method_1_only, method_2_only),
                }
            )
    holm_adjust(results, alpha)
    return results


def main():
    args = parse_args()
    stage25_path = Path(args.stage25_predictions).resolve()
    stage26_path = Path(args.stage26_predictions).resolve()
    backbone_dir = Path(args.backbone_predictions_dir).resolve()
    split_dir = Path(args.split_dir).resolve()
    output = Path(args.output_dir).resolve()
    predictions_dir = output / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    fold_map = expected_fold_map(split_dir)
    source_paths = {}
    method_rows = {}
    method_slugs = {}
    for method, slug, source_type, filename in METHOD_SOURCES:
        if source_type == "stage25":
            source = stage25_path
        elif source_type == "stage26":
            source = stage26_path
        else:
            source = backbone_dir / filename
        rows = normalize_rows(source, source_type, fold_map)
        source_paths[method] = source
        method_rows[method] = rows
        method_slugs[method] = slug

    reference = [row["recording_id"] for row in method_rows[next(iter(method_rows))]]
    reference_pairs = [(row["fold"], row["true_label"]) for row in method_rows[next(iter(method_rows))]]
    for method, rows in method_rows.items():
        if [row["recording_id"] for row in rows] != reference:
            raise RuntimeError(f"paired recording IDs do not align: {method}")
        if [(row["fold"], row["true_label"]) for row in rows] != reference_pairs:
            raise RuntimeError(f"paired fold/label values do not align: {method}")

    prediction_fields = (
        "recording_id",
        "fold",
        "true_label",
        "pred_label",
        *[f"prob_{label}" for label in LABELS],
    )
    normalized_outputs = {}
    for method, rows in method_rows.items():
        path = predictions_dir / f"{method_slugs[method]}_oof_predictions.csv"
        write_csv(path, rows, prediction_fields)
        normalized_outputs[method] = path

    confusion_long = []
    confusion_wide = []
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    method_arrays = {}
    for method, rows in method_rows.items():
        y_true_labels = [row["true_label"] for row in rows]
        y_pred_labels = [row["pred_label"] for row in rows]
        matrix = confusion_matrix(y_true_labels, y_pred_labels)
        method_arrays[method] = (
            [label_to_index[value] for value in y_true_labels],
            [label_to_index[value] for value in y_pred_labels],
        )
        for true_index, true_label in enumerate(LABELS):
            confusion_wide.append(
                {
                    "method": method,
                    "true_label": true_label,
                    **{
                        f"pred_{pred_label}": matrix[true_index][pred_index]
                        for pred_index, pred_label in enumerate(LABELS)
                    },
                    "support": sum(matrix[true_index]),
                }
            )
            for pred_index, pred_label in enumerate(LABELS):
                confusion_long.append(
                    {
                        "method": method,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": matrix[true_index][pred_index],
                    }
                )
    write_csv(output / "confusion_matrices_long.csv", confusion_long)
    write_csv(output / "confusion_matrices_wide.csv", confusion_wide)

    bootstrap_rows = bootstrap_intervals(
        method_arrays,
        args.bootstrap_resamples,
        args.bootstrap_seed,
        args.alpha,
    )
    write_csv(output / "bootstrap_95ci.csv", bootstrap_rows)

    mcnemar_rows = pairwise_mcnemar(method_rows, args.alpha)
    write_csv(output / "mcnemar_all_pairwise.csv", mcnemar_rows)

    output_files = [
        output / "confusion_matrices_long.csv",
        output / "confusion_matrices_wide.csv",
        output / "bootstrap_95ci.csv",
        output / "mcnemar_all_pairwise.csv",
        *normalized_outputs.values(),
    ]
    summary = {
        "analysis": "recording_level_confusion_bootstrap_mcnemar",
        "n_recordings": 941,
        "labels": LABELS,
        "methods": list(method_rows),
        "split_seed": 20268020,
        "bootstrap": {
            "unit": "recording",
            "method": "nonparametric percentile",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "confidence_level": 1.0 - args.alpha,
            "same_resample_indices_shared_across_methods": True,
        },
        "mcnemar": {
            "test": "exact two-sided paired McNemar",
            "comparisons": len(mcnemar_rows),
            "multiplicity_adjustment": "Holm across all pairwise comparisons",
            "alpha": args.alpha,
        },
        "validation": {
            "all_methods_have_941_unique_recording_ids": True,
            "all_methods_have_identical_recording_ids": True,
            "all_fold_and_true_labels_match_fixed_split": True,
            "all_probabilities_valid_and_predictions_reconstructable": True,
        },
        "input_provenance": {
            method: {"path": str(path), "sha256": sha256(path)}
            for method, path in source_paths.items()
        },
        "output_provenance": {
            path.name: {"path": str(path), "sha256": sha256(path)}
            for path in output_files
        },
    }
    summary_path = output / "recording_level_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[DONE] methods={len(method_rows)} recordings=941 "
        f"mcnemar_pairs={len(mcnemar_rows)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
