#!/usr/bin/env python3
"""Recompute NORMAL-versus-CHD screening metrics from formal OOF predictions."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


LABELS = ("ASD", "NORMAL", "PDA", "PFO", "VSD")
METHOD_ARGUMENTS = (
    ("Stage25 fixed curriculum", "stage25_predictions"),
    ("Stage26 multimodal", "stage26_predictions"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage25-predictions", required=True)
    parser.add_argument("--stage26-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
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


def roc_auc(y_true, scores):
    """Return tie-aware Mann-Whitney ROC AUC for binary labels."""
    ranked = sorted(zip(scores, y_true), key=lambda item: item[0])
    positive_rank_sum = 0.0
    position = 0
    while position < len(ranked):
        end = position + 1
        while end < len(ranked) and ranked[end][0] == ranked[position][0]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in ranked[position:end]
        )
        position = end

    n_positive = sum(y_true)
    n_negative = len(y_true) - n_positive
    if not n_positive or not n_negative:
        raise RuntimeError("ROC AUC requires both positive and negative samples")
    return (
        positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def validate_and_score(method, path):
    rows = read_csv(path)
    required = {
        "recording_id",
        "fold",
        "true_label",
        "pred_label",
        *(f"prob_{label}" for label in LABELS),
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"missing required columns: {path}")

    ids = [row["recording_id"] for row in rows]
    if len(rows) != 941 or len(set(ids)) != 941:
        raise RuntimeError(f"expected 941 unique OOF recordings: {path}")

    tn = fp = fn = tp = 0
    binary_rows = []
    y_true = []
    scores = []
    for row in rows:
        probabilities = [float(row[f"prob_{label}"]) for label in LABELS]
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities):
            raise RuntimeError(f"invalid probability: {row['recording_id']}")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
            raise RuntimeError(f"probabilities do not sum to one: {row['recording_id']}")
        reconstructed = LABELS[max(range(len(LABELS)), key=probabilities.__getitem__)]
        if reconstructed != row["pred_label"]:
            raise RuntimeError(f"predicted label mismatch: {row['recording_id']}")

        true_positive = int(row["true_label"] != "NORMAL")
        pred_positive = int(row["pred_label"] != "NORMAL")
        chd_score = 1.0 - float(row["prob_NORMAL"])
        y_true.append(true_positive)
        scores.append(chd_score)

        if true_positive and pred_positive:
            tp += 1
        elif true_positive:
            fn += 1
        elif pred_positive:
            fp += 1
        else:
            tn += 1

        binary_rows.append(
            {
                "method": method,
                "recording_id": row["recording_id"],
                "fold": row["fold"],
                "true_binary_label": "CHD" if true_positive else "NORMAL",
                "pred_binary_label": "CHD" if pred_positive else "NORMAL",
                "prob_CHD": repr(chd_score),
                "prob_NORMAL": row["prob_NORMAL"],
            }
        )

    n = len(rows)
    metrics = {
        "method": method,
        "n": n,
        "n_negative_normal": tn + fp,
        "n_positive_chd": tp + fn,
        "accuracy": (tp + tn) / n,
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "roc_auc": roc_auc(y_true, scores),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    return metrics, binary_rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_binary_rows = []
    provenance = {}
    for method, argument in METHOD_ARGUMENTS:
        path = Path(getattr(args, argument)).resolve()
        metrics, binary_rows = validate_and_score(method, path)
        all_metrics.append(metrics)
        all_binary_rows.extend(binary_rows)
        provenance[method] = {"path": str(path), "sha256": sha256(path)}

    metrics_path = output_dir / "normal_vs_chd_metrics.csv"
    predictions_path = output_dir / "normal_vs_chd_predictions.csv"
    write_csv(metrics_path, all_metrics, list(all_metrics[0]))
    write_csv(
        predictions_path,
        all_binary_rows,
        [
            "method",
            "recording_id",
            "fold",
            "true_binary_label",
            "pred_binary_label",
            "prob_CHD",
            "prob_NORMAL",
        ],
    )

    summary = {
        "analysis": "NORMAL_vs_CHD_binary_screening",
        "negative_definition": "NORMAL",
        "positive_definition": ["ASD", "PDA", "PFO", "VSD"],
        "decision_rule": "predicted five-class label is not NORMAL",
        "roc_score": "1 - prob_NORMAL",
        "n_recordings": 941,
        "validation": {
            "unique_recording_ids_per_method": True,
            "probabilities_valid": True,
            "predictions_reconstructable": True,
        },
        "metrics": all_metrics,
        "input_provenance": provenance,
        "outputs": {
            metrics_path.name: sha256(metrics_path),
            predictions_path.name: sha256(predictions_path),
        },
    }
    summary_path = output_dir / "normal_vs_chd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for metrics in all_metrics:
        print(json.dumps(metrics, ensure_ascii=True))


if __name__ == "__main__":
    main()
