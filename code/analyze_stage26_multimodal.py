#!/usr/bin/env python
"""Paired uncertainty and subgroup analysis for fixed Stage26 predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score, log_loss, recall_score


LABELS = ("ASD", "NORMAL", "PDA", "PFO", "VSD")
MINORITY = np.asarray([0, 2, 3], dtype=np.int64)
NORMAL = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20262626)
    parser.add_argument(
        "--tag", default="stage26_fixed_prediction_paired_analysis"
    )
    return parser.parse_args()


def metrics(y, probabilities):
    prediction = probabilities.argmax(axis=1)
    recalls = recall_score(
        y, prediction, labels=np.arange(len(LABELS)), average=None, zero_division=0
    )
    f1 = f1_score(
        y, prediction, labels=np.arange(len(LABELS)), average=None, zero_division=0
    )
    return np.asarray([
        np.mean(prediction == y),
        f1.mean(),
        recalls[MINORITY].mean(),
        f1[MINORITY].mean(),
        recalls[NORMAL],
        log_loss(y, probabilities, labels=np.arange(len(LABELS))),
    ], dtype=np.float64)


def subgroup(y, acoustic, multimodal, mask):
    if not np.any(mask):
        return {"count": 0}
    acoustic_pred = acoustic[mask].argmax(axis=1)
    multimodal_pred = multimodal[mask].argmax(axis=1)
    return {
        "count": int(mask.sum()),
        "acoustic_accuracy": float(np.mean(acoustic_pred == y[mask])),
        "multimodal_accuracy": float(np.mean(multimodal_pred == y[mask])),
        "accuracy_delta": float(
            np.mean(multimodal_pred == y[mask])
            - np.mean(acoustic_pred == y[mask])
        ),
        "acoustic_nll": float(log_loss(
            y[mask], acoustic[mask], labels=np.arange(len(LABELS))
        )),
        "multimodal_nll": float(log_loss(
            y[mask], multimodal[mask], labels=np.arange(len(LABELS))
        )),
    }


def main():
    args = parse_args()
    with Path(args.predictions).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    y = np.asarray([LABELS.index(row["true_label"]) for row in rows], dtype=np.int64)
    sex = np.asarray([float(row["sex01"]) for row in rows], dtype=np.float64)
    age = np.asarray([float(row["age_days"]) for row in rows], dtype=np.float64)
    probabilities = {
        family: np.asarray([
            [float(row[f"{family}_prob_{label}"]) for label in LABELS]
            for row in rows
        ], dtype=np.float64)
        for family in ("acoustic", "multimodal")
    }
    names = (
        "accuracy", "macro_f1", "minority_recall", "minority_f1",
        "normal_recall", "nll",
    )
    observed = {
        family: metrics(y, values)
        for family, values in probabilities.items()
    }
    observed_delta = observed["multimodal"] - observed["acoustic"]

    rng = np.random.default_rng(args.seed)
    class_indices = [np.flatnonzero(y == index) for index in range(len(LABELS))]
    bootstrap = np.empty((args.bootstrap_repeats, len(names)), dtype=np.float64)
    for repeat in range(args.bootstrap_repeats):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        bootstrap[repeat] = (
            metrics(y[sampled], probabilities["multimodal"][sampled])
            - metrics(y[sampled], probabilities["acoustic"][sampled])
        )
    intervals = {}
    for index, name in enumerate(names):
        values = bootstrap[:, index]
        intervals[name] = {
            "observed_delta": float(observed_delta[index]),
            "percentile_95_ci": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ],
            "bootstrap_probability_delta_gt_zero": float(np.mean(values > 0.0)),
            "bootstrap_probability_delta_lt_zero": float(np.mean(values < 0.0)),
        }

    acoustic_pred = probabilities["acoustic"].argmax(axis=1)
    multimodal_pred = probabilities["multimodal"].argmax(axis=1)
    acoustic_correct = acoustic_pred == y
    multimodal_correct = multimodal_pred == y
    acoustic_only = int(np.sum(acoustic_correct & ~multimodal_correct))
    multimodal_only = int(np.sum(~acoustic_correct & multimodal_correct))
    discordant = acoustic_only + multimodal_only
    mcnemar_p = (
        float(binomtest(min(acoustic_only, multimodal_only), discordant, 0.5).pvalue)
        if discordant else 1.0
    )
    age_years = age / 365.25
    subgroups = {
        "sex_0": subgroup(y, probabilities["acoustic"], probabilities["multimodal"], sex == 0),
        "sex_1": subgroup(y, probabilities["acoustic"], probabilities["multimodal"], sex == 1),
        "age_under_1_year": subgroup(
            y, probabilities["acoustic"], probabilities["multimodal"], age_years < 1
        ),
        "age_1_to_5_years": subgroup(
            y,
            probabilities["acoustic"],
            probabilities["multimodal"],
            (age_years >= 1) & (age_years <= 5),
        ),
        "age_over_5_years": subgroup(
            y, probabilities["acoustic"], probabilities["multimodal"], age_years > 5
        ),
    }
    payload = {
        "tag": args.tag,
        "selection_or_tuning_performed": False,
        "bootstrap": {
            "type": "paired class-stratified patient bootstrap",
            "repeats": args.bootstrap_repeats,
            "seed": args.seed,
            "metric_deltas": intervals,
        },
        "mcnemar_accuracy": {
            "acoustic_only_correct": acoustic_only,
            "multimodal_only_correct": multimodal_only,
            "discordant": discordant,
            "exact_two_sided_p": mcnemar_p,
        },
        "descriptive_subgroups": subgroups,
        "subgroup_warning": (
            "Subgroups are descriptive only and were not used for model selection."
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[DONE] Paired analysis saved to {output}")
    for name in names:
        item = intervals[name]
        print(
            f"{name}: delta={item['observed_delta']:+.6f} "
            f"95%CI=[{item['percentile_95_ci'][0]:+.6f},"
            f"{item['percentile_95_ci'][1]:+.6f}]"
        )
    print(
        f"McNemar acoustic_only={acoustic_only} multimodal_only={multimodal_only} "
        f"p={mcnemar_p:.6f}"
    )


if __name__ == "__main__":
    main()
