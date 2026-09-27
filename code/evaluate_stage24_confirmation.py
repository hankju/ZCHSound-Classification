#!/usr/bin/env python
"""Evaluate the transferred Stage24 lock on fresh patient folds exactly once."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from select_nested12_multiscale_gate import verify_lock
from select_stage24_conditional_development import conditional_fusion
from select_stage24_confirmation import (
    CURRICULUM,
    FLAT,
    LABELS,
    DEFAULT_SPLIT_SEED,
    TRANSFERRED_BETA,
    metrics,
)
from strict_samefold_sparse_stacking import normalize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--development-lock", required=True)
    parser.add_argument("--selection-lock", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    return parser.parse_args()


def read_component(root, split_seed, fold, name):
    path = (
        root
        / f"seed{split_seed}"
        / f"fold{fold}"
        / "components"
        / name
        / "test_file_probs.csv"
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty Stage24 test file: {path}")
    file_ids = [row["file_id"] for row in rows]
    y = np.asarray(
        [LABELS.index(row["true_label"]) for row in rows], dtype=np.int64
    )
    probs = np.asarray(
        [[float(row[f"prob_{label}"]) for label in LABELS] for row in rows],
        dtype=np.float64,
    )
    return file_ids, y, normalize(probs)


def main():
    args = parse_args()
    output = Path(args.output_dir).resolve()
    summary_path = output / "stage24_conditional_fusion_summary.json"
    if summary_path.exists():
        raise RuntimeError(f"Stage24 confirmation already evaluated: {summary_path}")
    development = verify_lock(args.development_lock)
    lock = verify_lock(args.selection_lock)
    if not (
        lock.get("tag")
        == "stage24_conditional_fusion_fresh_patient_cv_confirmation"
        and lock.get("test_used_for_selection") is False
        and lock.get("outer_test_probabilities_loaded") is False
        and float(lock.get("locked_conditional_beta", -1.0)) == TRANSFERRED_BETA
        and lock.get("development_selection_sha256")
        == development["selection_sha256"]
    ):
        raise RuntimeError("Stage24 confirmation lock does not match fixed protocol")

    test_root = Path(args.test_root).resolve()
    split_root = Path(args.split_root).resolve()
    decision_map = {
        (item["split_seed"], item["fold"]): item for item in lock["decisions"]
    }
    rows = []
    fold_results = []
    split_seeds = (args.split_seed,)
    if tuple(lock.get("split_seeds", ())) != split_seeds:
        raise RuntimeError("Stage24 confirmation split seed does not match lock")
    for split_seed in split_seeds:
        split_dir = split_root / f"splits_stage24_confirm_seed{split_seed}"
        for fold in range(5):
            decision = decision_map[(split_seed, fold)]
            if float(decision["locked_conditional_beta"]) != TRANSFERRED_BETA:
                raise RuntimeError(f"Stage24 fold {fold} beta mismatch")
            loaded = {}
            reference_ids = None
            reference_y = None
            for name in FLAT + CURRICULUM:
                file_ids, y, probs = read_component(
                    test_root, split_seed, fold, name
                )
                if reference_ids is None:
                    reference_ids, reference_y = file_ids, y
                elif file_ids != reference_ids or not np.array_equal(y, reference_y):
                    raise RuntimeError(
                        f"Stage24 test alignment failed seed={split_seed} fold={fold}"
                    )
                loaded[name] = probs
            with (split_dir / f"cv5_tvt_fold{fold}.json").open(
                "r", encoding="utf-8"
            ) as handle:
                split = json.load(handle)
            train = {Path(item).stem for item in split["train_files"]}
            val = {Path(item).stem for item in split["val_files"]}
            test = {Path(item).stem for item in split["test_files"]}
            disjoint = not (train & val or train & test or val & test)
            manifest_match = set(reference_ids) == set(split["test_files"])
            if not (disjoint and manifest_match):
                raise RuntimeError(
                    f"Stage24 provenance failed seed={split_seed} fold={fold}"
                )
            flat = normalize(np.mean([loaded[name] for name in FLAT], axis=0))
            curriculum = normalize(
                np.mean([loaded[name] for name in CURRICULUM], axis=0)
            )
            final = conditional_fusion(
                flat, curriculum, TRANSFERRED_BETA, LABELS.index("NORMAL")
            )
            normal_preserved = bool(
                np.allclose(
                    final[:, LABELS.index("NORMAL")],
                    flat[:, LABELS.index("NORMAL")],
                    atol=1e-12,
                )
            )
            if not normal_preserved:
                raise RuntimeError(f"Stage24 fold {fold} changed P(NORMAL)")
            fold_results.append({
                "split_seed": split_seed,
                "fold": fold,
                "locked_conditional_beta": TRANSFERRED_BETA,
                "flat_test_metrics": metrics(reference_y, flat),
                "curriculum_test_metrics": metrics(reference_y, curriculum),
                "locked_test_metrics": metrics(reference_y, final),
                "train_validation_test_patient_disjoint": disjoint,
                "test_files_match_fixed_split": manifest_match,
                "normal_probability_preserved_exactly": normal_preserved,
                "test_used_for_selection": False,
            })
            pred = final.argmax(axis=1)
            for index, file_id in enumerate(reference_ids):
                row = {
                    "split_seed": split_seed,
                    "fold": fold,
                    "file_id": file_id,
                    "true_label": LABELS[int(reference_y[index])],
                    "pred_label": LABELS[int(pred[index])],
                    "locked_conditional_beta": TRANSFERRED_BETA,
                }
                for family, values in (
                    ("flat", flat),
                    ("curriculum", curriculum),
                    ("final", final),
                ):
                    for class_index, label in enumerate(LABELS):
                        row[f"{family}_prob_{label}"] = float(
                            values[index, class_index]
                        )
                rows.append(row)
            print(
                f"[FOLD] seed={split_seed} fold={fold} beta={TRANSFERRED_BETA:.2f} "
                f"flat_acc={fold_results[-1]['flat_test_metrics']['accuracy']:.4f} "
                f"curriculum_acc={fold_results[-1]['curriculum_test_metrics']['accuracy']:.4f} "
                f"final_acc={fold_results[-1]['locked_test_metrics']['accuracy']:.4f}",
                flush=True,
            )

    repeat_results = []
    for split_seed in split_seeds:
        current_rows = [row for row in rows if row["split_seed"] == split_seed]
        y = np.asarray(
            [LABELS.index(row["true_label"]) for row in current_rows],
            dtype=np.int64,
        )
        current = {"split_seed": split_seed}
        for family in ("flat", "curriculum", "final"):
            probs = np.asarray(
                [
                    [row[f"{family}_prob_{label}"] for label in LABELS]
                    for row in current_rows
                ],
                dtype=np.float64,
            )
            current[f"{family}_pooled"] = metrics(y, probs)
            if family == "final":
                current["classification_report"] = classification_report(
                    y,
                    probs.argmax(axis=1),
                    labels=np.arange(len(LABELS)),
                    target_names=LABELS,
                    output_dict=True,
                    zero_division=0,
                )
        repeat_results.append(current)

    mean_metrics = {}
    for family in ("flat", "curriculum", "final"):
        mean_metrics[family] = {
            metric: float(
                np.mean(
                    [item[f"{family}_pooled"][metric] for item in repeat_results]
                )
            )
            for metric in (
                "accuracy",
                "macro_f1",
                "minority_recall",
                "minority_f1",
                "normal_recall",
                "nll",
            )
        }
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "stage24_conditional_fusion_predictions.csv"
    fields = [
        "split_seed",
        "fold",
        "file_id",
        "true_label",
        "pred_label",
        "locked_conditional_beta",
    ]
    for family in ("flat", "curriculum", "final"):
        fields.extend(f"{family}_prob_{label}" for label in LABELS)
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "tag": "stage24_conditional_fusion_fresh_patient_cv_locked_evaluation",
        "selection_sha256": lock["selection_sha256"],
        "development_selection_sha256": development["selection_sha256"],
        "test_used_for_selection": False,
        "outer_tests_evaluated_once_after_lock": True,
        "split_seeds": list(split_seeds),
        "locked_conditional_beta": TRANSFERRED_BETA,
        "repeat_results": repeat_results,
        "mean_across_repeats": mean_metrics,
        "fold_results": fold_results,
        "prediction_path": str(prediction_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"[DONE] flat_acc={mean_metrics['flat']['accuracy']:.6f} "
        f"curriculum_acc={mean_metrics['curriculum']['accuracy']:.6f} "
        f"final_acc={mean_metrics['final']['accuracy']:.6f} "
        f"final_macro={mean_metrics['final']['macro_f1']:.6f} "
        f"final_minf1={mean_metrics['final']['minority_f1']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
