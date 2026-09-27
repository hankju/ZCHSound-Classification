#!/usr/bin/env python
"""Lock a global Stage24 abnormal-conditional fusion beta from validation only."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from optimize_legacy_fusion_cv import load_component_names
from select_nested7_multimodal_gate import canonical_hash
from select_nested12_multiscale_gate import verify_lock
from select_nested23_curriculum_consensus import metrics, stage21_predict
from select_robust_anchor_fusion import load_meta_oof
from strict_samefold_sparse_stacking import normalize


LABELS = ("ASD", "NORMAL", "PDA", "PFO", "VSD")
FIXED_CURRICULUM_FAMILIES = (
    "ast_curriculum_balanced_seed2_mean",
    "beats_curriculum_conservative_seed2_mean",
)
BETAS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0)
VALIDATION_SEEDS = (20262424, 20263424, 20264424)
METRIC_NAMES = (
    "accuracy",
    "macro_f1",
    "minority_recall",
    "minority_f1",
    "normal_recall",
    "nll",
)


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "inner-output-root",
        "outer-prob-root",
        "component-grid",
        "outer-split-dir",
        "inner-split-dir",
        "acoustic-lock",
        "mil-lock",
        "nested11-lock",
        "nested12-lock",
        "nested13-lock",
        "nested14-lock",
        "nested16-lock",
        "nested20-lock",
        "nested21-lock",
        "output-lock",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--cv-repeats", type=int, default=3)
    return parser.parse_args()


def conditional_fusion(baseline, curriculum, beta, normal_index):
    """Preserve baseline normal mass and blend conditional abnormal classes."""
    abnormal = [index for index in range(baseline.shape[1]) if index != normal_index]
    baseline_conditional = normalize(baseline[:, abnormal])
    curriculum_conditional = normalize(curriculum[:, abnormal])
    conditional = normalize(
        (1.0 - float(beta)) * baseline_conditional
        + float(beta) * curriculum_conditional
    )
    result = np.zeros_like(baseline)
    result[:, normal_index] = baseline[:, normal_index]
    result[:, abnormal] = (1.0 - baseline[:, normal_index, None]) * conditional
    result = normalize(result)
    if not np.allclose(
        result[:, normal_index], baseline[:, normal_index], atol=1e-12
    ):
        raise RuntimeError("Stage24 conditional fusion changed P(NORMAL)")
    return result


def evaluate_run(
    probs,
    y,
    label_names,
    family_indices,
    locks,
    stage16_fold,
    stage20_fold,
    stage21_fold,
    component_names,
    seed,
    cv_repeats,
):
    normal_index = label_names.index("NORMAL")
    splitter = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=cv_repeats, random_state=seed
    )
    baseline_sum = np.zeros((len(y), len(label_names)), dtype=np.float64)
    fused_sums = {beta: np.zeros_like(baseline_sum) for beta in BETAS}
    counts = np.zeros(len(y), dtype=np.int16)
    for train_index, valid_index in splitter.split(np.zeros(len(y)), y):
        baseline = stage21_predict(
            probs[train_index],
            y[train_index],
            probs[valid_index],
            normal_index,
            locks,
            stage16_fold,
            stage20_fold,
            stage21_fold,
            component_names,
        )
        curriculum = normalize(
            probs[valid_index][:, family_indices, :].mean(axis=1)
        )
        baseline_sum[valid_index] += baseline
        counts[valid_index] += 1
        for beta in BETAS:
            fused_sums[beta][valid_index] += conditional_fusion(
                baseline, curriculum, beta, normal_index
            )
    if not np.all(counts == cv_repeats):
        raise RuntimeError("incomplete Stage24 development cross-fit")
    baseline_metrics = metrics(
        y, normalize(baseline_sum / counts[:, None]), label_names
    )
    candidates = {
        beta: metrics(
            y, normalize(fused_sums[beta] / counts[:, None]), label_names
        )
        for beta in BETAS
    }
    return baseline_metrics, candidates


def aggregate(items):
    result = {
        name: float(np.mean([item[name] for item in items]))
        for name in METRIC_NAMES
    }
    for name in ("per_class_recall", "per_class_f1"):
        result[name] = np.mean(
            [np.asarray(item[name], dtype=np.float64) for item in items], axis=0
        ).tolist()
    return result


def main():
    args = parse_args()
    outer_root = Path(args.outer_prob_root).resolve()
    if list(outer_root.glob("**/test_file_probs.csv")):
        raise RuntimeError("Stage24 development root contains outer-test probabilities")

    acoustic = verify_lock(args.acoustic_lock)
    mil = verify_lock(args.mil_lock)
    nested11 = verify_lock(args.nested11_lock)
    nested12 = verify_lock(args.nested12_lock)
    nested13 = verify_lock(args.nested13_lock)
    nested14 = verify_lock(args.nested14_lock)
    nested16 = verify_lock(args.nested16_lock)
    nested20 = verify_lock(args.nested20_lock)
    nested21 = verify_lock(args.nested21_lock)
    if nested16["nested14_selection_sha256"] != nested14["selection_sha256"]:
        raise RuntimeError("Stage24 Stage16/Stage14 lock mismatch")
    if nested20["nested16_selection_sha256"] != nested16["selection_sha256"]:
        raise RuntimeError("Stage24 Stage20/Stage16 lock mismatch")
    if nested21["nested20_selection_sha256"] != nested20["selection_sha256"]:
        raise RuntimeError("Stage24 Stage21/Stage20 lock mismatch")

    component_names = load_component_names(args.component_grid)
    if len(component_names) != 24 or component_names[:20] != nested21["component_names"]:
        raise RuntimeError("Stage24 component provenance mismatch")
    family_indices = [
        component_names.index(name) for name in FIXED_CURRICULUM_FAMILIES
    ]
    selection_args = SimpleNamespace(
        inner_output_root=args.inner_output_root,
        outer_prob_root=args.outer_prob_root,
        outer_split_dir=args.outer_split_dir,
        inner_split_dir=args.inner_split_dir,
        inner_folds=args.inner_folds,
    )

    baseline_runs = []
    candidate_runs = {beta: [] for beta in BETAS}
    run_records = []
    label_names = None
    for fold in range(5):
        probs, y, label_names = load_meta_oof(
            selection_args, component_names, fold, label_names
        )
        locks = (
            acoustic["folds"][fold],
            mil["folds"][fold],
            nested11["folds"][fold],
            nested12["folds"][fold],
            nested13["folds"][fold],
        )
        for base_seed in VALIDATION_SEEDS:
            seed = base_seed + fold
            baseline, candidates = evaluate_run(
                probs,
                y,
                label_names,
                family_indices,
                locks,
                nested16["folds"][fold],
                nested20["folds"][fold],
                nested21["folds"][fold],
                component_names,
                seed,
                args.cv_repeats,
            )
            baseline_runs.append(baseline)
            for beta in BETAS:
                candidate_runs[beta].append(candidates[beta])
            run_records.append({
                "outer_fold": fold,
                "validation_seed": seed,
                "meta_oof_count": len(y),
                "baseline": baseline,
                "candidates": [
                    {"beta": beta, **candidates[beta]} for beta in BETAS
                ],
            })

    baseline = aggregate(baseline_runs)
    candidates = []
    for beta in BETAS:
        current = aggregate(candidate_runs[beta])
        paired = np.asarray(
            [
                candidate["macro_f1"] - base["macro_f1"]
                for candidate, base in zip(candidate_runs[beta], baseline_runs)
            ],
            dtype=np.float64,
        )
        current.update({
            "beta": beta,
            "macro_f1_gain": current["macro_f1"] - baseline["macro_f1"],
            "paired_macro_f1_delta_se": float(
                paired.std(ddof=1) / math.sqrt(len(paired))
            ) if beta else 0.0,
            "paired_macro_f1_win_fraction": float(np.mean(paired > 0.0)),
        })
        current["guardrails_passed"] = bool(
            current["accuracy"] >= baseline["accuracy"] - 1.0 / 941.0
            and current["normal_recall"] >= baseline["normal_recall"] - 0.005
            and current["nll"] <= baseline["nll"] + 0.02
        )
        candidates.append(current)

    best = sorted(candidates, key=lambda item: (-item["macro_f1"], item["nll"]))[0]
    one_se_cutoff = best["macro_f1"] - best["paired_macro_f1_delta_se"]
    eligible = [
        item
        for item in candidates
        if item["beta"] > 0.0
        and item["guardrails_passed"]
        and item["macro_f1"] >= one_se_cutoff
    ]
    selected = sorted(eligible, key=lambda item: (item["beta"], item["nll"]))[0]
    required_gain = max(0.005, selected["paired_macro_f1_delta_se"])
    gate_passed = bool(
        selected["macro_f1_gain"] > required_gain
        and selected["paired_macro_f1_win_fraction"] >= 0.60
    )
    locked_beta = float(selected["beta"] if gate_passed else 0.0)

    payload = {
        "tag": "stage24_conditional_fusion_development_selection",
        "selection_only": True,
        "selection_root": str(outer_root),
        "selection_root_contains_test_probabilities": False,
        "outer_test_probabilities_loaded": False,
        "test_used_for_selection": False,
        "component_names": component_names,
        "fixed_curriculum_families": list(FIXED_CURRICULUM_FAMILIES),
        "beta_grid": list(BETAS),
        "validation_cv_base_seeds": list(VALIDATION_SEEDS),
        "validation_cv_repeats_per_seed": args.cv_repeats,
        "baseline_aggregate": baseline,
        "candidate_aggregates": candidates,
        "best_macro_f1_candidate": best,
        "one_se_macro_f1_cutoff": one_se_cutoff,
        "one_se_selected_candidate": selected,
        "required_macro_f1_gain": required_gain,
        "development_gate_passed": gate_passed,
        "locked_conditional_beta": locked_beta,
        "normal_probability_source": "immutable_Stage21",
        "conditional_probability_source": "fixed_AST_balanced_plus_BEATs_conservative_curriculum_mean",
        "selection_rule": (
            "Choose the smallest nonzero beta within one paired-SE of the best "
            "macro F1, subject to one-patient accuracy, 0.005 normal-recall, and "
            "0.02 NLL guardrails; require macro-F1 gain above max(0.005, paired SE) "
            "and at least 60% paired fold-seed wins. Preserve P(NORMAL) exactly."
        ),
        "nested21_selection_sha256": nested21["selection_sha256"],
        "run_records": run_records,
    }
    payload["selection_sha256"] = canonical_hash(payload)
    output = Path(args.output_lock).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(
        f"[LOCKED] beta={locked_beta:.2f} gate={gate_passed} "
        f"baseline_acc={baseline['accuracy']:.6f} "
        f"selected_acc={selected['accuracy']:.6f} "
        f"baseline_macro={baseline['macro_f1']:.6f} "
        f"selected_macro={selected['macro_f1']:.6f} "
        f"path={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
