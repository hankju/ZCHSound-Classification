#!/usr/bin/env python
"""Lock Stage23 curriculum admission without outer-test probabilities."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from optimize_legacy_fusion_cv import load_component_names
from select_nested7_multimodal_gate import canonical_hash
from select_nested7_multimodal_multiobjective import detailed_metrics
from select_nested12_multiscale_gate import verify_lock
from select_nested14_beats_gate import nested13_predict
from select_robust_anchor_fusion import load_meta_oof
from strict_samefold_sparse_stacking import normalize


VALIDATION_SEEDS = (20262323, 20263323, 20264323)
BRANCHES = ("accuracy", "macro_f1", "minority_f1")
SPLIT_METRICS = BRANCHES
SCALAR_METRICS = (
    "accuracy", "macro_f1", "minority_recall", "minority_f1",
    "normal_recall", "nll",
)
ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0)


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "inner-output-root", "outer-prob-root", "component-grid", "outer-split-dir",
        "inner-split-dir", "acoustic-lock", "mil-lock", "nested11-lock",
        "nested12-lock", "nested13-lock", "nested14-lock", "nested16-lock",
        "nested20-lock", "nested21-lock", "output-lock",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--cv-repeats", type=int, default=3)
    return parser.parse_args()


def stage16_predict(train_probs, train_y, eval_probs, normal_idx, locks, stage16_fold):
    nested13, _ = nested13_predict(train_probs, train_y, eval_probs, normal_idx, locks)
    alpha = float(stage16_fold["locked_beats_mil_blend_alpha"])
    return normalize((1.0 - alpha) * nested13 + alpha * eval_probs[:, 11])


def metrics(y, probs, label_names):
    normal_idx = label_names.index("NORMAL")
    minority_idx = np.asarray(
        [label_names.index(name) for name in ("ASD", "PDA", "PFO")], dtype=np.int64
    )
    return detailed_metrics(y, probs, minority_idx, normal_idx)


def select_families(candidate_probs, y, family_names):
    scored = []
    for index, name in enumerate(family_names):
        current = metrics(y, candidate_probs[:, index], ("ASD", "NORMAL", "PDA", "PFO", "VSD"))
        scored.append({"index": index, "name": name, **current})
    selected = []
    cutoffs = {}
    for encoder in ("ast", "beats"):
        group = [
            item for item in scored
            if item["name"].startswith(f"{encoder}_curriculum_")
        ]
        if len(group) != 2:
            raise RuntimeError(f"Stage23 expected two {encoder} objective families")
        best_accuracy = max(item["accuracy"] for item in group)
        cutoff = best_accuracy - 1.0 / len(y)
        eligible = [item for item in group if item["accuracy"] >= cutoff]
        chosen = sorted(
            eligible,
            key=lambda item: (item["nll"], -item["macro_f1"], item["name"]),
        )[0]
        selected.append(chosen)
        cutoffs[encoder] = cutoff
    return [item["index"] for item in selected], scored, cutoffs


def stage20_predict(
    train_probs,
    train_y,
    eval_probs,
    normal_idx,
    locks,
    stage16_fold,
    stage20_fold,
    component_names,
):
    baseline = stage16_predict(
        train_probs, train_y, eval_probs, normal_idx, locks, stage16_fold
    )
    family_indices = [
        component_names.index(name)
        for name in stage20_fold["locked_candidate_families"]
    ]
    candidate = normalize(eval_probs[:, family_indices, :].mean(axis=1))
    alpha = float(stage20_fold["locked_candidate_blend_alpha"])
    return normalize((1.0 - alpha) * baseline + alpha * candidate)


def stage21_predict(
    train_probs,
    train_y,
    eval_probs,
    normal_idx,
    locks,
    stage16_fold,
    stage20_fold,
    stage21_fold,
    component_names,
):
    baseline = stage20_predict(
        train_probs,
        train_y,
        eval_probs,
        normal_idx,
        locks,
        stage16_fold,
        stage20_fold,
        component_names,
    )
    family_indices = [
        component_names.index(name)
        for name in stage21_fold["locked_candidate_families"]
    ]
    candidate = normalize(eval_probs[:, family_indices, :].mean(axis=1))
    alpha = float(stage21_fold["locked_candidate_blend_alpha"])
    return normalize((1.0 - alpha) * baseline + alpha * candidate)


def evaluate_seed(
    probs,
    y,
    label_names,
    family_names,
    locks,
    stage16_fold,
    stage20_fold,
    stage21_fold,
    component_names,
    seed,
    cv_repeats,
):
    normal_idx = label_names.index("NORMAL")
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=cv_repeats, random_state=seed)
    baseline_sum = np.zeros((len(y), len(label_names)), dtype=np.float64)
    candidate_sums = {alpha: np.zeros_like(baseline_sum) for alpha in ALPHAS}
    counts = np.zeros(len(y), dtype=np.int16)
    baseline_split = {metric: [] for metric in SPLIT_METRICS}
    candidate_split = {
        alpha: {metric: [] for metric in SPLIT_METRICS} for alpha in ALPHAS
    }
    family_counter = Counter()
    for train_idx, valid_idx in splitter.split(np.zeros(len(y)), y):
        baseline = stage21_predict(
            probs[train_idx], y[train_idx], probs[valid_idx], normal_idx, locks,
            stage16_fold, stage20_fold, stage21_fold, component_names,
        )
        selected, _, _ = select_families(
            probs[train_idx, 20:, :], y[train_idx], family_names
        )
        family_counter.update(family_names[index] for index in selected)
        candidate = normalize(
            probs[valid_idx][:, 20 + np.asarray(selected, dtype=np.int64), :].mean(axis=1)
        )
        baseline_sum[valid_idx] += baseline
        counts[valid_idx] += 1
        current = metrics(y[valid_idx], baseline, label_names)
        for metric in SPLIT_METRICS:
            baseline_split[metric].append(current[metric])
        for alpha in ALPHAS:
            predicted = normalize((1.0 - alpha) * baseline + alpha * candidate)
            candidate_sums[alpha][valid_idx] += predicted
            current = metrics(y[valid_idx], predicted, label_names)
            for metric in SPLIT_METRICS:
                candidate_split[alpha][metric].append(current[metric])
    if not np.all(counts == cv_repeats):
        raise RuntimeError("incomplete Stage23 validation cross-fit")
    baseline = metrics(y, baseline_sum / counts[:, None], label_names)
    baseline["split_metrics"] = baseline_split
    candidates = []
    for alpha in ALPHAS:
        current = metrics(y, candidate_sums[alpha] / counts[:, None], label_names)
        current["blend_alpha"] = alpha
        current["split_metrics"] = candidate_split[alpha]
        for metric in SPLIT_METRICS:
            values = np.asarray(candidate_split[alpha][metric], dtype=np.float64)
            current[f"{metric}_split_se"] = float(values.std(ddof=1) / math.sqrt(len(values)))
        candidates.append(current)
    return baseline, candidates, dict(family_counter)


def candidate_for_alpha(candidates, alpha):
    matches = [item for item in candidates if np.isclose(item["blend_alpha"], alpha)]
    if len(matches) != 1:
        raise RuntimeError(f"Stage23 alpha lookup failed: {alpha}")
    return matches[0]


def aggregate_metrics(items, alpha=None):
    result = {metric: float(np.mean([item[metric] for item in items])) for metric in SCALAR_METRICS}
    for metric in ("per_class_recall", "per_class_f1"):
        result[metric] = np.mean([
            np.asarray(item[metric], dtype=np.float64) for item in items
        ], axis=0).tolist()
    result["split_metrics"] = {
        metric: [value for item in items for value in item["split_metrics"][metric]]
        for metric in SPLIT_METRICS
    }
    if alpha is not None:
        result["blend_alpha"] = float(alpha)
        for metric in SPLIT_METRICS:
            values = np.asarray(result["split_metrics"][metric], dtype=np.float64)
            result[f"{metric}_split_se"] = float(values.std(ddof=1) / math.sqrt(len(values)))
    return result


def select_one_se(candidates, branch):
    best = sorted(candidates, key=lambda item: (-item[branch], item["nll"]))[0]
    cutoff = best[branch] - best[f"{branch}_split_se"]
    eligible = [item for item in candidates if item[branch] >= cutoff]
    selected = sorted(eligible, key=lambda item: (item["blend_alpha"], item["nll"], -item[branch]))[0]
    return selected, best, cutoff


def gate(branch, selected, baseline, meta_count):
    minimum_gain = {"accuracy": 0.002, "macro_f1": 0.005, "minority_f1": 0.010}[branch]
    paired = (
        np.asarray(selected["split_metrics"][branch], dtype=np.float64)
        - np.asarray(baseline["split_metrics"][branch], dtype=np.float64)
    )
    paired_se = float(paired.std(ddof=1) / math.sqrt(len(paired)))
    gain = float(selected[branch] - baseline[branch])
    wins = float(np.mean(paired > 0.0))
    one_patient = 1.0 / meta_count
    if branch == "accuracy":
        guardrail = (
            selected["macro_f1"] >= baseline["macro_f1"] - 0.002
            and selected["minority_f1"] >= baseline["minority_f1"] - 0.005
            and selected["normal_recall"] >= baseline["normal_recall"] - 0.005
        )
    elif branch == "macro_f1":
        guardrail = (
            selected["accuracy"] >= baseline["accuracy"] - one_patient
            and selected["normal_recall"] >= baseline["normal_recall"] - 0.005
        )
    else:
        guardrail = (
            selected["accuracy"] >= baseline["accuracy"] - one_patient
            and selected["macro_f1"] >= baseline["macro_f1"] - 0.002
            and selected["normal_recall"] >= baseline["normal_recall"] - 0.005
        )
    required = max(minimum_gain, paired_se)
    passed = bool(
        gain > required
        and wins >= 0.60
        and selected["nll"] <= baseline["nll"] + 0.02
        and guardrail
    )
    return {
        "branch": branch,
        "selected_candidate": selected,
        "gain": gain,
        "paired_delta_se": paired_se,
        "required_gain": required,
        "paired_split_win_fraction": wins,
        "guardrail_passed": bool(guardrail),
        "gate_passed": passed,
    }


def main():
    args = parse_args()
    outer_root = Path(args.outer_prob_root).resolve()
    if list(outer_root.glob("**/test_file_probs.csv")):
        raise RuntimeError("Stage23 selection view contains outer-test probabilities")
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
        raise RuntimeError("Stage16/Stage14 lock mismatch")
    component_names = load_component_names(args.component_grid)
    if nested20["nested16_selection_sha256"] != nested16["selection_sha256"]:
        raise RuntimeError("Stage23 baseline Stage20/Stage16 lock mismatch")
    if nested21["nested20_selection_sha256"] != nested20["selection_sha256"]:
        raise RuntimeError("Stage23 baseline Stage21/Stage20 lock mismatch")
    if len(component_names) != 24 or component_names[:20] != nested21["component_names"]:
        raise RuntimeError("Stage23 component provenance mismatch")
    family_names = component_names[20:]
    folds = []
    label_names = None
    for fold in range(5):
        probs, y, label_names = load_meta_oof(args, component_names, fold, label_names)
        locks = (
            acoustic["folds"][fold], mil["folds"][fold], nested11["folds"][fold],
            nested12["folds"][fold], nested13["folds"][fold],
        )
        seed_runs = [
            evaluate_seed(
                probs, y, label_names, family_names, locks, nested16["folds"][fold],
                nested20["folds"][fold], nested21["folds"][fold], component_names,
                seed + fold, args.cv_repeats,
            )
            for seed in VALIDATION_SEEDS
        ]
        aggregate_baseline = aggregate_metrics([item[0] for item in seed_runs])
        aggregate_candidates = [
            aggregate_metrics(
                [candidate_for_alpha(item[1], alpha) for item in seed_runs], alpha
            )
            for alpha in ALPHAS
        ]
        branch_results = []
        for branch in BRANCHES:
            selected, best, cutoff = select_one_se(aggregate_candidates, branch)
            aggregate_gate = gate(branch, selected, aggregate_baseline, len(y))
            seed_gates = []
            for seed, (seed_baseline, seed_candidates, _) in zip(VALIDATION_SEEDS, seed_runs):
                current = gate(
                    branch,
                    candidate_for_alpha(seed_candidates, selected["blend_alpha"]),
                    seed_baseline,
                    len(y),
                )
                seed_gates.append({
                    "seed": seed + fold,
                    "gate_passed": current["gate_passed"],
                    "gain": current["gain"],
                    "paired_split_win_fraction": current["paired_split_win_fraction"],
                    "guardrail_passed": current["guardrail_passed"],
                })
            votes = sum(item["gate_passed"] for item in seed_gates)
            aggregate_gate.update({
                "best_candidate": best,
                "one_se_cutoff": cutoff,
                "validation_seed_gates": seed_gates,
                "consensus_votes": votes,
                "consensus_required": 2,
                "aggregate_gate_passed": aggregate_gate["gate_passed"],
            })
            aggregate_gate["gate_passed"] = bool(
                aggregate_gate["aggregate_gate_passed"] and votes >= 2
            )
            branch_results.append(aggregate_gate)
        locked_branch = "stage21_only"
        locked_alpha = 0.0
        for result in branch_results:
            if result["gate_passed"]:
                locked_branch = result["branch"]
                locked_alpha = result["selected_candidate"]["blend_alpha"]
                break
        selected_indices, family_metrics, family_cutoff = select_families(
            probs[:, 20:, :], y, family_names
        )
        locked_families = [family_names[index] for index in selected_indices]
        folds.append({
            "outer_fold": fold,
            "meta_oof_count": len(y),
            "aggregate_baseline": aggregate_baseline,
            "branch_results": branch_results,
            "locked_branch": locked_branch,
            "locked_candidate_blend_alpha": locked_alpha,
            "locked_candidate_families": locked_families,
            "full_meta_family_metrics": family_metrics,
            "full_meta_family_accuracy_cutoff": family_cutoff,
            "validation_cv_family_selection_frequency": [item[2] for item in seed_runs],
            "aggregate_candidates": aggregate_candidates,
        })
        print(
            f"[FOLD] {fold} baseline_acc={aggregate_baseline['accuracy']:.4f} "
            f"locked={locked_branch} alpha={locked_alpha:.2f} families={'+'.join(locked_families)}",
            flush=True,
        )
    payload = {
        "tag": "nested23_curriculum_consensus_selection",
        "selection_only": True,
        "selection_root": str(outer_root),
        "selection_root_contains_test_probabilities": False,
        "outer_test_probabilities_loaded": False,
        "test_used_for_selection": False,
        "component_names": component_names,
        "candidate_family_names": family_names,
        "acoustic_selection_sha256": acoustic["selection_sha256"],
        "mil_selection_sha256": mil["selection_sha256"],
        "nested11_selection_sha256": nested11["selection_sha256"],
        "nested12_selection_sha256": nested12["selection_sha256"],
        "nested13_selection_sha256": nested13["selection_sha256"],
        "nested14_selection_sha256": nested14["selection_sha256"],
        "nested16_selection_sha256": nested16["selection_sha256"],
        "nested20_selection_sha256": nested20["selection_sha256"],
        "nested21_selection_sha256": nested21["selection_sha256"],
        "validation_cv_base_seeds": list(VALIDATION_SEEDS),
        "validation_cv_repeats_per_seed": args.cv_repeats,
        "branch_priority": list(BRANCHES),
        "consensus_rule": "aggregate gate and at least two of three validation-CV seed gates must pass for the same alpha",
        "selection_protocol": (
            "One of two predeclared flat-first curriculum objectives is selected separately for AST and "
            "BEATs inside each validation-CV training split. Their equal mean is admitted over "
            "immutable Stage21 "
            "only under one-SE, paired stability, class guardrails, and three-seed consensus. "
            "Final fold family identities are selected on all same-fold meta OOF patients and "
            "serialized before any outer-test probability is available."
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
