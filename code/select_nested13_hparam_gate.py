#!/usr/bin/env python
"""Lock cross-fitted AST-LoRA hyperparameter admission over Nested12."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from optimize_legacy_fusion_cv import load_component_names
from select_ast_lora_anchor_gate import nested8_predict
from select_nested7_multimodal_gate import canonical_hash
from select_nested7_multimodal_multiobjective import detailed_metrics
from select_nested12_multiscale_gate import verify_lock
from select_robust_anchor_fusion import load_meta_oof
from strict_samefold_sparse_stacking import normalize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-output-root", required=True)
    parser.add_argument("--outer-prob-root", required=True)
    parser.add_argument("--component-grid", required=True)
    parser.add_argument("--outer-split-dir", required=True)
    parser.add_argument("--inner-split-dir", required=True)
    parser.add_argument("--acoustic-lock", required=True)
    parser.add_argument("--mil-lock", required=True)
    parser.add_argument("--nested11-lock", required=True)
    parser.add_argument("--nested12-lock", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-lock", required=True)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260913)
    return parser.parse_args()


def nested12_predict(train_probs, train_y, eval_probs, normal_idx, acoustic_fold, mil_fold, nested11_fold, nested12_fold):
    nested8, selected = nested8_predict(
        train_probs, train_y, eval_probs, normal_idx, acoustic_fold, mil_fold
    )
    alpha11 = float(nested11_fold["locked_lora_blend_alpha"])
    nested11 = normalize((1.0 - alpha11) * nested8 + alpha11 * eval_probs[:, 8])
    alpha12 = float(nested12_fold["locked_multiscale_blend_alpha"])
    nested12 = normalize((1.0 - alpha12) * nested11 + alpha12 * eval_probs[:, 9])
    return nested12, selected


def evaluate_alphas(probs, y, label_names, locks, seed):
    alphas = (0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0)
    normal_idx = label_names.index("NORMAL")
    minority_idx = np.asarray([label_names.index(name) for name in ("ASD", "PDA", "PFO")])
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    sums = {alpha: np.zeros((len(y), len(label_names))) for alpha in alphas}
    baseline_sum = np.zeros((len(y), len(label_names)))
    counts = np.zeros(len(y), dtype=np.int16)
    metric_names = ("accuracy", "macro_f1", "minority_recall")
    baseline_splits = {metric: [] for metric in metric_names}
    candidate_splits = {alpha: {metric: [] for metric in metric_names} for alpha in alphas}
    for train_idx, valid_idx in splitter.split(np.zeros(len(y)), y):
        baseline, _ = nested12_predict(
            probs[train_idx], y[train_idx], probs[valid_idx], normal_idx, *locks
        )
        baseline_sum[valid_idx] += baseline
        counts[valid_idx] += 1
        current = detailed_metrics(y[valid_idx], baseline, minority_idx, normal_idx)
        for metric in metric_names:
            baseline_splits[metric].append(current[metric])
        for alpha in alphas:
            predicted = normalize((1.0 - alpha) * baseline + alpha * probs[valid_idx, 10])
            sums[alpha][valid_idx] += predicted
            current = detailed_metrics(y[valid_idx], predicted, minority_idx, normal_idx)
            for metric in metric_names:
                candidate_splits[alpha][metric].append(current[metric])
    if not np.all(counts == 5):
        raise RuntimeError("incomplete repeated meta-OOF predictions")
    baseline = detailed_metrics(y, baseline_sum / counts[:, None], minority_idx, normal_idx)
    baseline["split_metrics"] = baseline_splits
    candidates = []
    for alpha in alphas:
        current = detailed_metrics(y, sums[alpha] / counts[:, None], minority_idx, normal_idx)
        current["blend_alpha"] = alpha
        current["split_metrics"] = candidate_splits[alpha]
        for metric in metric_names:
            values = np.asarray(candidate_splits[alpha][metric], dtype=np.float64)
            current[f"{metric}_split_se"] = float(values.std(ddof=1) / math.sqrt(len(values)))
        candidates.append(current)
    return baseline, candidates


def select_one_se(candidates, metric):
    best = sorted(candidates, key=lambda item: (-item[metric], item["nll"]))[0]
    cutoff = best[metric] - best[f"{metric}_split_se"]
    selected = sorted(
        [item for item in candidates if item[metric] >= cutoff],
        key=lambda item: (item["blend_alpha"], item["nll"], -item[metric]),
    )[0]
    return selected, best, cutoff


def gate_branch(branch, selected, baseline):
    minimum_gain = {"accuracy": 0.002, "macro_f1": 0.005, "minority_recall": 0.01}[branch]
    paired = np.asarray(selected["split_metrics"][branch]) - np.asarray(baseline["split_metrics"][branch])
    paired_se = float(paired.std(ddof=1) / math.sqrt(len(paired)))
    gain = float(selected[branch] - baseline[branch])
    wins = float(np.mean(paired > 0.0))
    common = gain > max(minimum_gain, paired_se) and wins >= 0.60 and selected["nll"] <= baseline["nll"] + 0.02
    if branch == "accuracy":
        guardrail = (
            selected["macro_f1"] >= baseline["macro_f1"] - 0.002
            and selected["minority_recall"] >= baseline["minority_recall"] - 0.005
            and selected["normal_recall"] >= baseline["normal_recall"] - 0.005
        )
    elif branch == "macro_f1":
        guardrail = selected["accuracy"] >= baseline["accuracy"] - 0.003 and selected["normal_recall"] >= baseline["normal_recall"] - 0.01
    else:
        guardrail = (
            selected["accuracy"] >= baseline["accuracy"] - 0.003
            and selected["macro_f1"] >= baseline["macro_f1"] - 0.002
            and selected["normal_recall"] >= baseline["normal_recall"] - 0.01
        )
    return {
        "branch": branch,
        "selected_candidate": selected,
        "gain": gain,
        "paired_delta_se": paired_se,
        "required_gain": max(minimum_gain, paired_se),
        "paired_split_win_fraction": wins,
        "guardrail_passed": bool(guardrail),
        "gate_passed": bool(common and guardrail),
    }


def main():
    args = parse_args()
    acoustic = verify_lock(args.acoustic_lock)
    mil = verify_lock(args.mil_lock)
    nested11 = verify_lock(args.nested11_lock)
    nested12 = verify_lock(args.nested12_lock)
    if nested12["acoustic_selection_sha256"] != acoustic["selection_sha256"]:
        raise RuntimeError("Nested12/acoustic lock mismatch")
    if nested12["mil_selection_sha256"] != mil["selection_sha256"]:
        raise RuntimeError("Nested12/MIL lock mismatch")
    if nested12["nested11_selection_sha256"] != nested11["selection_sha256"]:
        raise RuntimeError("Nested12/Nested11 lock mismatch")
    component_names = load_component_names(args.component_grid)
    if len(component_names) != 11 or component_names[:10] != nested12["component_names"] or component_names[-1] != args.candidate_name:
        raise RuntimeError("Nested12 component provenance mismatch")

    folds = []
    label_names = None
    for fold in range(5):
        probs, y, label_names = load_meta_oof(args, component_names, fold, label_names)
        locks = (acoustic["folds"][fold], mil["folds"][fold], nested11["folds"][fold], nested12["folds"][fold])
        baseline, candidates = evaluate_alphas(probs, y, label_names, locks, args.seed + fold)
        branch_results = []
        for branch in ("accuracy", "macro_f1", "minority_recall"):
            selected, best, cutoff = select_one_se(candidates, branch)
            result = gate_branch(branch, selected, baseline)
            result["best_candidate"] = best
            result["one_se_cutoff"] = cutoff
            branch_results.append(result)
        locked_branch, locked_alpha = "nested12_only", 0.0
        for result in branch_results:
            if result["gate_passed"]:
                locked_branch = result["branch"]
                locked_alpha = result["selected_candidate"]["blend_alpha"]
                break
        folds.append({
            "outer_fold": fold,
            "meta_oof_count": len(y),
            "baseline": baseline,
            "branch_results": branch_results,
            "locked_branch": locked_branch,
            "locked_hparam_blend_alpha": locked_alpha,
            "all_candidates": candidates,
        })
        print(f"[FOLD] {fold} baseline_acc={baseline['accuracy']:.4f} baseline_macro={baseline['macro_f1']:.4f} locked={locked_branch} alpha={locked_alpha:.3f}", flush=True)

    payload = {
        "tag": "nested13_ast_lora_hparam_selection",
        "candidate_name": args.candidate_name,
        "selection_only": True,
        "outer_test_probabilities_loaded": False,
        "test_used_for_selection": False,
        "component_names": component_names,
        "acoustic_selection_sha256": acoustic["selection_sha256"],
        "mil_selection_sha256": mil["selection_sha256"],
        "nested11_selection_sha256": nested11["selection_sha256"],
        "nested12_selection_sha256": nested12["selection_sha256"],
        "branch_priority": ["accuracy", "macro_f1", "minority_recall"],
        "seed": args.seed,
        "selection_protocol": "Leave-one-meta-block-out top-two LoRA hyperparameter ensemble, blended with immutable Nested12 using repeated validation OOF and fixed guardrails. Outer-test candidate files are materialized only after this hash lock.",
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
