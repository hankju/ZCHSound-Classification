#!/usr/bin/env python
"""Lock AST-MIL blending from meta-OOF predictions without outer-test access."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from optimize_legacy_fusion_cv import load_component_names
from select_ast_anchor_gate import anchor_predict
from select_nested7_multimodal_gate import canonical_hash
from select_nested7_multimodal_multiobjective import detailed_metrics
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
    parser.add_argument("--output-lock", required=True)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def current_acoustic_predict(train_probs, train_y, eval_probs, normal_idx, fold_lock):
    anchor, selected = anchor_predict(
        train_probs[:, :6],
        train_y,
        eval_probs[:, :6],
        normal_idx,
        fold_lock["anchor_config"],
    )
    alpha = float(fold_lock["locked_blend_alpha"])
    current = normalize((1.0 - alpha) * anchor + alpha * eval_probs[:, 6])
    return current, selected


def evaluate_alphas(probs, y, label_names, fold_lock, seed):
    alphas = (0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0)
    normal_idx = label_names.index("NORMAL")
    minority_idx = np.asarray(
        [label_names.index(name) for name in ("ASD", "PDA", "PFO")],
        dtype=np.int64,
    )
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    sums = {alpha: np.zeros((len(y), len(label_names))) for alpha in alphas}
    counts = np.zeros(len(y), dtype=np.int16)
    baseline_sum = np.zeros((len(y), len(label_names)))
    baseline_splits = {metric: [] for metric in ("accuracy", "macro_f1", "minority_recall")}
    candidate_splits = {
        alpha: {metric: [] for metric in baseline_splits}
        for alpha in alphas
    }
    for train_idx, valid_idx in splitter.split(np.zeros(len(y)), y):
        baseline, _ = current_acoustic_predict(
            probs[train_idx],
            y[train_idx],
            probs[valid_idx],
            normal_idx,
            fold_lock,
        )
        baseline_sum[valid_idx] += baseline
        counts[valid_idx] += 1
        baseline_metrics = detailed_metrics(y[valid_idx], baseline, minority_idx, normal_idx)
        for metric in baseline_splits:
            baseline_splits[metric].append(baseline_metrics[metric])
        mil = probs[valid_idx, 7]
        for alpha in alphas:
            predicted = normalize((1.0 - alpha) * baseline + alpha * mil)
            sums[alpha][valid_idx] += predicted
            current = detailed_metrics(y[valid_idx], predicted, minority_idx, normal_idx)
            for metric in baseline_splits:
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
        for metric in baseline_splits:
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


def gate_branch(branch, metric, selected, baseline):
    minimum_gain = {"accuracy": 0.002, "macro_f1": 0.005, "minority_recall": 0.01}[branch]
    selected_splits = np.asarray(selected["split_metrics"][metric], dtype=np.float64)
    baseline_splits = np.asarray(baseline["split_metrics"][metric], dtype=np.float64)
    paired = selected_splits - baseline_splits
    paired_se = float(paired.std(ddof=1) / math.sqrt(len(paired)))
    gain = float(selected[metric] - baseline[metric])
    wins = float(np.mean(paired > 0.0))
    common = (
        gain > max(minimum_gain, paired_se)
        and wins >= 0.60
        and selected["nll"] <= baseline["nll"] + 0.03
    )
    if branch == "accuracy":
        guardrail = selected["macro_f1"] >= baseline["macro_f1"] - 0.005
    elif branch == "macro_f1":
        guardrail = (
            selected["accuracy"] >= baseline["accuracy"] - 0.005
            and selected["normal_recall"] >= baseline["normal_recall"] - 0.01
        )
    else:
        guardrail = (
            selected["accuracy"] >= baseline["accuracy"] - 0.005
            and selected["macro_f1"] >= baseline["macro_f1"] - 0.002
            and selected["normal_recall"] >= baseline["normal_recall"] - 0.01
        )
    return {
        "branch": branch,
        "metric": metric,
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
    with open(args.acoustic_lock, "r", encoding="utf-8") as handle:
        acoustic_lock = json.load(handle)
    acoustic_hash = acoustic_lock.pop("selection_sha256")
    if canonical_hash(acoustic_lock) != acoustic_hash:
        raise RuntimeError("acoustic lock hash mismatch")
    acoustic_lock["selection_sha256"] = acoustic_hash
    component_names = load_component_names(args.component_grid)
    if len(component_names) != 8 or component_names[-1] != "ast_gated_mil":
        raise RuntimeError("expected locked seven components followed by ast_gated_mil")
    if component_names[:7] != acoustic_lock["component_names"]:
        raise RuntimeError("acoustic component provenance mismatch")

    folds = []
    label_names = None
    for fold in range(5):
        probs, y, label_names = load_meta_oof(args, component_names, fold, label_names)
        baseline, candidates = evaluate_alphas(
            probs,
            y,
            label_names,
            acoustic_lock["folds"][fold],
            args.seed + fold,
        )
        branch_results = []
        for branch in ("accuracy", "macro_f1", "minority_recall"):
            selected, best, cutoff = select_one_se(candidates, branch)
            result = gate_branch(branch, branch, selected, baseline)
            result["best_candidate"] = best
            result["one_se_cutoff"] = cutoff
            branch_results.append(result)
        locked_branch = "acoustic_only"
        locked_alpha = 0.0
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
            "locked_mil_blend_alpha": locked_alpha,
            "all_candidates": candidates,
        })
        print(
            f"[FOLD] {fold} baseline_acc={baseline['accuracy']:.4f} "
            f"baseline_macro={baseline['macro_f1']:.4f} locked={locked_branch} "
            f"alpha={locked_alpha:.3f}",
            flush=True,
        )
    payload = {
        "tag": "nested8_ast_mil_selection",
        "selection_only": True,
        "outer_test_probabilities_loaded": False,
        "test_used_for_selection": False,
        "component_names": component_names,
        "acoustic_selection_sha256": acoustic_lock["selection_sha256"],
        "branch_priority": ["accuracy", "macro_f1", "minority_recall"],
        "seed": args.seed,
        "selection_protocol": (
            "The locked seven-component acoustic method is the anchor. Fixed nonnegative "
            "AST-MIL blend weights are evaluated by repeated 5-fold x 5 meta-OOF CV. "
            "Accuracy, macro-F1, and ASD/PDA/PFO recall branches use one-SE selection, "
            "paired split stability, NLL limits, and accuracy/normal-recall guardrails. "
            "Outer-test artifacts are not loaded by this program."
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
