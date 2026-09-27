#!/usr/bin/env python
"""Lock multiscale AST-LoRA admission over immutable Nested11 validation OOF."""

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
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-lock", required=True)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def verify_lock(path):
    with open(path, "r", encoding="utf-8") as handle:
        lock = json.load(handle)
    recorded = lock.pop("selection_sha256")
    if canonical_hash(lock) != recorded:
        raise RuntimeError(f"selection lock hash mismatch: {path}")
    lock["selection_sha256"] = recorded
    if not lock.get("selection_only") or lock.get("test_used_for_selection"):
        raise RuntimeError(f"invalid selection provenance: {path}")
    return lock


def nested11_predict(
    train_probs, train_y, eval_probs, normal_idx, acoustic_fold, mil_fold,
    nested11_fold,
):
    nested8, selected = nested8_predict(
        train_probs, train_y, eval_probs, normal_idx, acoustic_fold, mil_fold
    )
    alpha = float(nested11_fold["locked_lora_blend_alpha"])
    result = normalize((1.0 - alpha) * nested8 + alpha * eval_probs[:, 8])
    return result, selected


def evaluate_alphas(
    probs, y, label_names, acoustic_fold, mil_fold, nested11_fold, seed,
):
    alphas = (0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0)
    normal_idx = label_names.index("NORMAL")
    minority_idx = np.asarray(
        [label_names.index(name) for name in ("ASD", "PDA", "PFO")],
        dtype=np.int64,
    )
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    sums = {alpha: np.zeros((len(y), len(label_names))) for alpha in alphas}
    baseline_sum = np.zeros((len(y), len(label_names)))
    counts = np.zeros(len(y), dtype=np.int16)
    metric_names = ("accuracy", "macro_f1", "minority_recall")
    baseline_splits = {metric: [] for metric in metric_names}
    candidate_splits = {
        alpha: {metric: [] for metric in metric_names} for alpha in alphas
    }
    for train_idx, valid_idx in splitter.split(np.zeros(len(y)), y):
        baseline, _ = nested11_predict(
            probs[train_idx], y[train_idx], probs[valid_idx], normal_idx,
            acoustic_fold, mil_fold, nested11_fold,
        )
        baseline_sum[valid_idx] += baseline
        counts[valid_idx] += 1
        metrics = detailed_metrics(y[valid_idx], baseline, minority_idx, normal_idx)
        for metric in metric_names:
            baseline_splits[metric].append(metrics[metric])
        candidate = probs[valid_idx, 9]
        for alpha in alphas:
            predicted = normalize((1.0 - alpha) * baseline + alpha * candidate)
            sums[alpha][valid_idx] += predicted
            metrics = detailed_metrics(y[valid_idx], predicted, minority_idx, normal_idx)
            for metric in metric_names:
                candidate_splits[alpha][metric].append(metrics[metric])
    if not np.all(counts == 5):
        raise RuntimeError("incomplete repeated meta-OOF predictions")
    baseline = detailed_metrics(
        y, baseline_sum / counts[:, None], minority_idx, normal_idx
    )
    baseline["split_metrics"] = baseline_splits
    candidates = []
    for alpha in alphas:
        current = detailed_metrics(
            y, sums[alpha] / counts[:, None], minority_idx, normal_idx
        )
        current["blend_alpha"] = alpha
        current["split_metrics"] = candidate_splits[alpha]
        for metric in metric_names:
            values = np.asarray(candidate_splits[alpha][metric], dtype=np.float64)
            current[f"{metric}_split_se"] = float(
                values.std(ddof=1) / math.sqrt(len(values))
            )
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
    paired = np.asarray(selected["split_metrics"][branch]) - np.asarray(
        baseline["split_metrics"][branch]
    )
    paired_se = float(paired.std(ddof=1) / math.sqrt(len(paired)))
    gain = float(selected[branch] - baseline[branch])
    wins = float(np.mean(paired > 0.0))
    common = (
        gain > max(minimum_gain, paired_se)
        and wins >= 0.60
        and selected["nll"] <= baseline["nll"] + 0.02
    )
    if branch == "accuracy":
        guardrail = (
            selected["macro_f1"] >= baseline["macro_f1"] - 0.002
            and selected["minority_recall"] >= baseline["minority_recall"] - 0.005
            and selected["normal_recall"] >= baseline["normal_recall"] - 0.005
        )
    elif branch == "macro_f1":
        guardrail = (
            selected["accuracy"] >= baseline["accuracy"] - 0.003
            and selected["normal_recall"] >= baseline["normal_recall"] - 0.01
        )
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
    acoustic_lock = verify_lock(args.acoustic_lock)
    mil_lock = verify_lock(args.mil_lock)
    nested11_lock = verify_lock(args.nested11_lock)
    if mil_lock["acoustic_selection_sha256"] != acoustic_lock["selection_sha256"]:
        raise RuntimeError("MIL/acoustic lock mismatch")
    if nested11_lock["acoustic_selection_sha256"] != acoustic_lock["selection_sha256"]:
        raise RuntimeError("Nested11/acoustic lock mismatch")
    if nested11_lock["mil_selection_sha256"] != mil_lock["selection_sha256"]:
        raise RuntimeError("Nested11/MIL lock mismatch")
    component_names = load_component_names(args.component_grid)
    if len(component_names) != 10 or component_names[-1] != args.candidate_name:
        raise RuntimeError("expected immutable Nested11 components plus multiscale candidate")
    if component_names[:9] != nested11_lock["component_names"]:
        raise RuntimeError("Nested11 component provenance mismatch")

    folds = []
    label_names = None
    for fold in range(5):
        probs, y, label_names = load_meta_oof(args, component_names, fold, label_names)
        baseline, candidates = evaluate_alphas(
            probs, y, label_names, acoustic_lock["folds"][fold],
            mil_lock["folds"][fold], nested11_lock["folds"][fold],
            args.seed + fold,
        )
        branch_results = []
        for branch in ("accuracy", "macro_f1", "minority_recall"):
            selected, best, cutoff = select_one_se(candidates, branch)
            result = gate_branch(branch, selected, baseline)
            result["best_candidate"] = best
            result["one_se_cutoff"] = cutoff
            branch_results.append(result)
        locked_branch = "nested11_only"
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
            "locked_multiscale_blend_alpha": locked_alpha,
            "all_candidates": candidates,
        })
        print(
            f"[FOLD] {fold} baseline_acc={baseline['accuracy']:.4f} "
            f"baseline_macro={baseline['macro_f1']:.4f} locked={locked_branch} "
            f"alpha={locked_alpha:.3f}", flush=True,
        )
    payload = {
        "tag": "nested12_ast_lora_multiscale_selection",
        "candidate_name": args.candidate_name,
        "selection_only": True,
        "outer_test_probabilities_loaded": False,
        "test_used_for_selection": False,
        "component_names": component_names,
        "acoustic_selection_sha256": acoustic_lock["selection_sha256"],
        "mil_selection_sha256": mil_lock["selection_sha256"],
        "nested11_selection_sha256": nested11_lock["selection_sha256"],
        "branch_priority": ["accuracy", "macro_f1", "minority_recall"],
        "seed": args.seed,
        "selection_protocol": (
            "Immutable Nested11 is blended with one fixed equal-view 2s/4s/6s "
            "balanced AST-LoRA candidate. Repeated 5-fold x 5 meta-OOF validation "
            "uses one-SE selection, paired stability, NLL, macro-F1, minority-recall, "
            "and NORMAL-recall guardrails. Candidate outer-test files are linked only "
            "after this hash lock is written."
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

