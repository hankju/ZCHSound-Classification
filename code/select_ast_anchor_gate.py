#!/usr/bin/env python
"""Lock a conservative AST blend using meta-OOF probabilities only."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

from nested_crossfit_sparse_stacking import (
    aligned_predict_proba,
    build_features,
    component_order,
    fit_stacker,
    score_probs,
)
from optimize_legacy_fusion_cv import load_component_names
from select_robust_anchor_fusion import load_meta_oof
from strict_samefold_sparse_stacking import normalize, select_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-output-root", required=True)
    parser.add_argument("--outer-prob-root", required=True)
    parser.add_argument("--component-grid", required=True)
    parser.add_argument("--outer-split-dir", required=True)
    parser.add_argument("--inner-split-dir", required=True)
    parser.add_argument("--output-lock", required=True)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--minimum-accuracy-gain", type=float, default=0.002)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.60)
    parser.add_argument("--maximum-nll-regression", type=float, default=0.01)
    return parser.parse_args()


def canonical_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def lock_anchor_config(base_probs, y, normal_idx, seed):
    ranked = select_config(base_probs, y, normal_idx, 5, 5, seed)
    _, config, metrics, std = ranked[0]
    selector, size, c_value, class_weight, alpha = config
    return {
        "selector": selector,
        "component_count": size,
        "C": c_value,
        "class_weight": class_weight,
        "stacking_blend_alpha": alpha,
        "selection_meta_oof_metrics": metrics,
        "selection_split_accuracy_std": std,
    }


def anchor_predict(train_probs, train_y, eval_probs, normal_idx, config):
    size = int(config["component_count"])
    selected = component_order(
        train_probs,
        train_y,
        config["selector"],
        size,
    )[:size]
    x_train, _ = build_features(train_probs, selected, normal_idx)
    x_eval, base_eval = build_features(eval_probs, selected, normal_idx)
    scaler, model = fit_stacker(
        x_train,
        train_y,
        float(config["C"]),
        config["class_weight"],
    )
    stacked = aligned_predict_proba(scaler, model, x_eval, eval_probs.shape[2])
    alpha = float(config["stacking_blend_alpha"])
    return normalize(alpha * stacked + (1.0 - alpha) * base_eval), selected


def evaluate_blends(probs, y, normal_idx, config, seed):
    blend_alphas = (
        0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00
    )
    splitter = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=seed,
    )
    sample_count, _, class_count = probs.shape
    anchor_sum = np.zeros((sample_count, class_count), dtype=np.float64)
    blend_sums = {
        alpha: np.zeros((sample_count, class_count), dtype=np.float64)
        for alpha in blend_alphas
    }
    counts = np.zeros(sample_count, dtype=np.int16)
    anchor_split_acc = []
    blend_split_acc = {alpha: [] for alpha in blend_alphas}

    for train_idx, valid_idx in splitter.split(np.zeros(sample_count), y):
        anchor, _ = anchor_predict(
            probs[train_idx, :6],
            y[train_idx],
            probs[valid_idx, :6],
            normal_idx,
            config,
        )
        ast = probs[valid_idx, 6]
        anchor_sum[valid_idx] += anchor
        counts[valid_idx] += 1
        anchor_split_acc.append(
            float(accuracy_score(y[valid_idx], anchor.argmax(axis=1)))
        )
        for alpha in blend_alphas:
            blended = normalize((1.0 - alpha) * anchor + alpha * ast)
            blend_sums[alpha][valid_idx] += blended
            blend_split_acc[alpha].append(
                float(accuracy_score(y[valid_idx], blended.argmax(axis=1)))
            )

    if not np.all(counts == 5):
        raise RuntimeError("incomplete repeated meta-OOF predictions")
    anchor_probs = anchor_sum / counts[:, None]
    anchor_metrics = score_probs(y, anchor_probs)
    anchor_metrics["split_accuracies"] = anchor_split_acc
    candidates = []
    for alpha in blend_alphas:
        current = score_probs(y, blend_sums[alpha] / counts[:, None])
        split_acc = np.asarray(blend_split_acc[alpha], dtype=np.float64)
        current.update({
            "blend_alpha": alpha,
            "split_accuracy_std": float(split_acc.std(ddof=1)),
            "split_accuracy_se": float(split_acc.std(ddof=1) / math.sqrt(len(split_acc))),
            "split_accuracies": split_acc.tolist(),
        })
        candidates.append(current)
    return anchor_metrics, candidates


def select_positive_one_se(candidates):
    best = sorted(candidates, key=lambda item: (-item["accuracy"], item["nll"]))[0]
    cutoff = best["accuracy"] - best["split_accuracy_se"]
    eligible = [item for item in candidates if item["accuracy"] >= cutoff]
    selected = sorted(
        eligible,
        key=lambda item: (item["blend_alpha"], item["nll"], -item["accuracy"]),
    )[0]
    return selected, best, cutoff


def main():
    args = parse_args()
    component_names = load_component_names(args.component_grid)
    if len(component_names) != 7 or component_names[-1] != "ast_frozen_audioset":
        raise RuntimeError("AST gate requires the six locked anchor components followed by AST")

    folds = []
    label_names = None
    for fold in range(5):
        probs, y, label_names = load_meta_oof(args, component_names, fold, label_names)
        normal_idx = label_names.index("NORMAL")
        config = lock_anchor_config(probs[:, :6], y, normal_idx, args.seed + fold)
        anchor, candidates = evaluate_blends(
            probs,
            y,
            normal_idx,
            config,
            args.seed + 100 + fold,
        )
        selected, best, cutoff = select_positive_one_se(candidates)
        selected_acc = np.asarray(selected["split_accuracies"])
        anchor_acc = np.asarray(anchor["split_accuracies"])
        paired_delta = selected_acc - anchor_acc
        paired_se = float(paired_delta.std(ddof=1) / math.sqrt(len(paired_delta)))
        required_gain = max(args.minimum_accuracy_gain, paired_se)
        gain = float(selected["accuracy"] - anchor["accuracy"])
        win_fraction = float(np.mean(paired_delta > 0.0))
        gate_passed = bool(
            gain > required_gain
            and win_fraction >= args.minimum_win_fraction
            and selected["nll"] <= anchor["nll"] + args.maximum_nll_regression
        )
        folds.append({
            "outer_fold": fold,
            "meta_oof_count": len(y),
            "anchor_config": config,
            "anchor_cv": anchor,
            "selected_positive_blend": selected,
            "best_positive_blend": best,
            "positive_one_se_cutoff": cutoff,
            "accuracy_gain_over_anchor": gain,
            "paired_accuracy_delta_se": paired_se,
            "required_accuracy_gain": required_gain,
            "paired_split_win_fraction": win_fraction,
            "gate_passed": gate_passed,
            "locked_family": "nested6_ast_fixed_blend" if gate_passed else "nested6_anchor",
            "locked_blend_alpha": selected["blend_alpha"] if gate_passed else 0.0,
            "all_positive_blends": candidates,
        })
        print(
            f"[FOLD] {fold} anchor={anchor['accuracy']:.4f} "
            f"blend={selected['accuracy']:.4f} alpha={selected['blend_alpha']:.3f} "
            f"gain={gain:+.4f} paired_se={paired_se:.4f} wins={win_fraction:.2f} "
            f"gate={gate_passed}",
            flush=True,
        )

    payload = {
        "tag": "nested7_ast_anchor_selection",
        "selection_only": True,
        "outer_test_probabilities_loaded": False,
        "test_used_for_selection": False,
        "component_names": component_names,
        "seed": args.seed,
        "minimum_accuracy_gain": args.minimum_accuracy_gain,
        "minimum_win_fraction": args.minimum_win_fraction,
        "maximum_nll_regression": args.maximum_nll_regression,
        "selection_protocol": (
            "For each outer fold, the original six-component sparse anchor configuration "
            "is selected on meta-OOF data. Fixed AST blend weights are compared by repeated "
            "5-fold x 5 meta-OOF CV. The one-SE rule favors the smallest positive weight. "
            "AST is admitted only when accuracy gain exceeds both 0.002 and the paired "
            "split-difference SE, wins at least 60% of paired splits, and NLL regresses by "
            "no more than 0.01. Outer-test artifacts are not loaded by this program."
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
