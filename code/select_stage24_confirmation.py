#!/usr/bin/env python
"""Transfer the Stage24 development lock to fresh validation folds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from select_nested7_multimodal_gate import canonical_hash
from select_nested7_multimodal_multiobjective import detailed_metrics
from select_nested12_multiscale_gate import verify_lock
from select_stage24_conditional_development import conditional_fusion
from strict_samefold_sparse_stacking import normalize


LABELS = ("ASD", "NORMAL", "PDA", "PFO", "VSD")
DEFAULT_SPLIT_SEED = 20267020
FLAT = ("ast_flat_seed2_mean", "beats_flat_seed2_mean")
CURRICULUM = (
    "ast_curriculum_balanced_seed2_mean",
    "beats_curriculum_conservative_seed2_mean",
)
TRANSFERRED_BETA = 0.75


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--development-lock", required=True)
    parser.add_argument("--output-lock", required=True)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    return parser.parse_args()


def read_component(root, split_seed, fold, name):
    path = (
        root
        / f"seed{split_seed}"
        / f"fold{fold}"
        / "components"
        / name
        / "val_file_probs.csv"
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty Stage24 validation file: {path}")
    file_ids = [row["file_id"] for row in rows]
    y = np.asarray(
        [LABELS.index(row["true_label"]) for row in rows], dtype=np.int64
    )
    probs = np.asarray(
        [[float(row[f"prob_{label}"]) for label in LABELS] for row in rows],
        dtype=np.float64,
    )
    return file_ids, y, normalize(probs)


def metrics(y, probs):
    minority = np.asarray([LABELS.index(name) for name in ("ASD", "PDA", "PFO")])
    return detailed_metrics(y, probs, minority, LABELS.index("NORMAL"))


def main():
    args = parse_args()
    validation_root = Path(args.validation_root).resolve()
    if list(validation_root.glob("**/test_file_probs.csv")):
        raise RuntimeError("Stage24 validation root contains test probabilities")
    development = verify_lock(args.development_lock)
    if not (
        development.get("tag")
        == "stage24_conditional_fusion_development_selection"
        and development.get("development_gate_passed") is True
        and float(development.get("locked_conditional_beta", -1.0))
        == TRANSFERRED_BETA
        and tuple(development.get("fixed_curriculum_families", ())) == CURRICULUM
        and development.get("test_used_for_selection") is False
    ):
        raise RuntimeError("Stage24 development lock does not match fixed protocol")

    split_root = Path(args.split_root).resolve()
    decisions = []
    split_seeds = (args.split_seed,)
    for split_seed in split_seeds:
        split_dir = split_root / f"splits_stage24_confirm_seed{split_seed}"
        for fold in range(5):
            loaded = {}
            reference_ids = None
            reference_y = None
            for name in FLAT + CURRICULUM:
                file_ids, y, probs = read_component(
                    validation_root, split_seed, fold, name
                )
                if reference_ids is None:
                    reference_ids, reference_y = file_ids, y
                elif file_ids != reference_ids or not np.array_equal(y, reference_y):
                    raise RuntimeError(
                        f"Stage24 validation alignment failed seed={split_seed} fold={fold}"
                    )
                loaded[name] = probs
            with (split_dir / f"cv5_tvt_fold{fold}.json").open(
                "r", encoding="utf-8"
            ) as handle:
                split = json.load(handle)
            if set(reference_ids) != set(split["val_files"]):
                raise RuntimeError(
                    f"Stage24 validation manifest mismatch seed={split_seed} fold={fold}"
                )
            flat = normalize(np.mean([loaded[name] for name in FLAT], axis=0))
            curriculum = normalize(
                np.mean([loaded[name] for name in CURRICULUM], axis=0)
            )
            final = conditional_fusion(
                flat, curriculum, TRANSFERRED_BETA, LABELS.index("NORMAL")
            )
            decisions.append({
                "split_seed": split_seed,
                "fold": fold,
                "validation_count": len(reference_y),
                "locked_conditional_beta": TRANSFERRED_BETA,
                "flat_validation_metrics": metrics(reference_y, flat),
                "curriculum_validation_metrics": metrics(reference_y, curriculum),
                "locked_validation_metrics": metrics(reference_y, final),
                "normal_probability_preserved_exactly": bool(
                    np.allclose(
                        final[:, LABELS.index("NORMAL")],
                        flat[:, LABELS.index("NORMAL")],
                        atol=1e-12,
                    )
                ),
            })
            print(
                f"[LOCK] seed={split_seed} fold={fold} beta={TRANSFERRED_BETA:.2f} "
                f"flat_acc={decisions[-1]['flat_validation_metrics']['accuracy']:.4f} "
                f"final_acc={decisions[-1]['locked_validation_metrics']['accuracy']:.4f}",
                flush=True,
            )
    payload = {
        "tag": "stage24_conditional_fusion_fresh_patient_cv_confirmation",
        "selection_only": True,
        "selection_root": str(validation_root),
        "selection_root_contains_test_probabilities": False,
        "outer_test_probabilities_loaded": False,
        "test_used_for_selection": False,
        "split_seeds": list(split_seeds),
        "folds_per_repeat": 5,
        "flat_families": list(FLAT),
        "curriculum_families": list(CURRICULUM),
        "locked_conditional_beta": TRANSFERRED_BETA,
        "development_selection_sha256": development["selection_sha256"],
        "selection_rule": (
            "Transfer beta=0.75 and the fixed AST-balanced plus BEATs-conservative "
            "curriculum pair from the pre-training development lock without fresh-test "
            "adaptation. Preserve paired-flat P(NORMAL) exactly."
        ),
        "decisions": decisions,
    }
    payload["selection_sha256"] = canonical_hash(payload)
    output = Path(args.output_lock).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[LOCKED] {output} sha256={payload['selection_sha256']}", flush=True)


if __name__ == "__main__":
    main()
