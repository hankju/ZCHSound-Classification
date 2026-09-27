#!/usr/bin/env python
"""Map one of 40 paired Stage24 fresh-split acoustic tasks."""

import argparse
import subprocess
from pathlib import Path


DEFAULT_SPLIT_SEED = 20267020
TASKS_PER_FAMILY = 10
FAMILIES = (
    {
        "name": "ast_curriculum_balanced",
        "encoder": "ast",
        "kind": "curriculum",
        "objective": "balanced",
        "binary_loss_weight": "0.50",
        "subtype_loss_weight": "1.00",
        "consistency_weight": "0.10",
    },
    {
        "name": "beats_curriculum_conservative",
        "encoder": "beats",
        "kind": "curriculum",
        "objective": "conservative",
        "binary_loss_weight": "0.25",
        "subtype_loss_weight": "0.50",
        "consistency_weight": "0.05",
    },
    {"name": "ast_flat", "encoder": "ast", "kind": "flat"},
    {"name": "beats_flat", "encoder": "beats", "kind": "flat"},
)
TOTAL_TASKS = len(FAMILIES) * TASKS_PER_FAMILY


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--platform-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def nonempty(path):
    return path.is_file() and path.stat().st_size > 100


def location(root, family, replica, fold, split_seed):
    component = f"{family['name']}_seed2_source"
    split = (
        root
        / "configs"
        / f"splits_stage24_confirm_seed{split_seed}"
        / f"cv5_tvt_fold{fold}.json"
    )
    output = (
        root
        / "stage24_confirmation"
        / f"seed{split_seed}"
        / family["name"]
        / f"replica{replica}"
        / f"fold{fold}"
        / "components"
        / component
    )
    return split, output


def common_args(root, split, output, run_id, seed):
    return [
        "--split-json",
        str(split),
        "--metadata-csv",
        str(root / "configs" / "clean_dataset_manifest_merged.csv"),
        "--output-dir",
        str(output),
        "--run-id",
        run_id,
        "--seed",
        str(seed),
    ]


def curriculum_command(args, root, family, split, output, run_id, seed):
    input_root = (
        root / "data" / "ast_input_values_4s_ov3"
        if family["encoder"] == "ast"
        else root / "data" / "beats_fbank_4s_ov3"
    )
    command = [
        args.python,
        str(root / "code" / "train_curriculum_multitask_patient.py"),
        "--encoder",
        family["encoder"],
        "--input-root",
        str(input_root),
        *common_args(root, split, output, run_id, seed),
        "--objective-name",
        family["objective"],
        "--binary-loss-weight",
        family["binary_loss_weight"],
        "--subtype-loss-weight",
        family["subtype_loss_weight"],
        "--consistency-weight",
        family["consistency_weight"],
    ]
    if family["encoder"] == "beats":
        command.extend([
            "--checkpoint",
            str(root / "checkpoints" / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"),
            "--official-code-root",
            str(root / "third_party" / "unilm_beats" / "beats"),
            "--eval-file-batch-size",
            "1",
        ])
    return command


def flat_command(args, root, family, split, output, run_id, seed):
    common = common_args(root, split, output, run_id, seed)
    if family["encoder"] == "ast":
        return [
            args.python,
            str(root / "code" / "train_ast_lora_patient.py"),
            "--input-root",
            str(root / "data" / "ast_input_values_4s_ov3"),
            *common,
            "--class-weight-power",
            "0.5",
        ]
    return [
        args.python,
        str(root / "code" / "train_beats_patient.py"),
        "--input-root",
        str(root / "data" / "beats_fbank_4s_ov3"),
        *common,
        "--checkpoint",
        str(root / "checkpoints" / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"),
        "--official-code-root",
        str(root / "third_party" / "unilm_beats" / "beats"),
        "--variant",
        "lora_qv",
        "--eval-file-batch-size",
        "1",
    ]


def main():
    args = parse_args()
    if not 0 <= args.task_id < TOTAL_TASKS:
        raise ValueError(f"task-id must be in [0, {TOTAL_TASKS})")
    root = Path(args.project_root).resolve()
    platform = Path(args.platform_root).resolve()
    family_index, local_task = divmod(args.task_id, TASKS_PER_FAMILY)
    family = FAMILIES[family_index]
    replica = local_task // 5 + 1
    fold = local_task % 5
    split, output = location(root, family, replica, fold, args.split_seed)
    summary_name = {
        "curriculum": "curriculum_summary.json",
        "flat_ast": "lora_summary.json",
        "flat_beats": "beats_summary.json",
    }[
        family["kind"]
        if family["kind"] == "curriculum"
        else f"flat_{family['encoder']}"
    ]
    expected = (
        output / "val_file_probs.csv",
        output / "test_file_probs.csv",
        output / summary_name,
    )
    print(
        f"[RUN] task={args.task_id}/{TOTAL_TASKS} family={family['name']} "
        f"replica={replica} fold={fold}",
        flush=True,
    )
    if all(nonempty(path) for path in expected):
        print(f"[SKIP] complete output={output}", flush=True)
        return
    encoder_index = 0 if family["encoder"] == "ast" else 1
    seed = 20332100 + encoder_index * 100000 + replica * 1000 + fold
    run_id = f"seed{args.split_seed}_replica{replica}_fold{fold}"
    if family["kind"] == "curriculum":
        command = curriculum_command(
            args, root, family, split, output, run_id, seed
        )
    else:
        command = flat_command(args, root, family, split, output, run_id, seed)
    print("[COMMAND] " + " ".join(str(item) for item in command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=platform)


if __name__ == "__main__":
    main()
