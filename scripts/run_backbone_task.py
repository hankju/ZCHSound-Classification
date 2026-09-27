#!/usr/bin/env python3
"""Run one of the 35 fixed single-backbone method/fold training tasks."""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID = ROOT / "configs" / "backbone7_retraining_grid.tsv"
DEFAULT_SPLITS = ROOT / "configs" / "splits_stage24_confirm_seed20268020"
DEFAULT_METADATA = ROOT / "configs" / "clean_dataset_manifest_merged.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--task-id", type=int, help="Task 0..34; method_index*5 + fold."
    )
    selection.add_argument("--method", help="method_id from the fixed TSV grid")
    parser.add_argument("--fold", type=int, help="Required with --method")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-root", type=Path, default=ROOT / "reproduced" / "backbones")
    parser.add_argument("--tensorboard-root", type=Path, default=ROOT / "reproduced" / "tb_backbones")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def read_grid(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 7 or len({row["method_id"] for row in rows}) != 7:
        raise RuntimeError("the fixed baseline grid must contain seven unique methods")
    return rows


def select_task(args, rows):
    if args.task_id is not None:
        if not 0 <= args.task_id < len(rows) * 5:
            raise ValueError("--task-id must be in 0..34")
        return rows[args.task_id // 5], args.task_id % 5
    if args.method is None or args.fold is None:
        raise ValueError("use --task-id, or provide both --method and --fold")
    if not 0 <= args.fold < 5:
        raise ValueError("--fold must be in 0..4")
    matches = [row for row in rows if row["method_id"] == args.method]
    if len(matches) != 1:
        raise ValueError(f"unknown method_id: {args.method}")
    return matches[0], args.fold


def validate_split(path, dataset_root):
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = [payload[name] for name in ("train_files", "val_files", "test_files")]
    sets = [set(group) for group in groups]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError(f"overlapping split IDs: {path}")
    missing = [item for group in groups for item in group if not (dataset_root / item).is_file()]
    if missing:
        raise RuntimeError(f"dataset is missing split files: {missing[:10]}")


def flag(command, row, field, option):
    if row[field] == "1":
        command.append(option)


def main():
    args = parse_args()
    rows = read_grid(args.grid)
    if args.list:
        for method_index, row in enumerate(rows):
            for fold in range(5):
                print(
                    f"{method_index * 5 + fold:2d}\t{row['method_id']}\tfold={fold}\t"
                    f"{row['backbone']}\tseed={row['seed']}"
                )
        return

    if args.dataset_root is None:
        raise ValueError("--dataset-root is required for training")
    dataset_root = args.dataset_root.expanduser().resolve()
    row, fold = select_task(args, rows)
    split_path = args.split_root / f"cv5_tvt_fold{fold}.json"
    validate_split(split_path, dataset_root)

    output_dir = args.output_root / row["method_id"] / f"fold{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_root.mkdir(parents=True, exist_ok=True)
    run_id = f"original941_seed20268020_{row['method_id']}_fold{fold}"

    command = [
        args.python,
        str(ROOT / "code" / "nostack_filelevel_ablate.py"),
        "--view", row["view"],
        "--backbone", row["backbone"],
        "--pooling", row["pooling"],
        "--topk_frac", row["topk_frac"],
        "--tta_offsets", row["tta_offsets"],
        "--window_sec", row["window_sec"],
        "--overlap_sec", row["overlap_sec"],
        "--loss", row["loss"],
        "--focal_gamma", row["focal_gamma"],
        "--focal_alpha", row["focal_alpha"],
        "--label_smoothing", row["label_smoothing"],
        "--spec_time_mask", row["spec_time_mask"],
        "--spec_freq_mask", row["spec_freq_mask"],
        "--mixup_alpha", row["mixup_alpha"],
        "--mixup_prob", row["mixup_prob"],
        "--exp_suffix", f"paper_{row['method_id']}",
        "--run_id", run_id,
        "--seed", row["seed"],
        "--dataset_path", str(dataset_root),
        "--split_json", str(split_path),
        "--metadata_csv", str(args.metadata_csv),
        "--tb_root", str(args.tensorboard_root),
        "--export_probs_dir", str(output_dir),
    ]
    flag(command, row, "no_sampler", "--no_sampler")
    flag(command, row, "no_entropy", "--no_entropy")
    flag(command, row, "no_mixup", "--no_mixup")
    flag(command, row, "no_specaug", "--no_specaug")

    environment = os.environ.copy()
    environment.update(
        {
            "BASE_LR": row["base_lr"],
            "WEIGHT_DECAY": row["weight_decay"],
            "BATCH_SIZE": row["batch_size"],
            "MAX_EPOCHS": row["max_epochs"],
            "EARLY_STOP_PATIENCE": row["early_stop_patience"],
            "LAMBDA_ENTROPY": row["lambda_entropy"],
            "WORKERS": str(args.workers),
            "PERSISTENT_WORKERS": "0",
        }
    )
    print(
        f"[TASK] method={row['method_id']} fold={fold} seed={row['seed']} "
        f"output={output_dir}",
        flush=True,
    )
    print("[COMMAND] " + shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
