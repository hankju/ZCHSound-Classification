#!/usr/bin/env python3
"""Run one Stage25 curriculum checkpoint on its validation or test fold."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import train_curriculum_multitask_patient as curriculum


DEFAULT_AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--encoder", choices=("ast", "beats"), required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-file-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ast-model-name", default=DEFAULT_AST_MODEL)
    parser.add_argument("--beats-base-checkpoint")
    parser.add_argument("--beats-official-code-root")
    parser.add_argument("--summary-json")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_args(args):
    if args.encoder == "beats" and (
        not args.beats_base_checkpoint or not args.beats_official_code_root
    ):
        raise ValueError(
            "BEATs inference requires --beats-base-checkpoint and "
            "--beats-official-code-root"
        )
    return SimpleNamespace(
        encoder=args.encoder,
        input_root=str(Path(args.input_root).resolve()),
        split_json=str(Path(args.split_json).resolve()),
        metadata_csv=str(Path(args.metadata_csv).resolve()),
        seed=args.seed,
        model_name=args.ast_model_name,
        tuning_mode="lora_qv",
        checkpoint=args.beats_base_checkpoint,
        official_code_root=args.beats_official_code_root,
        variant="lora_qv",
        rank=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
        train_segments_per_file=8,
        train_file_batch_size=4,
        eval_file_batch_size=args.eval_file_batch_size,
        num_workers=args.num_workers,
        sampler_power=0.25,
    )


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    curriculum.set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    run_args = runtime_args(args)
    files, labels, label_names = curriculum.load_split(run_args)
    (
        implementation,
        _,
        loaders,
        model,
        predictor,
        _,
        _,
        _,
        _,
        _,
    ) = curriculum.build_runtime(run_args, files, labels)

    checkpoint_path = Path(args.trained_checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("encoder") != args.encoder:
        raise RuntimeError("checkpoint encoder does not match --encoder")
    if tuple(checkpoint.get("label_names", ())) != tuple(label_names):
        raise RuntimeError("checkpoint labels do not match the fixed split")
    export_config = checkpoint.get("export_config")
    if not isinstance(export_config, dict):
        raise RuntimeError("checkpoint is missing export_config")

    implementation.restore_trainable_state(model, checkpoint["trainable_state_dict"])
    model.to(device)
    rows = predictor(model, loaders[args.split], device)
    probabilities = curriculum.probabilities_from_rows(
        rows,
        label_names,
        float(export_config["temperature"]),
        float(export_config["blend_alpha"]),
    )
    metadata = implementation.load_metadata(run_args.metadata_csv)
    output_path = Path(args.output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    implementation.export_predictions(
        output_path,
        args.split,
        f"stage25_{args.encoder}_{checkpoint_path.stem}",
        rows,
        probabilities,
        label_names,
        metadata,
    )

    summary = {
        "encoder": args.encoder,
        "evaluated_split": args.split,
        "seed": args.seed,
        "recordings": len(rows),
        "trained_checkpoint": str(checkpoint_path),
        "trained_checkpoint_sha256": sha256(checkpoint_path),
        "split_json": str(Path(args.split_json).resolve()),
        "split_json_sha256": sha256(Path(args.split_json).resolve()),
        "export_config": export_config,
        "output_csv": str(output_path),
        "output_csv_sha256": sha256(output_path),
    }
    summary_path = (
        Path(args.summary_json).resolve()
        if args.summary_json
        else output_path.with_suffix(".summary.json")
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] encoder={args.encoder} split={args.split} rows={len(rows)}")


if __name__ == "__main__":
    main()
