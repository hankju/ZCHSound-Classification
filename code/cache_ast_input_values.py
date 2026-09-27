#!/usr/bin/env python
"""Cache label-free AST filterbank inputs for repeated parameter-efficient tuning."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import librosa
import numpy as np
import torch
import transformers
from transformers import ASTFeatureExtractor

from extract_ast_segment_embeddings import (
    apply_bandpass,
    build_segment_starts,
    extract_segment,
    scan_files,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--overlap-sec", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def valid_existing(path):
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        return (
            values.ndim == 3
            and values.shape[0] > 0
            and tuple(values.shape[1:]) == (1024, 128)
            and values.dtype == np.float16
            and np.all(np.isfinite(values[0]))
        )
    except Exception:
        return False


def atomic_save(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.overlap_sec < 0 or args.overlap_sec >= args.window_sec:
        raise ValueError("overlap must be in [0, window)")
    dataset = Path(args.dataset_path).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = scan_files(dataset)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"no WAV files found under {dataset}")

    extractor = ASTFeatureExtractor.from_pretrained(args.model_name)
    win_len = int(round(args.window_sec * args.sample_rate))
    stride_len = int(round((args.window_sec - args.overlap_sec) * args.sample_rate))
    completed = 0
    skipped = 0
    total_segments = 0
    total_bytes = 0
    failures = []
    inventory = []

    for file_index, source in enumerate(files, start=1):
        file_id = source.relative_to(dataset).as_posix()
        destination = output / Path(file_id).with_suffix(".npy")
        if not args.force and valid_existing(destination):
            values = np.load(destination, mmap_mode="r", allow_pickle=False)
            skipped += 1
            total_segments += len(values)
            total_bytes += destination.stat().st_size
            inventory.append({"file_id": file_id, "segments": len(values)})
            continue
        try:
            signal, _ = librosa.load(source, sr=args.sample_rate, mono=True)
            signal = apply_bandpass(signal, args.sample_rate)
            starts = build_segment_starts(len(signal), win_len, stride_len)
            batches = []
            for begin in range(0, len(starts), args.batch_size):
                current = starts[begin : begin + args.batch_size]
                segments = [extract_segment(signal, start, win_len) for start in current]
                encoded = extractor(
                    segments,
                    sampling_rate=args.sample_rate,
                    return_tensors="np",
                )["input_values"]
                batches.append(np.asarray(encoded, dtype=np.float16))
            values = np.concatenate(batches, axis=0)
            if (
                values.shape != (len(starts), 1024, 128)
                or not np.all(np.isfinite(values))
            ):
                raise RuntimeError(f"invalid AST input shape: {values.shape}")
            atomic_save(destination, values)
            completed += 1
            total_segments += len(values)
            total_bytes += destination.stat().st_size
            inventory.append({"file_id": file_id, "segments": len(values)})
            if file_index % 25 == 0 or file_index == len(files):
                print(
                    f"[CACHE] {file_index}/{len(files)} completed={completed} "
                    f"skipped={skipped} segments={total_segments}",
                    flush=True,
                )
        except Exception as exc:
            failures.append({"file_id": file_id, "error": repr(exc)})
            print(f"[ERROR] {file_id}: {exc!r}", flush=True)

    manifest = {
        "tag": "ast_input_values_cache_v1",
        "model_name": args.model_name,
        "labels_used_during_cache_creation": False,
        "split_information_used_during_cache_creation": False,
        "dataset_path": str(dataset),
        "sample_rate": args.sample_rate,
        "window_sec": args.window_sec,
        "overlap_sec": args.overlap_sec,
        "feature_shape": [1024, 128],
        "storage_dtype": "float16",
        "files_requested": len(files),
        "files_completed": completed,
        "files_skipped": skipped,
        "segments": total_segments,
        "stored_bytes": total_bytes,
        "inventory": inventory,
        "failures": failures,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "librosa": librosa.__version__,
            "numpy": np.__version__,
        },
    }
    manifest["configuration_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path = output / "cache_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    if failures:
        raise RuntimeError(f"AST input caching failed for {len(failures)} files")
    print(
        f"[DONE] files={len(files)} segments={total_segments} "
        f"size_gib={total_bytes / 2**30:.3f} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
