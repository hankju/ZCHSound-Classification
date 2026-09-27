#!/usr/bin/env python
"""Cache label-free official BEATs filterbanks for 4 s patient segments."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi

from extract_ast_segment_embeddings import (
    apply_bandpass,
    build_segment_starts,
    extract_segment,
    scan_files,
)


FEATURE_SHAPE = (398, 128)
FBANK_MEAN = 15.41663
FBANK_STD = 6.55582


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--official-code-root", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--overlap-sec", type=float, default=3.0)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_existing(path):
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        return (
            values.ndim == 3
            and values.shape[0] > 0
            and tuple(values.shape[1:]) == FEATURE_SHAPE
            and values.dtype == np.float16
            and np.all(np.isfinite(values[0]))
        )
    except Exception:
        return False


def atomic_save(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def segment_fbank(segment):
    waveform = torch.from_numpy(np.asarray(segment, dtype=np.float32)).unsqueeze(0)
    fbank = ta_kaldi.fbank(
        waveform * 2**15,
        num_mel_bins=128,
        sample_frequency=16000,
        frame_length=25,
        frame_shift=10,
    )
    fbank = (fbank - FBANK_MEAN) / (2 * FBANK_STD)
    if tuple(fbank.shape) != FEATURE_SHAPE:
        raise RuntimeError(f"unexpected BEATs fbank shape: {tuple(fbank.shape)}")
    return fbank.numpy().astype(np.float16)


def main():
    args = parse_args()
    if args.sample_rate != 16000:
        raise ValueError("official BEATs preprocessing requires 16 kHz")
    if args.overlap_sec < 0 or args.overlap_sec >= args.window_sec:
        raise ValueError("overlap must be in [0, window)")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    torch.set_num_threads(1)
    dataset = Path(args.dataset_path).resolve()
    output = Path(args.output_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    official_code = Path(args.official_code_root).resolve()
    if not checkpoint.is_file() or not (official_code / "BEATs.py").is_file():
        raise RuntimeError("missing BEATs checkpoint or official source")
    output.mkdir(parents=True, exist_ok=True)
    files = scan_files(dataset)
    if args.max_files is not None:
        files = files[: args.max_files]
    files = files[args.shard_index :: args.shard_count]
    if not files:
        raise RuntimeError(f"no WAV files found under {dataset}")

    win_len = int(round(args.window_sec * args.sample_rate))
    stride_len = int(round((args.window_sec - args.overlap_sec) * args.sample_rate))
    completed = skipped = total_segments = total_bytes = 0
    inventory, failures = [], []
    for file_index, source in enumerate(files, start=1):
        file_id = source.relative_to(dataset).as_posix()
        destination = output / Path(file_id).with_suffix(".npy")
        if not args.force and valid_existing(destination):
            values = np.load(destination, mmap_mode="r", allow_pickle=False)
            skipped += 1
        else:
            try:
                signal, _ = librosa.load(source, sr=args.sample_rate, mono=True)
                signal = apply_bandpass(signal, args.sample_rate)
                starts = build_segment_starts(len(signal), win_len, stride_len)
                values = np.stack(
                    [segment_fbank(extract_segment(signal, start, win_len)) for start in starts]
                )
                if values.dtype != np.float16 or not np.all(np.isfinite(values)):
                    raise RuntimeError("invalid cached BEATs values")
                atomic_save(destination, values)
                completed += 1
            except Exception as exc:
                failures.append({"file_id": file_id, "error": repr(exc)})
                print(f"[ERROR] {file_id}: {exc!r}", flush=True)
                continue
        total_segments += len(values)
        total_bytes += destination.stat().st_size
        inventory.append({"file_id": file_id, "segments": len(values)})
        if file_index % 25 == 0 or file_index == len(files):
            print(
                f"[CACHE] {file_index}/{len(files)} completed={completed} "
                f"skipped={skipped} segments={total_segments}", flush=True,
            )

    manifest = {
        "tag": "beats_official_fbank_cache_v1",
        "labels_used_during_cache_creation": False,
        "split_information_used_during_cache_creation": False,
        "dataset_path": str(dataset),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "official_code_root": str(official_code),
        "official_unilm_commit": "833df7e7832e5064a281131ee64a481afa8e5b95",
        "sample_rate": args.sample_rate,
        "window_sec": args.window_sec,
        "overlap_sec": args.overlap_sec,
        "feature_shape": list(FEATURE_SHAPE),
        "storage_dtype": "float16",
        "fbank_mean": FBANK_MEAN,
        "fbank_std": FBANK_STD,
        "files_requested": len(files),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "files_completed": completed,
        "files_skipped": skipped,
        "segments": total_segments,
        "stored_bytes": total_bytes,
        "inventory": inventory,
        "failures": failures,
        "versions": {
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "librosa": librosa.__version__,
            "numpy": np.__version__,
        },
    }
    manifest["configuration_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_name = (
        "cache_manifest.json" if args.shard_count == 1
        else f"cache_manifest_shard{args.shard_index:02d}.json"
    )
    with (output / manifest_name).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    if failures:
        raise RuntimeError(f"BEATs cache failed for {len(failures)} files")
    print(
        f"[DONE] files={len(files)} segments={total_segments} "
        f"size_gib={total_bytes / 2**30:.3f}", flush=True,
    )


if __name__ == "__main__":
    main()
