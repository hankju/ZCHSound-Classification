#!/usr/bin/env python
"""Extract label-free frozen AST embeddings for fixed heart-sound segments."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import librosa
import numpy as np
import scipy
import torch
import transformers
from scipy.signal import butter, filtfilt
from transformers import ASTFeatureExtractor, ASTForAudioClassification


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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_segment_starts(signal_len, win_len, stride_len):
    if signal_len <= 0 or signal_len < win_len:
        return [0]
    max_start = max(signal_len - win_len, 0)
    count = int(np.ceil((signal_len - win_len) / stride_len)) + 1
    starts = [min(idx * stride_len, max_start) for idx in range(count)]
    starts.append(max_start)
    return sorted(set(int(start) for start in starts)) or [0]


def extract_segment(signal, start, win_len):
    segment = signal[start : start + win_len]
    if len(segment) < win_len:
        segment = np.pad(segment, (0, win_len - len(segment)))
    return np.asarray(segment, dtype=np.float32)


def apply_bandpass(signal, sample_rate, lowcut=20.0, highcut=650.0, order=3):
    nyquist = 0.5 * sample_rate
    low = max(lowcut / nyquist, 1e-4)
    high = min(highcut / nyquist, 1.0 - 1e-4)
    b, a = butter(order, [low, high], btype="band")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(signal) <= padlen:
        return np.asarray(signal, dtype=np.float32)
    return filtfilt(b, a, signal).astype(np.float32)


def scan_files(root):
    return sorted(
        path
        for path in root.glob("*/*.wav")
        if path.is_file()
    )


def valid_existing(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            embeddings = data["embeddings"]
            starts = data["starts"]
        return (
            embeddings.ndim == 2
            and embeddings.shape[0] == len(starts)
            and embeddings.shape[1] > 0
            and np.all(np.isfinite(embeddings))
        )
    except Exception:
        return False


def atomic_save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main():
    args = parse_args()
    dataset = Path(args.dataset_path).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.overlap_sec < 0 or args.overlap_sec >= args.window_sec:
        raise ValueError("overlap must be in [0, window)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    files = scan_files(dataset)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"no WAV files found under {dataset}")

    extractor = ASTFeatureExtractor.from_pretrained(args.model_name)
    full_model = ASTForAudioClassification.from_pretrained(args.model_name)
    encoder = full_model.audio_spectrogram_transformer.to(args.device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    win_len = int(round(args.window_sec * args.sample_rate))
    stride_len = int(round((args.window_sec - args.overlap_sec) * args.sample_rate))
    completed = 0
    skipped = 0
    segment_total = 0
    failures = []

    for file_index, source in enumerate(files, start=1):
        file_id = source.relative_to(dataset).as_posix()
        destination = output / Path(file_id).with_suffix(".npz")
        if not args.force and destination.is_file() and valid_existing(destination):
            skipped += 1
            continue
        try:
            signal, _ = librosa.load(source, sr=args.sample_rate, mono=True)
            signal = apply_bandpass(signal, args.sample_rate)
            starts = build_segment_starts(len(signal), win_len, stride_len)
            embeddings = []
            for begin in range(0, len(starts), args.batch_size):
                current_starts = starts[begin : begin + args.batch_size]
                segments = [
                    extract_segment(signal, start, win_len) for start in current_starts
                ]
                inputs = extractor(
                    segments,
                    sampling_rate=args.sample_rate,
                    return_tensors="pt",
                )
                values = inputs["input_values"].to(args.device)
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
                ):
                    encoded = encoder(input_values=values).pooler_output
                embeddings.append(encoded.detach().float().cpu().numpy())
            stacked = np.concatenate(embeddings, axis=0)
            if stacked.shape[0] != len(starts) or not np.all(np.isfinite(stacked)):
                raise RuntimeError("invalid AST embedding output")
            atomic_save(
                destination,
                embeddings=stacked.astype(np.float16),
                starts=np.asarray(starts, dtype=np.int64),
                sample_rate=np.asarray(args.sample_rate, dtype=np.int32),
                window_sec=np.asarray(args.window_sec, dtype=np.float32),
                overlap_sec=np.asarray(args.overlap_sec, dtype=np.float32),
            )
            completed += 1
            segment_total += len(starts)
            print(
                f"[FILE] {file_index}/{len(files)} {file_id} segments={len(starts)} "
                f"embedding_dim={stacked.shape[1]}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"file_id": file_id, "error": repr(exc)})
            print(f"[ERROR] {file_id}: {exc!r}", flush=True)

    manifest = {
        "model_name": args.model_name,
        "model_type": "frozen_audio_spectrogram_transformer",
        "labels_used_during_embedding_extraction": False,
        "dataset_path": str(dataset),
        "sample_rate": args.sample_rate,
        "window_sec": args.window_sec,
        "overlap_sec": args.overlap_sec,
        "batch_size": args.batch_size,
        "files_requested": len(files),
        "files_completed": completed,
        "files_skipped": skipped,
        "segments_newly_extracted": segment_total,
        "failures": failures,
        "versions": {
            "python_torch": torch.__version__,
            "transformers": transformers.__version__,
            "librosa": librosa.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    manifest["configuration_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with open(output / "embedding_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    if failures:
        raise RuntimeError(f"AST extraction failed for {len(failures)} files")
    print(
        f"[DONE] completed={completed} skipped={skipped} segments={segment_total} "
        f"manifest={output / 'embedding_manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
