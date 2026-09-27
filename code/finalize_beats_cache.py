#!/usr/bin/env python
"""Validate all BEATs cache shards and write one immutable manifest."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from extract_ast_segment_embeddings import scan_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    dataset = Path(args.dataset_path).resolve()
    output = Path(args.output_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    shard_paths = [output / f"cache_manifest_shard{index:02d}.json" for index in range(args.shard_count)]
    if any(not path.is_file() for path in shard_paths):
        raise RuntimeError("incomplete BEATs cache shard manifests")
    shards = [json.load(path.open("r", encoding="utf-8")) for path in shard_paths]
    if any(item.get("failures") for item in shards):
        raise RuntimeError("a BEATs cache shard reported failures")
    files = scan_files(dataset)
    inventory, total_segments, total_bytes = [], 0, 0
    for index, source in enumerate(files, start=1):
        file_id = source.relative_to(dataset).as_posix()
        path = output / Path(file_id).with_suffix(".npy")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 3 or tuple(values.shape[1:]) != (398, 128) or values.dtype != np.float16 or not np.all(np.isfinite(values[0])):
            raise RuntimeError(f"invalid BEATs cache: {path}")
        inventory.append({"file_id": file_id, "segments": len(values)})
        total_segments += len(values)
        total_bytes += path.stat().st_size
        if index % 100 == 0:
            print(f"[VALIDATE] {index}/{len(files)} segments={total_segments}", flush=True)
    payload = {
        "tag": "beats_official_fbank_cache_v1_finalized",
        "labels_used_during_cache_creation": False,
        "split_information_used_during_cache_creation": False,
        "dataset_path": str(dataset),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "official_unilm_commit": "833df7e7832e5064a281131ee64a481afa8e5b95",
        "sample_rate": 16000,
        "window_sec": 4.0,
        "overlap_sec": 3.0,
        "feature_shape": [398, 128],
        "storage_dtype": "float16",
        "files": len(files),
        "segments": total_segments,
        "stored_bytes": total_bytes,
        "shard_count": args.shard_count,
        "shard_manifest_sha256": [sha256(path) for path in shard_paths],
        "inventory": inventory,
    }
    payload["configuration_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with (output / "cache_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[DONE] files={len(files)} segments={total_segments} size_gib={total_bytes / 2**30:.3f}", flush=True)


if __name__ == "__main__":
    main()
