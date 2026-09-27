#!/usr/bin/env python3
"""Build or verify the byte-level manifest for the 941 source WAV files."""

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "dataset_audio_sha256.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Create or replace the manifest. Verification is the default.",
    )
    return parser.parse_args()


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def scan(dataset_root):
    files = sorted(
        path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"
    )
    return [
        {
            "relative_path": path.relative_to(dataset_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in files
    ]


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("relative_path", "size_bytes", "sha256")
        )
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    actual = scan(dataset_root)
    if args.write:
        if len(actual) != 941:
            raise RuntimeError(f"refusing to write: expected 941 WAV files, found {len(actual)}")
        write_manifest(args.manifest, actual)
        print(f"WROTE: files={len(actual)} manifest={args.manifest}")
        return

    expected = read_manifest(args.manifest)
    expected_map = {row["relative_path"]: row for row in expected}
    actual_map = {row["relative_path"]: row for row in actual}
    missing = sorted(set(expected_map) - set(actual_map))
    extra = sorted(set(actual_map) - set(expected_map))
    changed = sorted(
        path
        for path in set(expected_map) & set(actual_map)
        if expected_map[path]["sha256"] != actual_map[path]["sha256"]
        or int(expected_map[path]["size_bytes"]) != int(actual_map[path]["size_bytes"])
    )
    if len(expected) != 941 or missing or extra or changed:
        raise RuntimeError(
            "dataset verification failed: "
            f"manifest_rows={len(expected)} actual={len(actual)} "
            f"missing={missing[:10]} extra={extra[:10]} changed={changed[:10]}"
        )
    print(f"PASSED: verified {len(actual)} WAV files against {args.manifest}")


if __name__ == "__main__":
    main()
