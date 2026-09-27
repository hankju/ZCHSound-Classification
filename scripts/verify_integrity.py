#!/usr/bin/env python3
"""Verify every file listed in SHA256SUMS."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    checksum_file = ROOT / "SHA256SUMS"
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}"
            )
        checked += 1
    print(f"PASSED: verified {checked} files")


if __name__ == "__main__":
    main()

