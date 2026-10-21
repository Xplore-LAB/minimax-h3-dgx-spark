#!/usr/bin/env python3
"""Dependency-free header/offset precheck; does not authenticate model weights.

The container downloader uses the official safetensors parser for full format
validation. This host-side precheck intentionally does not interpret tensor
dtypes or compare tensor shapes with byte lengths, and never reads tensor data.
Format: https://github.com/safetensors/safetensors#format
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import struct
import sys

MAX_HEADER_BYTES = 100_000_000


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate header key: {key}")
        result[key] = value
    return result


def check_header(path: Path) -> int:
    """Check a bounded JSON header and contiguous offsets against file size."""
    with path.open("rb") as stream:
        file_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("not a regular file")
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("missing 8-byte safetensors header length")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 0 < header_size <= MAX_HEADER_BYTES:
            raise ValueError("invalid or oversized safetensors header")
        if 8 + header_size > file_stat.st_size:
            raise ValueError("truncated safetensors header")
        raw_header = stream.read(header_size)
        if not raw_header.startswith(b"{"):
            raise ValueError("safetensors header must begin with '{'")
        header = json.loads(raw_header, object_pairs_hook=_unique_keys)
        if not isinstance(header, dict):
            raise ValueError("safetensors header is not an object")

    intervals = []
    payload_size = file_stat.st_size - 8 - header_size
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict) or not all(
                isinstance(value, str) for value in tensor.values()
            ):
                raise ValueError("invalid safetensors metadata")
            continue
        if not isinstance(tensor, dict):
            raise ValueError(f"invalid tensor header: {name}")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if not isinstance(tensor.get("dtype"), str) or not tensor["dtype"]:
            raise ValueError(f"missing tensor dtype: {name}")
        if not isinstance(shape, list) or not all(type(dim) is int and dim >= 0 for dim in shape):
            raise ValueError(f"invalid tensor shape: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(i) is int for i in offsets):
            raise ValueError(f"invalid tensor offsets: {name}")
        begin, end = offsets
        if not 0 <= begin <= end <= payload_size:
            raise ValueError(f"tensor data is truncated or out of bounds: {name}")
        intervals.append((begin, end))
    if not intervals:
        raise ValueError("model file has no tensors")
    cursor = 0
    for begin, end in sorted(intervals):
        if begin != cursor:
            raise ValueError("tensor offsets overlap or leave gaps")
        cursor = end
    if cursor != payload_size:
        raise ValueError("unindexed trailing data")
    return file_stat.st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    failed = False
    for path in args.files:
        try:
            size = check_header(path)
        except (OSError, ValueError, UnicodeError, RecursionError) as exc:
            print(f"INVALID {path}: {exc}", file=sys.stderr)
            failed = True
        else:
            print(f"HEADER OK {path} ({size} bytes)")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
