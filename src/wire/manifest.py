# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from wire.schema import RELEASE_FILES


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(data_dir: Path) -> dict:
    files = {}
    for name, schema in RELEASE_FILES.items():
        path = data_dir / name
        table = pq.read_table(path)
        if table.schema != schema:
            raise RuntimeError(f"{name}: schema differs from wire.schema\n{table.schema}\n---\n{schema}")
        files[name] = {"sha256": sha256_of(path), "rows": table.num_rows, "bytes": path.stat().st_size}
    return files


def write_manifest(data_dir: Path) -> dict:
    manifest = {
        "created": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": describe(data_dir),
    }
    (data_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Write MANIFEST.json (sha256 and row counts) for the release tables.")
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_manifest(args.data), indent=2))


if __name__ == "__main__":
    main()
