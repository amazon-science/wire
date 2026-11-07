# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import collections
import json
from pathlib import Path

import pyarrow.parquet as pq

from wire.manifest import describe
from wire.schema import RELEASE_FILES


def verify_release(data_dir: Path) -> dict:
    """Recompute sha256 and row counts of the release tables and compare with MANIFEST.json."""
    manifest = json.loads((data_dir / "MANIFEST.json").read_text())
    actual = describe(data_dir)
    mismatches = {
        name: {"manifest": manifest["files"][name], "actual": actual[name]}
        for name in RELEASE_FILES
        if manifest["files"][name]["sha256"] != actual[name]["sha256"]
        or manifest["files"][name]["rows"] != actual[name]["rows"]
    }
    return {"files": actual, "mismatches": mismatches}


def verify_build(build_dir: Path) -> dict:
    """Summarize a rebuilt dataset: what was fetched and what is usable, per split."""
    images = pq.read_table(build_dir / "wire_images.parquet").to_pylist()
    labels = pq.read_table(
        build_dir / "wire_labels.parquet", columns=["query_id", "image_id", "split", "label"]
    ).to_pylist()
    log = {r["image_id"]: r for r in pq.read_table(build_dir / "fetch_log.parquet").to_pylist()}
    on_disk = {p.stem for p in (build_dir / "images").glob("*.jpg")}

    status = collections.Counter(r["status"] for r in log.values())
    by_split: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    queries_complete: dict[str, set] = collections.defaultdict(set)
    queries_all: dict[str, set] = collections.defaultdict(set)
    for r in labels:
        ok = r["image_id"] in on_disk
        by_split[r["split"]]["rows"] += 1
        by_split[r["split"]]["rows_with_image"] += ok
        queries_all[r["split"]].add(r["query_id"])
        if not ok:
            queries_complete[r["split"]].add(r["query_id"])
    for split, c in by_split.items():
        c["queries"] = len(queries_all[split])
        c["queries_with_every_image"] = len(queries_all[split] - queries_complete[split])

    return {
        "images_listed": len(images),
        "images_on_disk": len(on_disk),
        "fetch_status": dict(status),
        "by_split": {k: dict(v) for k, v in by_split.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the release tables against MANIFEST.json and summarize a rebuild."
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="release directory (or a BUILD_DIR, which contains copies)"
    )
    parser.add_argument("--build", type=Path, default=None, help="rebuilt dataset directory to summarize")
    args = parser.parse_args()
    report = {"release": verify_release(args.data)}
    if report["release"]["mismatches"]:
        raise SystemExit(
            f"release tables differ from MANIFEST.json:\n{json.dumps(report['release']['mismatches'], indent=2)}"
        )
    if args.build is not None:
        report["build"] = verify_build(args.build)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
