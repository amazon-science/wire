# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import datetime
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import concatenate_datasets, load_dataset

from wire.fetch import fetch_images
from wire.mint import build_mint_documents
from wire.schema import NQ_DATASET, RELEASE_FILES
from wire.verify import verify_release

QUERIES_FULL = pa.schema(
    [
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("query", pa.string(), nullable=False),
        pa.field("nq_key", pa.string(), nullable=True),
        pa.field("question", pa.string(), nullable=True),
        pa.field("answers", pa.list_(pa.string()), nullable=True),
        pa.field("context", pa.string(), nullable=True),
    ]
)


def now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def copy_release(release: Path, out: Path) -> dict:
    report = verify_release(release)
    if report["mismatches"]:
        raise SystemExit(f"release tables differ from MANIFEST.json: {json.dumps(report['mismatches'], indent=2)}")
    out.mkdir(parents=True, exist_ok=True)
    for name in list(RELEASE_FILES) + ["MANIFEST.json"]:
        shutil.copy2(release / name, out / name)
    return report["files"]


def build_nq(out: Path, revision: str | None) -> dict:
    """Restore the Natural Questions question, answers and context of every query through its `nq_key`."""
    nq = concatenate_datasets(list(load_dataset(NQ_DATASET, revision=revision).values()))
    by_key = {k: (q, a, c) for k, q, a, c in zip(nq["key"], nq["question"], nq["answers"], nq["context"])}
    rows = []
    missing = 0
    for r in pq.read_table(out / "wire_queries.parquet").to_pylist():
        hit = by_key.get(r["nq_key"]) if r["nq_key"] else None
        if hit is None:
            missing += 1
        rows.append(
            {
                "query_id": r["query_id"],
                "query": r["query"],
                "nq_key": r["nq_key"],
                "question": hit[0] if hit else None,
                "answers": hit[1] if hit else None,
                "context": hit[2] if hit else None,
            }
        )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=QUERIES_FULL), out / "wire_queries_full.parquet", compression="zstd"
    )
    return {
        "dataset": NQ_DATASET,
        "revision": revision,
        "nq_rows": nq.num_rows,
        "queries": len(rows),
        "queries_without_nq_row": missing,
    }


def tool_versions() -> dict:
    out = {}
    for pkg in ("wire", "aiohttp", "pillow", "datasets", "pyarrow"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the full WIRE dataset from the minimized release.")
    parser.add_argument("--release", type=Path, default=Path("data"), help="directory with the release tables")
    parser.add_argument("--out", type=Path, required=True, help="BUILD_DIR to create")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["tables", "nq", "images", "mint"],
        choices=["tables", "nq", "images", "mint"],
        help="tables: copy the release; nq: restore the Natural Questions fields; images: download the images; "
        "mint: recover the paragraphs around each image from the public MINT-1T-HTML files",
    )
    parser.add_argument("--nq-revision", default=None, help="git revision of the Natural Questions dataset to pin")
    parser.add_argument("--concurrency", type=int, default=64, help="images fetched at the same time")
    parser.add_argument("--per-host-interval", type=float, default=0.2, help="seconds between two requests to one host")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2, help="extra attempts for transient failures")
    parser.add_argument("--max-bytes", type=int, default=32 << 20)
    parser.add_argument("--refetch-failed", action="store_true", help="retry URLs that failed in a previous run")
    parser.add_argument("--no-research-agent", action="store_true", help="skip the research-agent pass over refusals")
    parser.add_argument("--patient-retries", type=int, default=5, help="attempts of the patient pass; 0 disables it")
    parser.add_argument("--no-archive", action="store_true", help="skip the Wayback Machine pass")
    parser.add_argument("--mint-concurrency", type=int, default=8, help="range requests in flight against the hub")
    args = parser.parse_args()

    manifest_path = args.out / "BUILD_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"steps": {}}
    if "tables" in args.steps:
        manifest["release_files"] = copy_release(args.release, args.out)
        manifest["steps"]["tables"] = now()
    if "nq" in args.steps:
        manifest["nq"] = build_nq(args.out, args.nq_revision)
        manifest["steps"]["nq"] = now()
    if "images" in args.steps:
        report = fetch_images(
            args.out,
            args.concurrency,
            args.per_host_interval,
            args.timeout,
            args.retries,
            args.max_bytes,
            args.refetch_failed,
            not args.no_research_agent,
            args.patient_retries,
            not args.no_archive,
        )
        (args.out / "fetch_report.json").write_text(json.dumps(report, indent=2) + "\n")
        manifest["images"] = {k: v for k, v in report.items() if k != "top_failing_hosts"}
        manifest["steps"]["images"] = now()
    if "mint" in args.steps:
        manifest["documents"] = dict(
            build_mint_documents(args.out, args.mint_concurrency, os.environ.get("HF_TOKEN")), source="mint"
        )
        manifest["steps"]["mint"] = now()
    manifest["tools"] = tool_versions()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
