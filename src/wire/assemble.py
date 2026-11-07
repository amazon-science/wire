# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value

FEATURES = Features(
    {
        "query_id": Value("string"),
        "query": Value("string"),
        "nq_key": Value("string"),
        "question": Value("string"),
        "answers": Sequence(Value("string")),
        "image_id": Value("string"),
        "image_url": Value("string"),
        "page_url": Value("string"),
        "caption": Value("string"),
        "page_title": Value("string"),
        "doc_text": Value("string"),
        "label": Value("string"),
        "safe_for_eval": Value("bool"),
        "split": Value("string"),
        "image": Image(),
    }
)


def load_side_tables(out: Path) -> tuple[dict, dict, dict, dict]:
    queries = {r["query_id"]: r for r in pq.read_table(out / "wire_queries.parquet").to_pylist()}
    images = {r["image_id"]: r for r in pq.read_table(out / "wire_images.parquet").to_pylist()}
    full = out / "wire_queries_full.parquet"
    nq = {r["query_id"]: r for r in pq.read_table(full).to_pylist()} if full.exists() else {}
    docs_path = out / "wire_documents.parquet"
    docs = {r["image_id"]: r for r in pq.read_table(docs_path).to_pylist()} if docs_path.exists() else {}
    return queries, images, nq, docs


def assemble(out: Path, splits: tuple[str, ...], include_missing: bool) -> dict:
    """Join labels, queries, images, restored NQ fields and page context into one `datasets` dataset per split.

    The `image` column holds the fetched bytes (decoded on access), so the saved dataset is self-contained.
    """
    queries, images, nq, docs = load_side_tables(out)
    labels = pq.read_table(out / "wire_labels.parquet").to_pylist()
    images_dir = out / "images"
    counts: dict[str, dict[str, int]] = {}

    def rows(split: str):
        n = counts.setdefault(split, {"rows": 0, "with_image": 0, "without_image": 0})
        for r in labels:
            if r["split"] != split:
                continue
            path = images_dir / f"{r['image_id']}.jpg"
            if not path.exists() and not include_missing:
                n["without_image"] += 1
                continue
            q, im = queries[r["query_id"]], images[r["image_id"]]
            nq_row, doc = nq.get(r["query_id"], {}), docs.get(r["image_id"], {})
            n["rows"] += 1
            n["with_image" if path.exists() else "without_image"] += 1
            yield {
                "query_id": r["query_id"],
                "query": q["query"],
                "nq_key": q["nq_key"],
                "question": nq_row.get("question"),
                "answers": nq_row.get("answers"),
                "image_id": r["image_id"],
                "image_url": im["image_url"],
                "page_url": im["page_url"],
                "caption": doc.get("caption"),
                "page_title": doc.get("page_title"),
                "doc_text": doc.get("doc_text"),
                "label": r["label"],
                "safe_for_eval": r["safe_for_eval"],
                "split": r["split"],
                "image": {"bytes": path.read_bytes(), "path": None} if path.exists() else None,
            }

    cache = out / "_hf_cache"
    dataset = DatasetDict(
        {
            split: Dataset.from_generator(
                rows, gen_kwargs={"split": split}, features=FEATURES, cache_dir=str(cache), writer_batch_size=500
            )
            for split in splits
        }
    )
    target = out / "wire_dataset"
    dataset.save_to_disk(str(target))
    shutil.rmtree(cache, ignore_errors=True)  # scratch copy of the same 13 GB
    return {"path": str(target), "splits": counts}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack the rebuilt dataset into a `datasets` DatasetDict with an Image column."
    )
    parser.add_argument("--out", type=Path, required=True, help="BUILD_DIR produced by wire-build")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--include-missing", action="store_true", help="keep rows whose image could not be fetched (image=None)"
    )
    args = parser.parse_args()
    report = assemble(args.out, tuple(args.splits), args.include_missing)
    manifest_path = args.out / "BUILD_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"steps": {}}
    manifest["dataset"] = report
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
