# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_from_disk
from PIL import Image

from wire.assemble import assemble
from wire.schema import IMAGES, LABELS_TABLE, QUERIES


def write_tables(build: Path) -> None:
    queries = [
        {"query_id": "q1", "query": "red square", "nq_key": "k1", "split": "test"},
        {"query_id": "q2", "query": "blue square", "nq_key": "k2", "split": "train"},
    ]
    images = [
        {"image_id": "a", "image_url": "https://x/a.jpg", "page_url": "https://x/"},
        {"image_id": "b", "image_url": "https://x/b.jpg", "page_url": None},
        {"image_id": "c", "image_url": "https://x/c.jpg", "page_url": None},
    ]
    labels = [
        {"query_id": "q1", "image_id": "a", "label": "highly relevant", "safe_for_eval": True, "split": "test"},
        {"query_id": "q1", "image_id": "c", "label": "irrelevant", "safe_for_eval": True, "split": "test"},
        {"query_id": "q2", "image_id": "b", "label": "somewhat relevant", "safe_for_eval": True, "split": "train"},
    ]
    pq.write_table(pa.Table.from_pylist(queries, schema=QUERIES), build / "wire_queries.parquet")
    pq.write_table(pa.Table.from_pylist(images, schema=IMAGES), build / "wire_images.parquet")
    pq.write_table(pa.Table.from_pylist(labels, schema=LABELS_TABLE), build / "wire_labels.parquet")
    (build / "images").mkdir()
    for image_id, colour in (("a", (200, 0, 0)), ("b", (0, 0, 200))):
        buf = io.BytesIO()
        Image.new("RGB", (6, 4), colour).save(buf, format="PNG")
        (build / "images" / f"{image_id}.jpg").write_bytes(buf.getvalue())


def test_assemble_dataset_with_image_column(tmp_path: Path):
    write_tables(tmp_path)
    report = assemble(tmp_path, ("train", "test"), include_missing=False)
    assert report["splits"]["test"] == {"rows": 1, "with_image": 1, "without_image": 1}
    assert report["splits"]["train"] == {"rows": 1, "with_image": 1, "without_image": 0}

    dataset = load_from_disk(str(tmp_path / "wire_dataset"))
    assert set(dataset) == {"train", "test"}
    row = dataset["test"][0]
    assert row["query"] == "red square" and row["label"] == "highly relevant" and row["question"] is None
    assert row["image"].size == (6, 4) and row["image"].getpixel((0, 0)) == (200, 0, 0)
    assert str(dataset["test"].features["image"]) == "Image(mode=None, decode=True)"


def test_assemble_keeps_missing_rows_when_asked(tmp_path: Path):
    write_tables(tmp_path)
    report = assemble(tmp_path, ("test",), include_missing=True)
    assert report["splits"]["test"] == {"rows": 2, "with_image": 1, "without_image": 1}
    dataset = load_from_disk(str(tmp_path / "wire_dataset"))
    assert [r["image"] is None for r in dataset["test"]] == [False, True]
