# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from wire.schema import LABELS, RELEASE_FILES, SPLITS, image_id_from_url, query_id_from_text

DATA = Path(__file__).resolve().parents[1] / "data"
FORBIDDEN = re.compile(r"AKIA[0-9A-Z]{16}|X-Amz-Credential|X-Amz-Signature|s3://")

pytestmark = pytest.mark.skipif(not (DATA / "MANIFEST.json").exists(), reason="release tables not built")


@pytest.fixture(scope="module")
def tables():
    return {name: pq.read_table(DATA / name) for name in RELEASE_FILES}


def test_schemas_and_manifest(tables):
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    for name, schema in RELEASE_FILES.items():
        assert tables[name].schema == schema, name
        assert manifest["files"][name]["rows"] == tables[name].num_rows, name


def test_no_internal_material(tables):
    for name, table in tables.items():
        for column in table.column_names:
            if table.schema.field(column).type != "string":
                continue
            for value in table.column(column).to_pylist():
                assert value is None or not FORBIDDEN.search(value), (name, column, value[:120])


def test_identifiers_are_derived(tables):
    queries = tables["wire_queries.parquet"].to_pylist()
    assert all(query_id_from_text(r["query"]) == r["query_id"] for r in queries)
    images = tables["wire_images.parquet"].to_pylist()
    assert all(image_id_from_url(r["image_url"]) == r["image_id"] for r in images)


def test_vocabularies(tables):
    queries = tables["wire_queries.parquet"]
    labels = tables["wire_labels.parquet"]
    assert set(queries.column("split").to_pylist()) <= set(SPLITS)
    assert set(labels.column("split").to_pylist()) <= set(SPLITS)
    assert all(queries.column("nq_key").to_pylist())
    assert set(labels.column("label").to_pylist()) == set(LABELS)


def test_referential_integrity(tables):
    labels = tables["wire_labels.parquet"]
    query_ids = set(tables["wire_queries.parquet"].column("query_id").to_pylist())
    image_ids = set(tables["wire_images.parquet"].column("image_id").to_pylist())
    assert set(labels.column("query_id").to_pylist()) == query_ids
    assert set(labels.column("image_id").to_pylist()) == image_ids
    pairs = list(zip(labels.column("query_id").to_pylist(), labels.column("image_id").to_pylist()))
    assert len(pairs) == len(set(pairs))


def test_split_invariants(tables):
    labels = tables["wire_labels.parquet"].to_pylist()
    for row in labels:
        if row["split"] in ("val", "test"):
            assert row["safe_for_eval"], row
    split_of_query = {}
    for row in labels:
        assert split_of_query.setdefault(row["query_id"], row["split"]) == row["split"]


def test_mint_locations_are_covered_by_pages(tables):
    from wire.parquet_pages import PageInfo, pages_for_row

    image_ids = set(tables["wire_images.parquet"].column("image_id").to_pylist())
    mint = tables["wire_mint.parquet"].to_pylist()
    assert {r["image_id"] for r in mint} <= image_ids
    keys = [(r["image_id"], r["mint_rank"]) for r in mint]
    assert len(keys) == len(set(keys))
    assert all(r["mint_file"].startswith("data_v1_") and r["mint_file"].endswith(".parquet") for r in mint)
    by_image = {}
    for r in mint:
        by_image.setdefault(r["image_id"], []).append(r["mint_rank"])
    assert all(0 in ranks for ranks in by_image.values()), "every image has a primary document"

    pages = {}
    for r in tables["wire_mint_pages.parquet"].to_pylist():
        assert r["column"] in ("texts", "metadata") and r["codec"] == "SNAPPY"
        pages.setdefault((r["mint_file"], r["column"]), []).append(
            PageInfo(r["offset"], r["size"], r["first_row"], r["num_rows"], r["continues"])
        )
    for key, infos in pages.items():
        infos.sort(key=lambda p: p.offset)
        assert all(a.offset + a.size <= b.offset for a, b in zip(infos, infos[1:])), key
    for r in mint:
        if r["mint_rank"] == 0:
            for column in ("texts", "metadata"):
                pages_for_row(pages[(r["mint_file"], column)], r["mint_row"])
