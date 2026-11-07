# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import hashlib

import pyarrow as pa

LABELS = ("highly relevant", "somewhat relevant", "irrelevant")
GRADE = {"highly relevant": 2, "somewhat relevant": 1, "irrelevant": 0}
SPLITS = ("train", "val", "test")

NQ_DATASET = "lucadiliello/naturalquestionsshortqa"


def image_id_from_url(image_url: str) -> str:
    """The stable image identifier used throughout WIRE: the first 32 hex chars of sha256(image_url).

    Verified against the annotated dataset on every row.
    """
    return hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:32]


def query_id_from_text(query: str) -> str:
    """Stable query identifier: the first 16 hex chars of sha1(query)."""
    return hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]


QUERIES = pa.schema(
    [
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("query", pa.string(), nullable=False),
        pa.field("nq_key", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
    ]
)


IMAGES = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("image_url", pa.string(), nullable=False),
        pa.field("page_url", pa.string(), nullable=True),
    ]
)

LABELS_TABLE = pa.schema(
    [
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("safe_for_eval", pa.bool_(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
    ]
)

# Where each image sits in the public MINT-1T-HTML corpus (`mlfoundations/MINT-1T-HTML`): the parquet file,
# the row in it and the image's slot in the document's interleaved text/image sequence. `mint_doc_idx` is the
# document's row index in the copy of the corpus the annotation pipeline used, kept as provenance. One row per
# document that contained the image; `mint_rank` 0 is the document the pipeline listed first.
MINT = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("mint_rank", pa.int32(), nullable=False),
        pa.field("mint_file", pa.string(), nullable=False),
        pa.field("mint_row", pa.int64(), nullable=False),
        pa.field("mint_image_idx", pa.int32(), nullable=False),
        pa.field("mint_num_slots", pa.int32(), nullable=False),
        pa.field("mint_doc_idx", pa.int64(), nullable=False),
        pa.field("page_url", pa.string(), nullable=False),
    ]
)

# The parquet data pages of MINT-1T-HTML (`texts` and `metadata` columns) that hold the primary rows above, so a
# rebuild reads pages instead of files: `offset` and `size` locate header plus body in the file, `first_row` /
# `num_rows` the records the page starts, `continues` whether it opens inside a record begun on the previous page.
MINT_PAGES = pa.schema(
    [
        pa.field("mint_file", pa.string(), nullable=False),
        pa.field("column", pa.string(), nullable=False),
        pa.field("codec", pa.string(), nullable=False),
        pa.field("offset", pa.int64(), nullable=False),
        pa.field("size", pa.int64(), nullable=False),
        pa.field("first_row", pa.int64(), nullable=False),
        pa.field("num_rows", pa.int64(), nullable=False),
        pa.field("continues", pa.bool_(), nullable=False),
    ]
)

RELEASE_FILES = {
    "wire_queries.parquet": QUERIES,
    "wire_images.parquet": IMAGES,
    "wire_labels.parquet": LABELS_TABLE,
    "wire_mint.parquet": MINT,
    "wire_mint_pages.parquet": MINT_PAGES,
}
