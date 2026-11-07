# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wire.parquet_pages import (
    PageInfo,
    decode_page_levels,
    extract_rows,
    pages_for_row,
    read_page_header,
    scan_pages,
)


def make_table(rows: int, seed: int = 0) -> pa.Table:
    rng = random.Random(seed)
    texts, urls = [], []
    for i in range(rows):
        kind = rng.random()
        if kind < 0.05:
            texts.append(None)
        elif kind < 0.10:
            texts.append([])
        else:
            n = rng.choice([1, 2, 3, 40, 400]) if rng.random() < 0.2 else rng.randint(1, 6)
            texts.append(
                [None if rng.random() < 0.3 else f"row {i} paragraph {j} " + "x" * rng.randint(0, 60) for j in range(n)]
            )
        urls.append(None if rng.random() < 0.05 else f"https://example.org/doc/{i}")
    return pa.table({"texts": pa.array(texts, pa.list_(pa.string())), "url": pa.array(urls, pa.string())})


def write(table: pa.Table, path: Path, compression: str) -> None:
    pq.write_table(
        table,
        path,
        compression=compression,
        use_dictionary=False,
        data_page_size=2000,
        write_batch_size=64,
        data_page_version="1.0",
        write_page_index=False,
        write_statistics=False,
    )


def column_chunk(path: Path, index: int):
    meta = pq.ParquetFile(path).metadata.row_group(0).column(index)
    data = path.read_bytes()
    start = meta.dictionary_page_offset if meta.dictionary_page_offset is not None else meta.data_page_offset
    return data[start:start + meta.total_compressed_size], start, meta.compression


@pytest.mark.parametrize("compression", ["snappy", "zstd", "none"])
def test_scan_and_extract_match_pyarrow(tmp_path: Path, compression: str):
    table = make_table(600)
    path = tmp_path / "t.parquet"
    write(table, path, compression)
    data = path.read_bytes()

    for index, name, max_rep, max_def in ((0, "texts", 1, 3), (1, "url", 0, 1)):
        chunk, chunk_offset, codec = column_chunk(path, index)
        pages = scan_pages(chunk, chunk_offset, codec, max_rep, max_def)
        assert len(pages) >= 4, "the page size must be small enough to produce several pages"
        assert sum(p.num_rows for p in pages) == table.num_rows
        assert pages[0].offset == chunk_offset and not pages[0].continues
        expected = table.column(name).to_pylist()
        for row in [0, 1, 17, 299, 300, 598, 599] + random.Random(1).sample(range(600), 20):
            chosen = pages_for_row(pages, row)
            decoded = []
            for p in chosen:
                header = read_page_header(data, p.offset)
                raw = data[p.offset + header.header_size:p.offset + p.size]
                decoded.append(decode_page_levels(header, raw, codec, max_rep, max_def))
            got = extract_rows(decoded, chosen[0].first_row, [row], max_rep, max_def)
            assert got[row] == expected[row], (name, row)


def test_long_rows_across_small_pages(tmp_path: Path):
    # 2 KB pages and 400-element rows: a row can be most of a page or spill into the next one
    table = make_table(300, seed=7)
    path = tmp_path / "t.parquet"
    write(table, path, "snappy")
    chunk, chunk_offset, codec = column_chunk(path, 0)
    pages = scan_pages(chunk, chunk_offset, codec, 1, 3)
    data = path.read_bytes()
    expected = table.column("texts").to_pylist()
    long_rows = [i for i, t in enumerate(expected) if t and len(t) >= 400]
    assert long_rows
    for row in long_rows[:5]:
        chosen = pages_for_row(pages, row)
        decoded = []
        for p in chosen:
            header = read_page_header(data, p.offset)
            raw = data[p.offset + header.header_size:p.offset + p.size]
            decoded.append(decode_page_levels(header, raw, codec, 1, 3))
        assert extract_rows(decoded, chosen[0].first_row, [row], 1, 3)[row] == expected[row]


def test_pages_for_row_follows_continuations():
    pages = [
        PageInfo(0, 10, 0, 5, False),
        PageInfo(10, 10, 5, 3, False),
        PageInfo(20, 10, 8, 0, True),
        PageInfo(30, 10, 8, 2, True),
        PageInfo(40, 10, 10, 4, False),
    ]
    assert [p.offset for p in pages_for_row(pages, 2)] == [0]
    assert [p.offset for p in pages_for_row(pages, 4)] == [0]  # last of its page, the next page does not continue it
    assert [p.offset for p in pages_for_row(pages, 7)] == [10, 20, 30]  # spills through an all-continuation page
    assert [p.offset for p in pages_for_row(pages, 8)] == [30]
    assert [p.offset for p in pages_for_row(pages, 9)] == [30]  # the next page starts fresh
    assert [p.offset for p in pages_for_row(pages, 13)] == [40]
    with pytest.raises(KeyError):
        pages_for_row(pages, 14)
    # a record that spills into a page where other records start: that page is the last one needed
    spill = [PageInfo(0, 10, 0, 3, False), PageInfo(10, 10, 3, 4, True), PageInfo(20, 10, 7, 4, True)]
    assert [p.offset for p in pages_for_row(spill, 2)] == [0, 10]
    assert [p.offset for p in pages_for_row(spill, 3)] == [10]
    # a pruned table: the next listed page is not the physically next one, so it cannot continue the record
    pruned = [PageInfo(0, 10, 0, 3, False), PageInfo(50, 10, 40, 4, True)]
    assert [p.offset for p in pages_for_row(pruned, 2)] == [0]


def test_dictionary_pages_are_refused(tmp_path: Path):
    table = make_table(50)
    path = tmp_path / "t.parquet"
    pq.write_table(table, path, use_dictionary=True, compression="snappy", write_page_index=False)
    chunk, chunk_offset, codec = column_chunk(path, 1)
    with pytest.raises(ValueError, match="dictionary"):
        scan_pages(chunk, chunk_offset, codec, 0, 1)
