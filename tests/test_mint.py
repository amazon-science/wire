# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import http.server
import json
import re
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import wire.mint as mint
from wire.mint import LEVELS, build_mint_documents
from wire.parquet_pages import scan_pages
from wire.schema import IMAGES, MINT, MINT_PAGES, image_id_from_url


def mint_table() -> pa.Table:
    """A MINT-like interleaved corpus: texts and images are aligned lists, one of them null at every slot."""
    docs = []
    for d in range(120):
        texts, images = [], []
        for slot in range(6):
            if slot % 2 == 1:
                texts.append(None)
                images.append(f"https://img.example/{d}/{slot}.jpg")
            else:
                texts.append(f"doc {d} paragraph {slot} " + "y" * (d % 17))
                images.append(None)
        docs.append((texts, images, f"https://page.example/{d}"))
    metadata = [
        json.dumps([None if u is None else {"src": u, "alt_text": f"alt {d} {i}"} for i, u in enumerate(images)])
        for d, (_, images, _) in enumerate(docs)
    ]
    return pa.table(
        {
            "images": pa.array([d[1] for d in docs], pa.list_(pa.string())),
            "texts": pa.array([d[0] for d in docs], pa.list_(pa.string())),
            "metadata": pa.array(metadata, pa.string()),
            "url": pa.array([d[2] for d in docs], pa.string()),
        }
    )


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with HTTP Range support, which SimpleHTTPRequestHandler lacks."""

    hits: list[str] = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = Path(self.directory) / self.path.lstrip("/")
        RangeHandler.hits.append(self.path)
        data = path.read_bytes()
        m = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            body = data[lo:hi + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def page_rows(parquet: Path, name: str) -> list[dict]:
    pf = pq.ParquetFile(parquet)
    data = parquet.read_bytes()
    rows = []
    for i in range(pf.metadata.row_group(0).num_columns):
        col = pf.metadata.row_group(0).column(i)
        column = col.path_in_schema.split(".")[0]
        if column not in LEVELS:
            continue
        chunk = data[col.data_page_offset:col.data_page_offset + col.total_compressed_size]
        for p in scan_pages(chunk, col.data_page_offset, col.compression, *LEVELS[column]):
            rows.append(
                {
                    "mint_file": name,
                    "column": column,
                    "codec": col.compression,
                    "offset": p.offset,
                    "size": p.size,
                    "first_row": p.first_row,
                    "num_rows": p.num_rows,
                    "continues": p.continues,
                }
            )
    return rows


def test_mint_documents_from_pages(tmp_path: Path, monkeypatch):
    serve_dir = tmp_path / "hub"
    (serve_dir / "data_v1_1").mkdir(parents=True)
    table = mint_table()
    name = "data_v1_1/shard_001_part_001.parquet"
    pq.write_table(
        table,
        serve_dir / name,
        compression="snappy",
        use_dictionary=False,
        data_page_size=1500,
        write_batch_size=16,
        data_page_version="1.0",
        write_page_index=False,
    )
    handler = type("H", (RangeHandler,), {"directory": str(serve_dir)})
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), lambda *a, **k: handler(*a, directory=str(serve_dir), **k)
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(mint, "RESOLVE_URL", f"http://127.0.0.1:{server.server_address[1]}/{{path}}")

    build = tmp_path / "build"
    build.mkdir()
    # three images: rows 3, 77 and 118 of the corpus at slots 1, 3 and 5; a fourth points at the wrong slot
    picks = [(3, 1, 0), (77, 3, 0), (118, 5, 0)]
    images, mint_rows = [], []
    for row, slot, rank in picks:
        url = table.column("images")[row][slot].as_py()
        page = f"https://page.example/{row}"
        images.append(
            {
                "image_id": image_id_from_url(url),
                "image_url": url,
                "page_url": page,
            }
        )
        mint_rows.append(
            {
                "image_id": image_id_from_url(url),
                "mint_rank": rank,
                "mint_file": name,
                "mint_row": row,
                "mint_image_idx": slot,
                "mint_num_slots": 6,
                "mint_doc_idx": 1000 + row,
                "page_url": page,
            }
        )
    # an image with no MINT document at all
    none_url = "https://img.example/none.jpg"
    images.append(
        {
            "image_id": image_id_from_url(none_url),
            "image_url": none_url,
            "page_url": None,
        }
    )
    pq.write_table(pa.Table.from_pylist(images, schema=IMAGES), build / "wire_images.parquet")
    pq.write_table(pa.Table.from_pylist(mint_rows, schema=MINT), build / "wire_mint.parquet")
    all_pages = page_rows(serve_dir / name, name)
    pq.write_table(pa.Table.from_pylist(all_pages, schema=MINT_PAGES), build / "wire_mint_pages.parquet")

    report = build_mint_documents(build, concurrency=4, token=None)
    docs = {r["image_id"]: r for r in pq.read_table(build / "wire_documents.parquet").to_pylist()}
    assert report["page_status"] == {"mint": 3} and report["rows_with_doc_text"] == 3 and report["files"] == 1
    for row, slot, _ in picks:
        d = docs[image_id_from_url(table.column("images")[row][slot].as_py())]
        assert d["image_found"] == "mint" and d["prev_text"] == f"doc {row} paragraph {slot - 1} " + "y" * (row % 17)
        assert d["caption"] == f"alt {row} {slot}"
        if slot < 5:
            assert d["next_text"].startswith(f"doc {row} paragraph {slot + 1}")
        else:
            assert d["next_text"] is None
    assert docs[image_id_from_url(none_url)]["page_status"] == "no_mint_document"

    # only pages were fetched, never the whole file: one resolve plus a few range requests per document
    assert all(h.endswith(name) for h in RangeHandler.hits)
    assert 1 < len(RangeHandler.hits) <= 1 + len(picks) * 2 * 2 * 2  # texts and metadata pages
    assert len([p for p in all_pages if p["column"] == "texts"]) > 3
    assert len([p for p in all_pages if p["column"] == "metadata"]) > 1

    # a stale pointer (a text slot instead of an image slot) is reported, not silently accepted
    bad = dict(mint_rows[0], mint_image_idx=0)
    pq.write_table(pa.Table.from_pylist([bad], schema=MINT), build / "wire_mint.parquet")
    (build / "wire_documents.parquet").unlink()
    with pytest.raises(SystemExit, match="do not match"):
        build_mint_documents(build, concurrency=4, token=None)
    server.shutdown()
