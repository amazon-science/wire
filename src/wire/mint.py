# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import asyncio
import collections
import json
import os
import re
import time
from pathlib import Path

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from wire.documents import DOCUMENTS, surrounding_text
from wire.parquet_pages import PageInfo, decode_page_levels, extract_rows, pages_for_row, read_page_header

MINT_REPO = "mlfoundations/MINT-1T-HTML"
MINT_REVISION = "906a8b85cea61198ff7339c4dd711ad0b5361847"  # the page table is bound to these exact files
RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
LIST_LEVELS = (1, 3)  # max repetition / definition level of MINT's optional list<optional string> columns
# texts: the interleaved text/image sequence, image slots are the null entries; metadata: a JSON list with one
# entry per image (`src`, `alt_text`, ...), the alt text being the caption
COLUMNS = ("texts", "metadata")
LEVELS = {"texts": LIST_LEVELS, "metadata": (0, 1)}
CAPTION_RE = re.compile(r"[\n\r\t\f\v]")
RETRIES = 6
CHECKPOINT_EVERY = 200  # files


class MintFetcher:
    """Range requests against the MINT-1T-HTML parquet files on the Hugging Face hub, a few at a time.

    Every file is resolved once to its CDN location; pages are then read with one HTTP range request each.
    429 and 5xx answers are retried with back-off, honouring Retry-After."""

    def __init__(self, session: aiohttp.ClientSession, concurrency: int, token: str | None):
        self.session = session
        self.slots = asyncio.Semaphore(concurrency)
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.bytes_fetched = 0
        self.requests = 0

    async def _get(self, url: str, headers: dict, allow_redirects: bool) -> tuple[int, dict, bytes]:
        backoff = 5.0
        for attempt in range(RETRIES + 1):
            async with self.slots:
                self.requests += 1
                try:
                    async with self.session.get(url, headers=headers, allow_redirects=allow_redirects) as response:
                        body = await response.read()
                        status, resp_headers = response.status, dict(response.headers)
                except aiohttp.ClientError as exc:
                    status, resp_headers, body = -1, {}, str(exc).encode()
            if status in (200, 206, 302, 303, 307) or (400 <= status < 500 and status != 429):
                return status, resp_headers, body
            if attempt == RETRIES:
                raise RuntimeError(f"{url}: HTTP {status} after {RETRIES} retries: {body[:200]!r}")
            retry_after = resp_headers.get("Retry-After")
            await asyncio.sleep(float(retry_after) if retry_after and retry_after.isdigit() else backoff)
            backoff = min(backoff * 2, 120.0)
        raise AssertionError("unreachable")

    async def resolve(self, path: str) -> str:
        url = RESOLVE_URL.format(repo=MINT_REPO, revision=MINT_REVISION, path=path)
        status, headers, body = await self._get(url, self.headers, allow_redirects=False)
        if status in (302, 303, 307):
            return headers["Location"]
        if status == 200:
            return url
        raise RuntimeError(f"{path}: cannot resolve on the hub (HTTP {status}): {body[:200]!r}")

    async def read_range(self, url: str, offset: int, size: int) -> bytes:
        headers = {"Range": f"bytes={offset}-{offset + size - 1}"}
        status, _, body = await self._get(url, headers, allow_redirects=True)
        if status != 206 or len(body) != size:
            raise RuntimeError(f"range {offset}+{size} of {url[:80]}: HTTP {status}, {len(body)} bytes")
        self.bytes_fetched += size
        return body


def load_page_tables(out: Path) -> tuple[dict[tuple[str, str], list[PageInfo]], dict[tuple[str, str], str]]:
    pages: dict[tuple[str, str], list[PageInfo]] = collections.defaultdict(list)
    codecs: dict[tuple[str, str], str] = {}
    for r in pq.read_table(out / "wire_mint_pages.parquet").to_pylist():
        key = (r["mint_file"], r["column"])
        pages[key].append(PageInfo(r["offset"], r["size"], r["first_row"], r["num_rows"], r["continues"]))
        codecs[key] = r["codec"]
    for key in pages:
        pages[key].sort(key=lambda p: p.offset)
    return pages, codecs


def caption_from_metadata(metadata: str | None, image_url: str) -> str:
    """The alt text MINT stored for the image, normalised the way the annotation pipeline did (line breaks to
    spaces, double quotes to single); empty when the document has none."""
    for entry in json.loads(metadata or "[]"):
        if entry and entry.get("src") == image_url and entry.get("alt_text") is not None:
            return CAPTION_RE.sub(" ", entry["alt_text"]).strip().replace('"', "'")
    return ""


def context_row(image_id: str, page_url: str, texts: list[str | None], image_idx: int, caption: str) -> dict:
    """The caption and the paragraph before and after the image slot, in the layout `wire-documents` produces."""
    items = [("text", t, "") if t is not None else ("img", "", "") for t in texts]
    prev, nxt = surrounding_text(items, image_idx)
    parts = [p for p in (prev, nxt) if p]
    return {
        "image_id": image_id,
        "page_url": page_url,
        "page_status": "mint",
        "image_found": "mint",
        "caption": caption,
        "page_title": None,
        "prev_text": prev or None,
        "next_text": nxt or None,
        "doc_text": "\n".join(parts) if parts else None,
    }


def failed_row(image_id: str, page_url: str, reason: str) -> dict:
    return {
        "image_id": image_id,
        "page_url": page_url,
        "page_status": reason,
        "image_found": "none",
        "caption": None,
        "page_title": None,
        "prev_text": None,
        "next_text": None,
        "doc_text": None,
    }


async def fetch_file(fetcher: MintFetcher, path: str, wanted: list[dict], pages, codecs) -> list[dict]:
    """All documents of one MINT file: fetch the pages their rows live in, decode, verify, extract the context."""
    cdn = await fetcher.resolve(path)
    needed: dict[tuple[str, int], PageInfo] = {}
    for w in wanted:
        for column in COLUMNS:
            for p in pages_for_row(pages[(path, column)], w["mint_row"]):
                needed[(column, p.offset)] = p
    raw = dict(
        zip(
            needed,
            await asyncio.gather(*(fetcher.read_range(cdn, p.offset, p.size) for p in needed.values())),
        )
    )
    decoded = {}
    for key, body in raw.items():
        header = read_page_header(body, 0)
        decoded[key] = decode_page_levels(header, body[header.header_size:], codecs[(path, key[0])], *LEVELS[key[0]])
    rows = []
    for w in wanted:
        values = {}
        for column in COLUMNS:
            chosen = pages_for_row(pages[(path, column)], w["mint_row"])
            values[column] = extract_rows(
                [decoded[(column, p.offset)] for p in chosen], chosen[0].first_row, [w["mint_row"]], *LEVELS[column]
            )[w["mint_row"]]
        texts, idx = values["texts"], w["mint_image_idx"]
        # the row must have the recorded layout: as many slots as when it was indexed, an image at the slot
        if texts is None or len(texts) != w["mint_num_slots"] or not 0 <= idx < len(texts) or texts[idx] is not None:
            rows.append(failed_row(w["image_id"], w["page_url"], "mint_layout_mismatch"))
        else:
            caption = caption_from_metadata(values["metadata"], w["image_url"])
            rows.append(context_row(w["image_id"], w["page_url"], texts, idx, caption))
    return rows


async def run(out: Path, concurrency: int, token: str | None) -> dict:
    images = {r["image_id"]: r for r in pq.read_table(out / "wire_images.parquet").to_pylist()}
    mint = pq.read_table(out / "wire_mint.parquet").to_pylist()
    pages, codecs = load_page_tables(out)
    target = out / "wire_documents.parquet"
    done: dict[str, dict] = {}
    if target.exists():
        done = {r["image_id"]: r for r in pq.read_table(target).to_pylist() if r["page_status"] == "mint"}

    # one document per image: the first one the annotation pipeline listed
    primary: dict[str, dict] = {}
    for r in sorted(mint, key=lambda r: (r["image_id"], r["mint_rank"])):
        primary.setdefault(r["image_id"], dict(r, image_url=images[r["image_id"]]["image_url"]))
    by_file: dict[str, list[dict]] = collections.defaultdict(list)
    for image_id, r in primary.items():
        if image_id not in done:
            by_file[r["mint_file"]].append(r)

    rows = dict(done)
    started = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_read=120)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        fetcher = MintFetcher(session, concurrency, token)
        for i, (path, wanted) in enumerate(tqdm(sorted(by_file.items()), desc="mint documents"), 1):
            for row in await fetch_file(fetcher, path, wanted, pages, codecs):
                rows[row["image_id"]] = row
            if i % CHECKPOINT_EVERY == 0:
                write_documents(out, rows, images, primary)
        bytes_fetched, requests = fetcher.bytes_fetched, fetcher.requests
    write_documents(out, rows, images, primary)
    status = collections.Counter(r["page_status"] for r in rows.values())
    return {
        "images": len(images),
        "images_with_mint_document": len(primary),
        "files": len(by_file),
        "requests": requests,
        "bytes_fetched": bytes_fetched,
        "elapsed_s": int(time.monotonic() - started),
        "page_status": dict(status),
        "rows_with_doc_text": sum(1 for r in rows.values() if r["doc_text"]),
    }


def write_documents(out: Path, rows: dict[str, dict], images: dict[str, dict], primary: dict[str, dict]) -> None:
    table = [
        rows.get(image_id) or failed_row(image_id, images[image_id]["page_url"] or "", "no_mint_document")
        for image_id in sorted(images)
        if image_id in rows or image_id not in primary
    ]
    pq.write_table(pa.Table.from_pylist(table, schema=DOCUMENTS), out / "wire_documents.parquet", compression="zstd")


def build_mint_documents(out: Path, concurrency: int = 8, token: str | None = None) -> dict:
    report = asyncio.run(run(out, concurrency, token))
    mismatches = sum(v for k, v in report["page_status"].items() if k.startswith("mint_"))
    if mismatches:
        raise SystemExit(
            f"{mismatches} images do not match the MINT-1T-HTML rows the release points at "
            f"(see page_status in wire_documents.parquet): the public files changed or the release tables are stale"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover the paragraphs around each image from the public MINT-1T-HTML parquet files, "
        "reading only the pages that hold the documents WIRE was built from."
    )
    parser.add_argument("--out", type=Path, required=True, help="BUILD_DIR produced by wire-build")
    parser.add_argument("--concurrency", type=int, default=8, help="range requests in flight")
    args = parser.parse_args()
    report = build_mint_documents(args.out, args.concurrency, os.environ.get("HF_TOKEN"))
    manifest_path = args.out / "BUILD_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"steps": {}}
    manifest["documents"] = dict(report, source="mint")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
