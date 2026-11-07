# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import collections
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import pyarrow as pa
import pyarrow.parquet as pq
import trafilatura.downloads
import urllib3
from lxml import html
from trafilatura.downloads import add_to_compressed_dict, buffered_downloads, load_download_buffer

from wire.schema import image_id_from_url

DOCUMENTS = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("page_url", pa.string(), nullable=False),
        pa.field("page_status", pa.string(), nullable=False),
        pa.field("image_found", pa.string(), nullable=False),
        pa.field("caption", pa.string(), nullable=True),
        pa.field("page_title", pa.string(), nullable=True),
        pa.field("prev_text", pa.string(), nullable=True),
        pa.field("next_text", pa.string(), nullable=True),
        pa.field("doc_text", pa.string(), nullable=True),
    ]
)

BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "figcaption", "td", "th", "dd", "dt", "pre"}
SKIP_TAGS = {"script", "style", "noscript", "head", "svg", "template"}
WS_RE = re.compile(r"\s+")


def interleave(root, base_url: str) -> list[tuple[str, str, str]]:
    """Walk the page in document order and emit ("text", paragraph, "") and ("img", absolute src, alt) items.

    This mirrors the interleaved text/image layout the original corpus stored, so the nearest paragraph
    before and after an image is defined the same way the published evaluation defined it.
    """
    items: list[tuple[str, str, str]] = []
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        tag = el.tag.lower()
        if tag in SKIP_TAGS or any(a.tag in SKIP_TAGS for a in el.iterancestors() if isinstance(a.tag, str)):
            continue
        if tag == "img":
            src = el.get("src") or el.get("data-src") or ""
            if src:
                items.append(("img", urljoin(base_url, src.strip()), WS_RE.sub(" ", el.get("alt") or "").strip()))
        elif tag in BLOCK_TAGS:
            text = WS_RE.sub(" ", el.text_content()).strip()
            if text:
                items.append(("text", text, ""))
    return items


def surrounding_text(items: list[tuple[str, str, str]], index: int) -> tuple[str, str]:
    prev = next((t for kind, t, _ in reversed(items[:index]) if kind == "text"), "")
    nxt = next((t for kind, t, _ in items[index + 1 :] if kind == "text"), "")
    return prev, nxt


def extract(page_html: str, page_url: str, image_url: str, image_id: str) -> dict:
    root = html.fromstring(page_html)
    base = root.find(".//base")
    base_url = urljoin(page_url, base.get("href")) if base is not None and base.get("href") else page_url
    title_el = root.find(".//title")
    title = WS_RE.sub(" ", title_el.text_content()).strip() if title_el is not None else None
    items = interleave(root, base_url)
    matches = {"exact": None, "hash": None, "basename": None}
    basename = image_url.rsplit("/", 1)[-1].split("?")[0]
    for i, (kind, src, _) in enumerate(items):
        if kind != "img":
            continue
        if src == image_url and matches["exact"] is None:
            matches["exact"] = i
        elif image_id_from_url(src) == image_id and matches["hash"] is None:
            matches["hash"] = i
        elif basename and src.rsplit("/", 1)[-1].split("?")[0] == basename and matches["basename"] is None:
            matches["basename"] = i
    how = next((k for k in ("exact", "hash", "basename") if matches[k] is not None), "none")
    if how == "none":
        return {
            "page_title": title,
            "image_found": "none",
            "caption": None,
            "prev_text": None,
            "next_text": None,
            "doc_text": None,
        }
    prev, nxt = surrounding_text(items, matches[how])
    parts = [p for p in (prev, nxt) if p]
    return {
        "page_title": title,
        "image_found": how,
        "caption": items[matches[how]][2] or None,
        "prev_text": prev or None,
        "next_text": nxt or None,
        "doc_text": "\n".join(parts) if parts else None,
    }


def build_documents(out: Path, threads: int, sleep_time: float) -> dict:
    """Fetch every parent page once and recover title and the paragraphs around each image."""
    images = pq.read_table(out / "wire_images.parquet", columns=["image_id", "image_url", "page_url"]).to_pylist()
    by_page: dict[str, list[dict]] = collections.defaultdict(list)
    for r in images:
        if r["page_url"]:
            by_page[r["page_url"]].append(r)

    pages_dir = out / "_pages"
    pages_dir.mkdir(exist_ok=True)
    # trafilatura's default policy honours Retry-After without a cap; a handful of hosts asking for hours of
    # back-off would otherwise pin worker threads and stall every download buffer.
    trafilatura.downloads.RETRY_STRATEGY = urllib3.util.Retry(
        total=2,
        redirect=2,
        connect=0,
        backoff_factor=1,
        status_forcelist=trafilatura.downloads.FORCE_STATUS,
        respect_retry_after_header=False,
    )
    pending = [u for u in by_page if not (pages_dir / f"{image_id_from_url(u)}.html").exists()]
    print(f"{len(by_page)} pages, {len(pending)} to fetch", flush=True)
    url_store = add_to_compressed_dict(pending)
    fetched = failed = 0
    while url_store.done is False:
        buffer, url_store = load_download_buffer(url_store, sleep_time=sleep_time)
        for url, result in buffered_downloads(buffer, threads):
            target = pages_dir / f"{image_id_from_url(url)}.html"
            if result:
                target.write_text(result, errors="replace")
                fetched += 1
            else:
                target.with_suffix(".failed").touch()
                failed += 1
        print(f"  fetched {fetched} failed {failed}", flush=True)

    rows = []
    found = collections.Counter()
    for page_url, recs in by_page.items():
        page_file = pages_dir / f"{image_id_from_url(page_url)}.html"
        if not page_file.exists():
            for r in recs:
                rows.append(
                    {
                        "image_id": r["image_id"],
                        "page_url": page_url,
                        "page_status": "failed",
                        "image_found": "none",
                        "caption": None,
                        "page_title": None,
                        "prev_text": None,
                        "next_text": None,
                        "doc_text": None,
                    }
                )
                found["page_failed"] += 1
            continue
        page_html = page_file.read_text(errors="replace")
        for r in recs:
            try:
                info = extract(page_html, page_url, r["image_url"], r["image_id"])
            except (ValueError, TypeError) as exc:  # unparseable markup: lxml raises on empty or binary bodies
                info = {
                    "page_title": None,
                    "image_found": "none",
                    "caption": None,
                    "prev_text": None,
                    "next_text": None,
                    "doc_text": None,
                }
                found[f"parse_error:{type(exc).__name__}"] += 1
            found[info["image_found"]] += 1
            rows.append({"image_id": r["image_id"], "page_url": page_url, "page_status": "ok", **info})
    pq.write_table(pa.Table.from_pylist(rows, schema=DOCUMENTS), out / "wire_documents.parquet", compression="zstd")
    return {
        "pages": len(by_page),
        "fetched_now": fetched,
        "failed_now": failed,
        "image_found": dict(found),
        "rows_with_doc_text": sum(1 for r in rows if r["doc_text"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover page title and surrounding paragraphs for every WIRE image.")
    parser.add_argument("--out", type=Path, required=True, help="BUILD_DIR produced by wire-build")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--sleep-time", type=float, default=2.0, help="minimum seconds between two requests to one domain"
    )
    args = parser.parse_args()
    report = build_documents(args.out, args.threads, args.sleep_time)
    manifest_path = args.out / "BUILD_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"steps": {}}
    manifest["documents"] = report
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
