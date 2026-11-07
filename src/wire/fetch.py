# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import asyncio
import collections
import datetime
import hashlib
import io
import json
import re
import socket
import ssl
import time
import warnings
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Large originals are fine: the byte cap bounds memory, Pillow's pixel cap would only reject them as bombs.
Image.MAX_IMAGE_PIXELS = 512_000_000
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

FETCH_LOG = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("image_url", pa.string(), nullable=False),
        pa.field("final_url", pa.string(), nullable=True),
        pa.field("status", pa.string(), nullable=False),
        pa.field("outcome", pa.string(), nullable=False),
        pa.field("recovered_by", pa.string(), nullable=True),
        pa.field("archive_outcome", pa.string(), nullable=True),
        pa.field("http_status", pa.int16(), nullable=True),
        pa.field("content_type", pa.string(), nullable=True),
        pa.field("bytes", pa.int64(), nullable=True),
        pa.field("md5", pa.string(), nullable=True),
        pa.field("format", pa.string(), nullable=True),
        pa.field("width", pa.int32(), nullable=True),
        pa.field("height", pa.int32(), nullable=True),
        pa.field("error", pa.string(), nullable=True),
        pa.field("attempts", pa.int8(), nullable=False),
        pa.field("elapsed_ms", pa.int32(), nullable=True),
        pa.field("fetched_at", pa.string(), nullable=False),
    ]
)

# Every failure lands in exactly one of these; the report groups them into the classes below.
OUTCOMES = (
    "ok",
    "cached",
    "http_404",
    "http_410",
    "http_403",
    "http_401",
    "http_402",
    "http_429",
    "http_4xx",
    "http_5xx",
    "http_other",
    "dns_failure",
    "connect_error",
    "ssl_error",
    "timeout",
    "too_many_redirects",
    "protocol_error",
    "robots_directive",
    "not_an_image",
    "decode_error",
    "too_large",
    "empty",
)
CLASSES = {
    "gone": ("http_404", "http_410", "dns_failure"),
    "blocked": ("http_403", "http_401", "http_402", "robots_directive"),
    "rate_limited": ("http_429",),
    "server_or_network": (
        "http_5xx",
        "http_other",
        "http_4xx",
        "connect_error",
        "ssl_error",
        "timeout",
        "too_many_redirects",
        "protocol_error",
    ),
    "not_usable": ("not_an_image", "decode_error", "too_large", "empty"),
}
TRANSIENT = {"http_429", "http_5xx", "timeout", "connect_error", "dns_failure", "protocol_error"}
# Recovery passes after the first attempt, each restricted to the outcomes it can plausibly fix.
BLOCKED_RETRY = ("http_403", "http_401", "http_402", "http_4xx", "not_an_image")
PATIENT_RETRY = ("http_5xx", "http_other", "http_429", "timeout", "connect_error", "protocol_error", "ssl_error")
ARCHIVE_DEFAULT = ("dns_failure", "http_404", "http_410")
ROBOTS_RE = re.compile(r"\b(noai|noimageai|noindex)\b", re.I)

USER_AGENT = "wire-benchmark-rebuild/1.0 (dataset rebuild; contact via the WIRE repository)"
RESEARCH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0 "
    "(compatible; WIRE-research-crawler/1.0; academic dataset reconstruction)"
)
WAYBACK_AVAILABLE = "https://archive.org/wayback/available"
WAYBACK_TIMESTAMP = "20250701"  # the annotation period; the closest snapshot to it is requested
# `im_` replays the capture as stored, without the toolbar or rewriting; a redirect capture is followed to the
# archived target instead of the live host (which `id_`, the raw mode, would do).
WAYBACK_REPLAY = "https://web.archive.org/web/{timestamp}im_/{url}"
ARCHIVE_BLOCK_PAUSE = 300.0  # archive.org answers 429 without Retry-After when it blocks a client for a few minutes
ARCHIVE_BLOCK_WAITS = 5  # how often one lookup waits out such a block before it is given up as `archive_error`


def sniff_format(head: bytes) -> str | None:
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:2] == b"BM":
        return "bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head[4:12] in (b"ftypavif", b"ftypheic", b"ftypmif1"):
        return "avif"
    return None


def classify_status(status: int) -> str:
    if status in (404, 410, 403, 401, 402, 429):
        return f"http_{status}"
    if 500 <= status < 600:
        return "http_5xx"
    if 400 <= status < 500:
        return "http_4xx"
    return "http_other"


def classify_exception(exc: BaseException) -> tuple[str, str]:
    message = f"{type(exc).__name__}: {exc}"[:200]
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout", message
    if isinstance(exc, aiohttp.TooManyRedirects):
        return "too_many_redirects", message
    if isinstance(exc, aiohttp.ClientConnectorCertificateError | aiohttp.ClientSSLError | ssl.SSLError):
        return "ssl_error", message
    if isinstance(exc, aiohttp.ClientConnectorError):
        if isinstance(exc.os_error, socket.gaierror):
            return "dns_failure", message
        return "connect_error", message
    if isinstance(exc, aiohttp.ClientPayloadError | aiohttp.ClientResponseError | aiohttp.ClientError):
        return "protocol_error", message
    if isinstance(exc, OSError):
        return "connect_error", message
    return "protocol_error", message


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_record(image_id: str, url: str) -> dict:
    return {
        "image_id": image_id,
        "image_url": url,
        "final_url": None,
        "status": "failed",
        "outcome": "protocol_error",
        "recovered_by": None,
        "archive_outcome": None,
        "http_status": None,
        "content_type": None,
        "bytes": None,
        "md5": None,
        "format": None,
        "width": None,
        "height": None,
        "error": None,
        "attempts": 0,
        "elapsed_ms": None,
        "fetched_at": _now(),
    }


class HostGate:
    """Serialize requests per host, `interval` seconds apart, and cap the number of requests in flight overall.

    The host lock is taken before the global slot, so tasks queued behind a busy host do not occupy slots that
    other hosts could use.
    """

    def __init__(self, concurrency: int, interval: float):
        self.global_slots = asyncio.Semaphore(concurrency)
        self.interval = interval
        self.locks: dict[str, asyncio.Lock] = collections.defaultdict(asyncio.Lock)
        self.last: dict[str, float] = {}

    async def __call__(self, host: str):
        lock = self.locks[host]
        await lock.acquire()
        wait = self.interval - (time.monotonic() - self.last.get(host, 0.0))
        if wait > 0:
            await asyncio.sleep(wait)
        await self.global_slots.acquire()
        return lock

    def pause(self, host: str, seconds: float) -> None:
        """Hold back every further request to `host` for `seconds` (a 429 from it means the whole client is blocked)."""
        self.last[host] = max(self.last.get(host, 0.0), time.monotonic() + seconds - self.interval)

    def release(self, host: str, lock: asyncio.Lock) -> None:
        self.last[host] = time.monotonic()
        self.global_slots.release()
        lock.release()


class ImageFetcher:
    """One download pass over a list of items ({image_id, image_url, page_url}) with a given politeness policy."""

    def __init__(
        self,
        images_dir: Path,
        concurrency: int = 64,
        per_host_interval: float = 0.2,
        timeout: float = 30.0,
        max_bytes: int = 32 << 20,
        retries: int = 2,
        backoff_base: float = 1.0,
        max_retry_after: float = 60.0,
        user_agent: str = USER_AGENT,
        send_referer: bool = False,
    ):
        self.images_dir = images_dir
        self.gate = HostGate(concurrency, per_host_interval)
        self.timeout = aiohttp.ClientTimeout(total=timeout * 3, connect=timeout, sock_read=timeout)
        self.max_bytes = max_bytes
        self.retries = retries
        self.backoff_base = backoff_base
        self.max_retry_after = max_retry_after
        self.user_agent = user_agent
        self.send_referer = send_referer
        self.concurrency = concurrency

    def _headers(self, item: dict) -> dict:
        headers = {"User-Agent": self.user_agent, "Accept": "image/*,*/*;q=0.5"}
        if self.send_referer and item.get("page_url"):
            headers["Referer"] = item["page_url"]
        return headers

    async def fetch_one(self, session: aiohttp.ClientSession, item: dict) -> dict:
        record = new_record(item["image_id"], item["image_url"])
        url = item["image_url"]
        host = urlparse(url).netloc.lower()
        started = time.monotonic()
        backoff = self.backoff_base
        for attempt in range(1, self.retries + 2):
            record["attempts"] = attempt
            lock = await self.gate(host)
            try:
                retry_after = await self._download(session, url, self._headers(item), record)
            finally:
                self.gate.release(host, lock)
            if record["status"] == "ok" or record["outcome"] not in TRANSIENT or attempt > self.retries:
                break
            await asyncio.sleep(min(retry_after or backoff, self.max_retry_after))
            backoff *= 4
        record["elapsed_ms"] = int(1000 * (time.monotonic() - started))
        return record

    async def _find_snapshot(self, session: aiohttp.ClientSession, item: dict, record: dict) -> str | None:
        """Timestamp of the snapshot closest to the annotation period, None when the archive has none.

        A 429 means archive.org has blocked the client for a while: the whole pass pauses and the lookup is repeated
        without spending an attempt. 5xx answers and network errors are retried with back-off; any other failure is
        recorded as `archive_error`."""
        params = {"url": item["image_url"], "timestamp": WAYBACK_TIMESTAMP}
        api_host = urlparse(WAYBACK_AVAILABLE).netloc
        backoff = self.backoff_base
        attempt = blocked = 0
        while True:
            lock = await self.gate(api_host)
            try:
                async with session.get(WAYBACK_AVAILABLE, params=params, headers=self._headers(item)) as response:
                    status = response.status
                    if status == 200:
                        snapshot = (await response.json(content_type=None)).get("archived_snapshots", {}).get("closest")
                        if not snapshot or not snapshot.get("available"):
                            record["archive_outcome"], record["error"] = "archive_missing", "no snapshot"
                            return None
                        return snapshot["timestamp"]
                    record["archive_outcome"], record["error"] = "archive_error", f"availability api {status}"
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            except Exception as exc:  # network and protocol failures are data here, not bugs
                record["archive_outcome"], record["error"] = "archive_error", classify_exception(exc)[1]
                status, retry_after = None, None
            finally:
                self.gate.release(api_host, lock)
            if status == 429 and blocked < ARCHIVE_BLOCK_WAITS:
                blocked += 1
                self.gate.pause(api_host, ARCHIVE_BLOCK_PAUSE if retry_after is None else retry_after)
                continue
            attempt += 1
            record["attempts"] = attempt
            if not (status is None or status == 429 or status >= 500) or attempt > self.retries:
                return None
            await asyncio.sleep(min(retry_after or backoff, self.max_retry_after))
            backoff *= 4

    async def fetch_archived(self, session: aiohttp.ClientSession, item: dict) -> dict:
        """Try the Wayback Machine: locate the snapshot closest to the annotation period, then replay it."""
        record = new_record(item["image_id"], item["image_url"])
        started = time.monotonic()
        timestamp = await self._find_snapshot(session, item, record)
        if timestamp is None:
            record["elapsed_ms"] = int(1000 * (time.monotonic() - started))
            return record
        replay_url = WAYBACK_REPLAY.format(timestamp=timestamp, url=item["image_url"])
        replay_host = urlparse(replay_url).netloc
        lock = await self.gate(replay_host)
        try:
            retry_after = await self._download(session, replay_url, self._headers(item), record)
        finally:
            self.gate.release(replay_host, lock)
        if record["http_status"] == 429:
            self.gate.pause(replay_host, ARCHIVE_BLOCK_PAUSE if retry_after is None else retry_after)
        record["archive_outcome"] = "archive_ok" if record["status"] == "ok" else "archive_error"
        record["elapsed_ms"] = int(1000 * (time.monotonic() - started))
        return record

    async def _download(self, session: aiohttp.ClientSession, url: str, headers: dict, record: dict) -> float | None:
        try:
            async with session.get(url, headers=headers, allow_redirects=True, max_redirects=5) as response:
                record["http_status"] = response.status
                record["final_url"] = str(response.url)
                record["content_type"] = response.headers.get("Content-Type")
                if response.status != 200:
                    record["outcome"] = classify_status(response.status)
                    record["error"] = response.reason
                    return _parse_retry_after(response.headers.get("Retry-After"))
                robots = response.headers.get("X-Robots-Tag", "")
                if ROBOTS_RE.search(robots):
                    record["outcome"], record["error"] = "robots_directive", robots[:100]
                    return None
                body = bytearray()
                async for chunk in response.content.iter_chunked(1 << 16):
                    body.extend(chunk)
                    if len(body) > self.max_bytes:
                        record["outcome"], record["error"] = "too_large", f"> {self.max_bytes} bytes"
                        return None
        except Exception as exc:  # network and protocol failures are data here, not bugs
            record["outcome"], record["error"] = classify_exception(exc)
            return None
        if not body:
            record["outcome"], record["error"] = "empty", "zero-length body"
            return None
        fmt = sniff_format(bytes(body[:16]))
        if fmt is None:
            record["outcome"] = "not_an_image"
            record["error"] = f"content-type {record['content_type']}, head {bytes(body[:8])!r}"
            return None
        try:
            with Image.open(io.BytesIO(body)) as image:
                record["width"], record["height"] = image.size
                image.verify()
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError) as exc:
            record["outcome"], record["error"] = "decode_error", f"{type(exc).__name__}: {exc}"[:200]
            return None
        target = self.images_dir / f"{record['image_id']}.jpg"
        tmp = target.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(target)
        record.update(
            status="ok", outcome="ok", bytes=len(body), md5=hashlib.md5(body).hexdigest(), format=fmt, error=None
        )
        return None

    async def run(self, items: list[dict], desc: str, archive: bool = False) -> list[dict]:
        connector = aiohttp.TCPConnector(limit=self.concurrency, limit_per_host=1, ttl_dns_cache=3600, ssl=False)
        results: list[dict] = []
        async with aiohttp.ClientSession(connector=connector, timeout=self.timeout) as session:
            worker = self.fetch_archived if archive else self.fetch_one
            tasks = [asyncio.create_task(worker(session, item)) for item in items]
            outcomes: collections.Counter[str] = collections.Counter()
            with tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc) as progress:
                for done in progress:
                    record = await done
                    results.append(record)
                    if archive:
                        outcomes[record["archive_outcome"]] += 1
                        progress.set_postfix(outcomes, refresh=False)
        return results


def write_log(records: dict[str, dict], path: Path) -> None:
    ordered = [records[k] for k in sorted(records)]
    pq.write_table(pa.Table.from_pylist(ordered, schema=FETCH_LOG), path, compression="zstd")


def cached_record(image_id: str, url: str, path: Path) -> dict:
    body = path.read_bytes()
    record = new_record(image_id, url)
    record.update(
        status="ok",
        outcome="cached",
        bytes=len(body),
        md5=hashlib.md5(body).hexdigest(),
        format=sniff_format(body[:16]),
    )
    return record


def _merge(records: dict[str, dict], results: list[dict], recovered_by: str | None) -> int:
    """Take a new result when it succeeded or when it is the first we have; otherwise keep the earlier outcome
    and only carry over the archive note and the attempt count."""
    recovered = 0
    for result in results:
        previous = records.get(result["image_id"])
        if result["status"] == "ok":
            if previous is not None and previous["status"] != "ok":
                result["recovered_by"] = recovered_by
                recovered += 1
            records[result["image_id"]] = result
        elif previous is None:
            records[result["image_id"]] = result
        else:
            previous["archive_outcome"] = result["archive_outcome"] or previous["archive_outcome"]
            previous["attempts"] = min(127, previous["attempts"] + result["attempts"])
    return recovered


def fetch_images(
    out: Path,
    concurrency: int = 64,
    per_host_interval: float = 0.2,
    timeout: float = 30.0,
    retries: int = 2,
    max_bytes: int = 32 << 20,
    refetch_failed: bool = False,
    research_agent: bool = True,
    patient_retries: int = 5,
    archive: bool = True,
    archive_outcomes: tuple[str, ...] = ARCHIVE_DEFAULT,
    patient_backoff: float = 10.0,
    archive_backoff: float = 30.0,
) -> dict:
    """Fetch every image of `wire_images.parquet` into `images/{image_id}.jpg`, resuming what is already on disk.

    Passes: (1) plain download with the project's user agent; (2) refusals retried with a browser-like research
    user agent and the parent page as referrer; (3) rate limits and server or network errors retried patiently,
    with long back-off and `Retry-After` honoured up to five minutes; (4) URLs whose host is gone (and, by
    default, 404/410) looked up in the Wayback Machine, closest snapshot to the annotation period.

    Keep concurrency modest: on AWS the VPC resolver caps DNS at roughly 1,000 queries per second per interface,
    and beyond that live hosts fail with `dns_failure`. One request at a time per host, `per_host_interval`
    seconds apart, is the politeness floor; large image CDNs receive a few requests per second at most.
    """
    images = pq.read_table(out / "wire_images.parquet", columns=["image_id", "image_url", "page_url"]).to_pylist()
    images_dir = out / "images"
    images_dir.mkdir(exist_ok=True)
    log_path = out / "fetch_log.parquet"
    previous = {r["image_id"]: r for r in pq.read_table(log_path).to_pylist()} if log_path.exists() else {}
    by_id = {r["image_id"]: r for r in images}

    records: dict[str, dict] = {}
    todo: list[dict] = []
    for r in images:
        path = images_dir / f"{r['image_id']}.jpg"
        if path.exists() and previous.get(r["image_id"], {}).get("status") == "ok":
            # keep the earlier log row (HTTP details, dimensions, which pass recovered it); only the outcome changes
            records[r["image_id"]] = dict(previous[r["image_id"]], outcome="cached")
        elif path.exists():
            records[r["image_id"]] = cached_record(r["image_id"], r["image_url"], path)
        elif r["image_id"] in previous and not refetch_failed:
            records[r["image_id"]] = previous[r["image_id"]]
        else:
            todo.append(r)
    print(f"{len(images)} images: {len(records)} already settled, {len(todo)} to fetch", flush=True)
    passes: dict[str, int] = {}

    def pending(outcomes, skip_archive_outcome: str | None = None) -> list[dict]:
        return [
            by_id[i]
            for i, rec in records.items()
            if rec["status"] != "ok"
            and rec["outcome"] in outcomes
            and (skip_archive_outcome is None or rec["archive_outcome"] != skip_archive_outcome)
        ]

    if todo:
        fetcher = ImageFetcher(images_dir, concurrency, per_host_interval, timeout, max_bytes, retries)
        _merge(records, asyncio.run(fetcher.run(todo, "fetch images")), None)
        write_log(records, log_path)
    if research_agent and (items := pending(BLOCKED_RETRY)):
        fetcher = ImageFetcher(
            images_dir,
            concurrency,
            max(per_host_interval, 0.5),
            timeout,
            max_bytes,
            retries=1,
            user_agent=RESEARCH_USER_AGENT,
            send_referer=True,
        )
        passes["research_agent"] = _merge(
            records, asyncio.run(fetcher.run(items, "retry refusals (research agent)")), "research_agent"
        )
        write_log(records, log_path)
    if patient_retries and (items := pending(PATIENT_RETRY)):
        fetcher = ImageFetcher(
            images_dir,
            concurrency,
            max(per_host_interval, 2.0),
            timeout,
            max_bytes,
            retries=patient_retries,
            backoff_base=patient_backoff,
            max_retry_after=300.0,
            user_agent=RESEARCH_USER_AGENT,
            send_referer=True,
        )
        passes["patient_retry"] = _merge(
            records, asyncio.run(fetcher.run(items, "retry transient (patient)")), "patient_retry"
        )
        write_log(records, log_path)
    if archive and (items := pending(archive_outcomes, skip_archive_outcome="archive_missing")):
        # archive.org asks for gentle use: one request at a time, at least a second apart, and a 429 pauses the
        # whole pass. It flags the browser-like research user agent as suspicious and blocks it, so this pass
        # identifies itself plainly. A URL the archive has no snapshot of is not asked again on a later run, one
        # whose lookup or download failed is.
        fetcher = ImageFetcher(
            images_dir,
            concurrency=4,
            per_host_interval=max(per_host_interval, 1.0),
            timeout=timeout,
            max_bytes=max_bytes,
            retries=2,
            backoff_base=archive_backoff,
            max_retry_after=300.0,
        )
        passes["archive"] = _merge(records, asyncio.run(fetcher.run(items, "wayback machine", archive=True)), "archive")
        write_log(records, log_path)
    write_log(records, log_path)
    report = summarize(list(records.values()))
    report["recovered_by_pass"] = passes
    return report


def summarize(records: list[dict]) -> dict:
    """Counts by outcome, by class, by HTTP status, by scheme and by recovery pass, plus the worst hosts."""
    by_outcome = collections.Counter(r["outcome"] for r in records)
    by_class = {name: sum(by_outcome[o] for o in outcomes) for name, outcomes in CLASSES.items()}
    by_class["ok"] = by_outcome["ok"] + by_outcome["cached"]
    by_status = collections.Counter(str(r["http_status"]) for r in records if r["http_status"] is not None)
    by_scheme: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    hosts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in records:
        parsed = urlparse(r["image_url"])
        by_scheme[parsed.scheme][r["status"]] += 1
        hosts[parsed.netloc.lower()][r["outcome"]] += 1
    failing_hosts = sorted(
        hosts.items(), key=lambda kv: -sum(v for o, v in kv[1].items() if o not in ("ok", "cached"))
    )[:25]
    return {
        "total": len(records),
        "ok": by_class["ok"],
        "failed": len(records) - by_class["ok"],
        "by_class": by_class,
        "by_outcome": dict(by_outcome.most_common()),
        "by_http_status": dict(by_status.most_common()),
        "by_scheme": {k: dict(v) for k, v in by_scheme.items()},
        "recovered_by": dict(collections.Counter(r["recovered_by"] for r in records if r["recovered_by"])),
        "archive": dict(collections.Counter(r["archive_outcome"] for r in records if r["archive_outcome"])),
        "top_failing_hosts": [{"host": h, "total": sum(c.values()), **dict(c.most_common())} for h, c in failing_hosts],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the WIRE images from their original URLs and log every outcome."
    )
    parser.add_argument("--out", type=Path, required=True, help="BUILD_DIR holding wire_images.parquet")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--per-host-interval", type=float, default=0.2, help="seconds between two requests to one host")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--retries", type=int, default=2, help="extra attempts for transient outcomes in the first pass"
    )
    parser.add_argument("--max-bytes", type=int, default=32 << 20)
    parser.add_argument("--refetch-failed", action="store_true", help="retry URLs that failed in a previous run")
    parser.add_argument("--no-research-agent", action="store_true", help="skip the research-agent pass over refusals")
    parser.add_argument("--patient-retries", type=int, default=5, help="attempts of the patient pass; 0 disables it")
    parser.add_argument("--no-archive", action="store_true", help="skip the Wayback Machine pass")
    parser.add_argument("--archive-outcomes", nargs="+", default=list(ARCHIVE_DEFAULT), choices=OUTCOMES)
    args = parser.parse_args()
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
        tuple(args.archive_outcomes),
    )
    (args.out / "fetch_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "top_failing_hosts"}, indent=2))


if __name__ == "__main__":
    main()
