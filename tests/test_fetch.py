# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import http.server
import io
import json
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

import wire.fetch as fetch
from wire.fetch import CLASSES, fetch_images, sniff_format, summarize
from wire.schema import IMAGES, image_id_from_url


def jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buf, format="JPEG")
    return buf.getvalue()


class Handler(http.server.BaseHTTPRequestHandler):
    """Each path exercises one outcome; a few depend on request headers or on how often they were hit."""

    hits: dict[str, int] = {}

    def log_message(self, *args):
        pass

    def send_image(self, body: bytes, extra: dict | None = None):
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        Handler.hits[path] = Handler.hits.get(path, 0) + 1
        if path == "/ok.jpg":
            self.send_image(jpeg_bytes())
        elif path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/ok.jpg")
            self.end_headers()
        elif path == "/html":
            body = b"<html><body>not an image</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/robots.jpg":
            self.send_image(jpeg_bytes(), {"X-Robots-Tag": "noai, noimageai"})
        elif path == "/big.jpg":
            self.send_image(jpeg_bytes() + b"\0" * 5000)
        elif path == "/truncated.jpg":
            self.send_image(jpeg_bytes()[:20])
        elif path == "/limited":
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
        elif path == "/flaky.jpg":
            if Handler.hits[path] < 2:
                self.send_response(503)
                self.end_headers()
            else:
                self.send_image(jpeg_bytes())
        elif path == "/very-flaky.jpg":
            if Handler.hits[path] < 5:
                self.send_response(503)
                self.end_headers()
            else:
                self.send_image(jpeg_bytes())
        elif path == "/needs-referer.jpg":
            if self.headers.get("Referer") and "Firefox" in self.headers.get("User-Agent", ""):
                self.send_image(jpeg_bytes())
            else:
                self.send_response(403)
                self.end_headers()
        elif path == "/wayback/available":
            # the first lookup is rate limited; `lost.jpg` has no snapshot, the other dead URLs have one
            if Handler.hits[path] == 1:
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.end_headers()
                return
            target = parse_qs(urlparse(self.path).query)["url"][0]
            closest = {"available": True, "timestamp": "20250630"} if not target.endswith("/lost.jpg") else None
            body = json.dumps({"archived_snapshots": {"closest": closest} if closest else {}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path.endswith("im_/http://dead-host.invalid/moved.jpg"):
            # a redirect capture: the archive replays it towards the archived target
            self.send_response(302)
            self.send_header("Location", path.replace("moved.jpg", "archived.jpg"))
            self.end_headers()
        elif path.startswith("/web/20250630im_/"):
            self.send_image(jpeg_bytes())
        else:
            code = {"/missing.jpg": 404, "/gone.jpg": 410, "/forbidden.jpg": 403}.get(path, 500)
            self.send_response(code)
            self.end_headers()


def serve():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def write_images(build: Path, urls: dict[str, str]) -> None:
    rows = [
        {
            "image_id": image_id_from_url(u),
            "image_url": u,
            "page_url": "https://page.example/",
        }
        for u in urls.values()
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=IMAGES), build / "wire_images.parquet")


def read_log(build: Path) -> dict[str, dict]:
    return {r["image_url"]: r for r in pq.read_table(build / "fetch_log.parquet").to_pylist()}


def test_sniff_format():
    assert sniff_format(jpeg_bytes()[:16]) == "jpeg"
    assert sniff_format(b"\x89PNG\r\n\x1a\n" + b"\0" * 8) == "png"
    assert sniff_format(b"<html>") is None


def test_first_pass_outcomes_and_report(tmp_path: Path):
    server = serve()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    urls = {
        "ok": f"{base}/ok.jpg",
        "redirect": f"{base}/redirect",
        "html": f"{base}/html",
        "robots": f"{base}/robots.jpg",
        "big": f"{base}/big.jpg",
        "truncated": f"{base}/truncated.jpg",
        "limited": f"{base}/limited",
        "flaky": f"{base}/flaky.jpg",
        "missing": f"{base}/missing.jpg",
        "gone": f"{base}/gone.jpg",
        "forbidden": f"{base}/forbidden.jpg",
        "error": f"{base}/error.jpg",
        "dns": "http://host.invalid/x.jpg",
    }
    write_images(tmp_path, urls)
    report = fetch_images(
        tmp_path,
        concurrency=8,
        per_host_interval=0.0,
        timeout=5.0,
        retries=2,
        max_bytes=4000,
        research_agent=False,
        patient_retries=0,
        archive=False,
    )
    by = {name: read_log(tmp_path)[u] for name, u in urls.items()}

    assert by["ok"]["outcome"] == "ok" and by["ok"]["format"] == "jpeg" and by["ok"]["width"] == 8
    assert (tmp_path / "images" / f"{by['ok']['image_id']}.jpg").read_bytes() == jpeg_bytes()
    assert by["redirect"]["outcome"] == "ok" and by["redirect"]["final_url"].endswith("/ok.jpg")
    assert by["html"]["outcome"] == "not_an_image"
    assert by["robots"]["outcome"] == "robots_directive"
    assert by["big"]["outcome"] == "too_large"
    assert by["truncated"]["outcome"] == "decode_error"
    assert by["limited"]["outcome"] == "http_429" and by["limited"]["attempts"] == 3
    assert by["flaky"]["outcome"] == "ok" and by["flaky"]["attempts"] == 2
    assert by["missing"]["outcome"] == "http_404" and by["missing"]["attempts"] == 1
    assert by["gone"]["outcome"] == "http_410"
    assert by["forbidden"]["outcome"] == "http_403"
    assert by["error"]["outcome"] == "http_5xx" and by["error"]["attempts"] == 3
    assert by["dns"]["outcome"] == "dns_failure"
    assert not list((tmp_path / "images").glob("*.part"))

    assert report["total"] == len(urls) and report["ok"] == 3 and report["failed"] == len(urls) - 3
    assert (
        report["by_class"]["gone"] == 3
        and report["by_class"]["blocked"] == 2
        and report["by_class"]["rate_limited"] == 1
    )
    assert report["by_class"]["not_usable"] == 3 and report["by_class"]["server_or_network"] == 1
    assert report["by_http_status"]["404"] == 1
    assert set(report["by_class"]) == set(CLASSES) | {"ok"}

    # A second run resumes: fetched files are reported as cached and failures are kept without re-requesting them.
    Handler.hits.clear()
    report2 = fetch_images(
        tmp_path,
        concurrency=8,
        per_host_interval=0.0,
        timeout=5.0,
        retries=0,
        max_bytes=4000,
        research_agent=False,
        patient_retries=0,
        archive=False,
    )
    assert report2["by_outcome"]["cached"] == 3 and not Handler.hits
    assert summarize(pq.read_table(tmp_path / "fetch_log.parquet").to_pylist())["failed"] == report["failed"]
    json.loads(json.dumps(report2))
    server.shutdown()


def test_recovery_passes(tmp_path: Path, monkeypatch):
    server = serve()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setattr(fetch, "WAYBACK_AVAILABLE", f"{base}/wayback/available")
    monkeypatch.setattr(fetch, "WAYBACK_REPLAY", base + "/web/{timestamp}im_/{url}")
    urls = {
        "referer": f"{base}/needs-referer.jpg",
        "very_flaky": f"{base}/very-flaky.jpg",
        "archived": "http://dead-host.invalid/archived.jpg",
        "moved": "http://dead-host.invalid/moved.jpg",
        "lost": "http://dead-host.invalid/lost.jpg",
        "forbidden": f"{base}/forbidden.jpg",
    }
    write_images(tmp_path, urls)
    report = fetch_images(
        tmp_path,
        concurrency=8,
        per_host_interval=0.0,
        timeout=5.0,
        retries=0,
        max_bytes=4000,
        research_agent=True,
        patient_retries=5,
        archive=True,
        patient_backoff=0.01,
        archive_backoff=0.01,
    )
    by = {name: read_log(tmp_path)[u] for name, u in urls.items()}

    assert by["referer"]["outcome"] == "ok" and by["referer"]["recovered_by"] == "research_agent"
    assert by["very_flaky"]["outcome"] == "ok" and by["very_flaky"]["recovered_by"] == "patient_retry"
    assert by["archived"]["outcome"] == "ok" and by["archived"]["recovered_by"] == "archive"
    assert by["archived"]["archive_outcome"] == "archive_ok" and "/web/20250630im_/" in by["archived"]["final_url"]
    assert by["moved"]["outcome"] == "ok"
    assert by["moved"]["final_url"].endswith("im_/http://dead-host.invalid/archived.jpg")
    assert by["lost"]["outcome"] == "dns_failure" and by["lost"]["archive_outcome"] == "archive_missing"
    assert by["forbidden"]["outcome"] == "http_403" and by["forbidden"]["recovered_by"] is None
    assert report["recovered_by"] == {"research_agent": 1, "patient_retry": 1, "archive": 2}
    assert report["recovered_by_pass"] == {"research_agent": 1, "patient_retry": 1, "archive": 2}
    assert report["archive"] == {"archive_ok": 2, "archive_missing": 1}
    assert Handler.hits["/wayback/available"] == 4  # one 429, then one lookup per dead URL

    # A later run keeps the earlier rows of fetched images (attribution, HTTP details) and does not ask the
    # archive again about a URL it has no capture of.
    Handler.hits.clear()
    report2 = fetch_images(
        tmp_path,
        concurrency=8,
        per_host_interval=0.0,
        timeout=5.0,
        retries=0,
        max_bytes=4000,
        research_agent=False,
        patient_retries=0,
        archive=True,
        archive_backoff=0.01,
    )
    by = {name: read_log(tmp_path)[u] for name, u in urls.items()}
    assert by["archived"]["outcome"] == "cached" and by["archived"]["recovered_by"] == "archive"
    assert by["referer"]["recovered_by"] == "research_agent" and by["referer"]["http_status"] == 200
    assert report2["recovered_by"] == report["recovered_by"] and report2["archive"] == report["archive"]
    assert "/wayback/available" not in Handler.hits
    server.shutdown()
