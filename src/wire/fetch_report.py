# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from wire.fetch import CLASSES, summarize

DESCRIPTIONS = {
    "ok": "fetched (or already on disk)",
    "gone": "404 / 410 / host no longer resolves: the image no longer exists at that URL",
    "blocked": "403 / 401 / 402 / X-Robots-Tag directive: the host refuses automated access",
    "rate_limited": "429 after every retry: the host throttled the rebuild",
    "server_or_network": "5xx, other statuses, connection or TLS errors, timeouts, redirect loops: retry later",
    "not_usable": "the URL serves something that is not a decodable image, or a file above the size cap",
}


def render(report: dict) -> str:
    total = report["total"]
    lines = [
        f"images: {total}   fetched: {report['ok']} ({100 * report['ok'] / total:.1f}%)   failed: {report['failed']}",
        "",
        "| class | count | share | meaning |",
        "|---|---|---|---|",
    ]
    for name in ["ok", *CLASSES]:
        n = report["by_class"][name]
        lines.append(f"| {name} | {n} | {100 * n / total:.1f}% | {DESCRIPTIONS[name]} |")
    lines += ["", "| outcome | count |", "|---|---|"]
    lines += [f"| {o} | {n} |" for o, n in report["by_outcome"].items()]
    lines += ["", "| HTTP status | count |", "|---|---|"]
    lines += [f"| {s} | {n} |" for s, n in report["by_http_status"].items()]
    lines += ["", "| scheme | ok | failed |", "|---|---|---|"]
    lines += [f"| {s} | {c.get('ok', 0)} | {c.get('failed', 0)} |" for s, c in report["by_scheme"].items()]
    if report.get("recovered_by"):
        lines += ["", "| recovered by | images |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in report["recovered_by"].items()]
    if report.get("archive"):
        lines += ["", "| Wayback Machine | urls |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in report["archive"].items()]
    lines += ["", "| host | urls | outcomes |", "|---|---|---|"]
    for h in report["top_failing_hosts"]:
        outcomes = ", ".join(f"{k} {v}" for k, v in h.items() if k not in ("host", "total"))
        lines.append(f"| {h['host']} | {h['total']} | {outcomes} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize fetch_log.parquet: what was fetched and why the rest failed."
    )
    parser.add_argument("--out", type=Path, required=True, help="BUILD_DIR holding fetch_log.parquet")
    parser.add_argument("--markdown", type=Path, default=None, help="also write the tables as Markdown")
    args = parser.parse_args()
    records = pq.read_table(args.out / "fetch_log.parquet").to_pylist()
    report = summarize(records)
    (args.out / "fetch_report.json").write_text(json.dumps(report, indent=2) + "\n")
    text = render(report)
    if args.markdown:
        args.markdown.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
