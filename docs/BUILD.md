# Rebuilding the full dataset

```bash
pip install -e '.[build]'
export HF_TOKEN=...                                  # huggingface.co token for the MINT step
wire-build --release data --out BUILD_DIR            # tables, Natural Questions join, images, MINT paragraphs
wire-documents --out BUILD_DIR                       # alternative to the MINT step: re-crawl the live pages
wire-fetch-report --out BUILD_DIR                    # why the missing images are missing
wire-verify --data data --build BUILD_DIR            # what was recovered, per split
wire-assemble --out BUILD_DIR                        # DatasetDict with an Image column, BUILD_DIR/wire_dataset
```

## Steps

**tables**: verifies `data/MANIFEST.json` and copies the five release tables into `BUILD_DIR`.

**nq**: downloads `lucadiliello/naturalquestionsshortqa` (62 MB) and attaches `question`, `answers`
and `context` to every query through `nq_key`, writing `wire_queries_full.parquet`. Pass
`--nq-revision` to pin a dataset revision. Unset any custom `HF_ENDPOINT` first.

**images**: `wire-fetch`, an asyncio downloader built on aiohttp. It identifies itself with a fixed
`User-Agent`, makes at most one request at a time per host with `--per-host-interval` seconds between
them (default 0.2 s, at most five requests per second to any host), caps the number of hosts in flight with `--concurrency` (default 64), follows up to
five redirects, refuses bodies above `--max-bytes` (default 32 MB; Pillow's pixel cap is lifted to 512 megapixels), honours `X-Robots-Tag: noai/noimageai/noindex`,
checks the magic bytes and decodes every file with Pillow before writing it atomically to
`images/{image_id}.jpg`. Only transient failures are retried (`--retries`, default 2): 429, 5xx,
timeouts, connection and DNS errors, with exponential back-off and a `Retry-After` header honoured up
to 60 s. Permanent failures (404, 410, 403, 401, robots directives, non-image or undecodable content)
are recorded once. The run checkpoints `fetch_log.parquet` every 2,000 images and resumes: files on
disk are kept, previously failed URLs are skipped unless `--refetch-failed` is given.

After the first pass, three recovery passes run over what failed, each restricted to the outcomes it can
plausibly fix (all can be switched off):

- **research agent** (`--no-research-agent` disables): 403/401/402, other 4xx and non-image responses are
  requested again with a browser-like research user agent (`WIRE-research-crawler`) and the parent page
  as `Referer`, since many CDNs refuse requests that arrive without either. Images whose `X-Robots-Tag`
  forbids reuse are never retried.
- **patient** (`--patient-retries`, default 5): 429, 5xx, timeouts and connection errors are retried with
  a 10 s base back-off growing fourfold per attempt, `Retry-After` honoured up to five minutes, at most one
  request every 2 s per host.
- **Wayback Machine** (`--no-archive` disables, `--archive-outcomes` selects; default DNS failures, 404,
  410): the snapshot closest to the annotation period (2025-07) is looked up through the availability API
  and replayed in image mode (`im_`), which serves the stored bytes and follows a redirect capture to its
  archived target rather than to the live host, one request at a time and at least a second apart, under
  the plain project user agent (archive.org flags the browser-like research one as suspicious).
  archive.org blocks a client for a few minutes once it asks too fast and answers 429 meanwhile: the
  whole pass then pauses for five minutes. The pass takes a few hours on thousands of dead URLs; it
  resumes, and a URL the archive has no snapshot of is not asked about again on a later run.

`recovered_by` records which pass produced the file (`research_agent`, `patient_retry`, `archive`);
`archive_outcome` records the Wayback lookup (`archive_ok`, `archive_missing`, `archive_error`).

Every URL gets one row in `fetch_log.parquet` with its `outcome` (`ok`, `cached`, `http_404`, `http_410`,
`http_403`, `http_401`, `http_402`, `http_429`, `http_4xx`, `http_5xx`, `http_other`, `dns_failure`,
`connect_error`, `ssl_error`, `timeout`, `too_many_redirects`, `protocol_error`, `robots_directive`,
`not_an_image`, `decode_error`, `too_large`, `empty`), the HTTP status, content type, final URL after
redirects, size, md5, decoded format and dimensions, the error text, the number of attempts and the
elapsed time. `wire-fetch-report --out BUILD_DIR [--markdown report.md]` groups the outcomes into
five classes (gone, blocked, rate limited, server or network, not usable), and breaks them down by HTTP
status, by URL scheme and by host, so link rot can be quantified instead of guessed.

Keep concurrency modest. On AWS the VPC resolver caps DNS at about 1,000 queries per second per
network interface; at a few thousand concurrent downloads more than half of the URLs fail with
`dns_failure` even though they are alive. The first pass finishes the 73,480 URLs in about 40 minutes;
the patient pass adds about an hour and the Wayback pass, serial by design, about three.

**mint** (`wire-mint`, run by default as the fourth step of `wire-build`): recovers the paragraphs
around each image from the public corpus the images were drawn from, `mlfoundations/MINT-1T-HTML`
(`data_v1_0`, pinned to one hub revision), without downloading it. Its parquet files hold one row group
each and no page index, so reaching a row would otherwise mean fetching the whole 0.5-2.5 GB file;
instead the release ships, for every image's document, the location of the parquet data pages of the
`texts` and `metadata` columns that hold it (`wire_mint_pages.parquet`, computed once from the public
files: 122,204 pages, 72 GB in total, 51 GB of paragraphs and 21 GB of metadata, about 1 MB per
image, 2% of the corpus). The step resolves each file to its hub location, fetches those pages with HTTP
range requests (`--mint-concurrency`, default 8, in flight), decodes them (RLE levels, PLAIN byte arrays,
snappy), checks that the row has the recorded number of slots with an image at the recorded one, and
stores the alt text MINT recorded for the image URL as `caption` (line breaks to spaces, double quotes to
single; empty for the 799 images whose page carries none) and the nearest
paragraph before and after the slot as `wire_documents.parquet` (`page_status` = `mint`;
`no_mint_document` for the 2 images without a page).
A stale pointer is reported as `mint_layout_mismatch` and the step exits non-zero rather than dropping
the row silently. Set `HF_TOKEN` to a huggingface.co token: the hub rate-limits anonymous requests and
the step makes one per page plus one per file (about 125,000 requests in all). On a machine in AWS us-east-1
with the default eight requests in flight the whole step took 1 h 31 min; raise `--mint-concurrency` to
shorten it. The run resumes from `wire_documents.parquet` if interrupted.

**documents** (`wire-documents`, the alternative to **mint**): fetches each distinct `page_url` once
through trafilatura's domain-aware downloader (`--sleep-time` seconds between two requests to the same
domain), locates the image on the page by exact URL, URL hash or file name, and stores the page title
and the nearest paragraph before and after the image as `wire_documents.parquet`. Pages that cannot be
fetched are retried on the next run. Live pages have drifted since the corpus was crawled, so this
recovers context for far fewer images than the MINT step.

## Verification

`wire-verify` recomputes the sha256 of the release tables against `MANIFEST.json` and, for a build,
reports fetch outcomes, how many of the listed images were recovered, and
per split how many label rows and how many queries still have every candidate. The annotated bytes
were JPEG re-encodes produced by the internal ingest, so they cannot be compared with a fresh fetch;
`fetch_log.parquet` records the md5 of every stored file so that two rebuilds can be compared with
each other.

## Local-only artifacts

`wire_queries_full.parquet` (Natural Questions text), `wire_documents.parquet` (page text) and
`images/` are third-party content assembled for local use and are not part of the release.

## Packaged dataset

`wire-assemble --out BUILD_DIR [--splits train val test] [--include-missing]` joins the labels with
the queries, the restored Natural Questions fields, the image metadata and the page context, embeds the
fetched bytes in an `Image` column and saves a `datasets.DatasetDict` to `BUILD_DIR/wire_dataset`
(`load_from_disk` reads it back; the image is decoded on access). Rows whose image could not be fetched
are dropped unless `--include-missing` keeps them with `image = None`. The 2026-11-06 rebuild packs
52,549 / 4,528 / 8,934 rows (train / val / test) into 10 GB.
