# WIRE: a human-annotated resource for multimodal image retrieval

Companion repository of the paper *WIRE: A Large Human-Annotated Resource for Multimodal Image
Retrieval* (under review). It ships the human relevance annotations, the identifiers needed to rebuild
the dataset from public sources, the toolkit that rebuilds it, and the paper's result tables.

## Abstract

While billions of weakly-supervised image-caption pairs exist for vision-language pre-training, no
comparable resource offers human relevance judgments for evaluating image retrieval from
information-seeking queries. We introduce WIRE, a benchmark of 99,089 human-annotated (query, image, web
document) tuples spanning 13,260 image-seeking queries targeting LLM-augmented search scenarios. Each
query is paired with a pool of candidate images, and every candidate includes the image (URL, pixels,
caption) and its parent web document from which arbitrary context can be derived, and carries graded
relevance labels from multiple annotators. Results of our experiments to assess the complexity and
utility of WIRE show that (i) modern vision-language models remain far from the oracle ceiling in the
zero-shot setting; (ii) fusing image, caption, and contextual signals from the parent web document
through fine-tuning consistently improves ranking performance.

## The dataset

| Feature | Value |
|---|---|
| Judged (query, image) pairs | 77,250 released (99,089 annotated) |
| Queries | 13,120 (train 10,165 / val 986 / test 1,969) |
| Images | 73,480, referenced by URL |
| Labels | 17,233 highly relevant, 48,140 somewhat relevant, 11,877 irrelevant |
| Query source | Natural Questions question/answer pairs rewritten into image-seeking queries |
| Page context | public `mlfoundations/MINT-1T-HTML` rows, located down to the parquet page |
| Languages | queries English; pages predominantly English |
| License | CC BY-NC 4.0, except `data/wire_queries.parquet` (CC BY-SA 3.0); see [License](#license) |

**How it was built.** Queries derive from Natural Questions: an LLM (`DeepSeek-R1-Distill-Qwen-32B`,
prompt in `prompts/image_seeking_query.txt`) judged whether a question/answer pair could be enriched by
an image and, if so, rewrote it into a short image-seeking query; unsuitable pairs were discarded. For
each query, the top-20 documents of a BM25 index over the MINT-1T-HTML web corpus were retrieved and
their images (above an icon-size threshold) became the candidate pool, about 7 images per query, each
with its alt-text caption and its parent page. Because candidates come from retrieved documents rather
than random sampling, every negative is topically related to the query, and several candidates often
share a page.

**Annotation.** Crowd workers on Amazon Mechanical Turk saw the query, the image and its caption and
rated each pair *Highly Relevant*, *Somewhat Relevant* or *Irrelevant* (a fourth option flagged images
that failed to load). Each pair was judged by at least two workers, with a third recruited on
disagreement; control questions with known answers filtered inattentive workers. The released label is
the majority vote over the retained judgments, ties resolved to *Somewhat Relevant*. Pairs whose
judgments split between *Irrelevant* and a relevant grade with no tie-breaker are flagged
`safe_for_eval = false`: they stay in the training split and are excluded from validation and test.

**Splits** are by query. The paper describes the full annotated set; the release leaves out the hard
disagreements of validation and test queries, the queries whose Natural Questions pair could not be
verified, and every pair whose image or page is hosted on a site that disallows automated access by
Amazon's crawlers or that is a licensed news or stock-photo source.

`docs/SCHEMA.md` documents every column; `docs/KNOWN_GAPS.md` lists the known limits;
`docs/RESULTS.md` reproduces the paper's result tables and baselines.

## What is shipped, and what is not

**Shipped**

- `data/wire_queries.parquet`, `data/wire_images.parquet`, `data/wire_labels.parquet`: the queries,
  the image and page URLs, and the labels with the train / val / test split.
- `data/wire_mint.parquet`, `data/wire_mint_pages.parquet`: where each image's page sits in the public
  MINT-1T-HTML corpus, down to the parquet data pages that hold it, so a rebuild reads 72 GB of pages
  instead of the 3 TB corpus.
- `data/MANIFEST.json` with the sha256 of every table, the query-generation prompt, and the toolkit that
  rebuilds the full dataset.

**Not shipped** (third-party content, rebuilt locally by the toolkit)

- Image bytes: fetched from `image_url`.
- The Natural Questions question, answers and context: joined from the public dataset
  `lucadiliello/naturalquestionsshortqa` through `nq_key`.
- Captions and page text: read from the public `mlfoundations/MINT-1T-HTML` rows the release points at
  (or re-crawled from `page_url`).
- Per-annotator judgments: only the consolidated label is released.

The tables have different schemas: load them one at a time (for example
`load_dataset("parquet", data_files="data/wire_labels.parquet")`) or read the parquet files directly.

## Rebuilding the full dataset

```bash
pip install -e '.[build]'                      # aiohttp, pillow, trafilatura, lxml, datasets
export HF_TOKEN=...                            # a huggingface.co token: the MINT step makes many small requests
wire-build --release data --out BUILD_DIR      # tables, Natural Questions join, images, paragraphs from MINT-1T-HTML
wire-verify --data data --build BUILD_DIR      # sha256 of the tables, fetch outcomes per split
wire-assemble --out BUILD_DIR                  # datasets.DatasetDict with an Image column
```

`wire-build` writes `BUILD_DIR/images/{image_id}.jpg`, a `fetch_log.parquet` with one row per image
(outcome, HTTP status, content type, size, md5, decoded format and dimensions, error, attempts),
`fetch_report.json`, `wire_queries_full.parquet` with the restored Natural Questions fields,
`wire_documents.parquet` with the caption and the paragraphs around each image, and `BUILD_MANIFEST.json`.

The downloader (`wire-fetch`) identifies itself, makes one request at a time per host with a pause
between them, honours `X-Robots-Tag` directives, verifies that what it stored decodes as an image, and
then runs three recovery passes over the failures: refusals with a browser-like research user agent and
the page as referrer, rate limits and server errors with long patient back-off, and dead hosts or 404s
through the Wayback Machine. `wire-fetch-report --out BUILD_DIR` breaks the remaining failures down into
gone, blocked, rate limited, server or network errors, and unusable content.

The MINT step reads, for every image, the MINT-1T-HTML document it was annotated in: not the corpus,
only the parquet data pages that hold that row (`wire_mint_pages.parquet` says where they are), about
1 MB per image over HTTP range requests. It yields the caption (the alt text MINT stored for the image)
and the paragraphs around the image. `wire-documents --out BUILD_DIR` is the alternative that re-crawls
the live `page_url` instead, and the only source of page titles.

`wire-assemble` packs everything into `BUILD_DIR/wire_dataset`, a `datasets.DatasetDict`
(train / val / test) with the fetched bytes in an `Image` column, readable with `load_from_disk`.
`docs/BUILD.md` describes every step and option.

`wire_queries_full.parquet`, `wire_documents.parquet` and `wire_dataset/` contain third-party content and
Natural Questions text; they are for local use and must not be redistributed.

### What a rebuild recovers

Web content decays, so a rebuild scores a smaller pool than the one the paper was computed on. Rebuild
on 2026-11-06, about 16 months after the images were annotated, with the defaults (64 concurrent
requests, at most 5 per second per host, 2 retries, the three recovery passes, 8 MINT requests in
flight):

| Rebuild outcome | Count |
|---|---|
| images fetched | 62,809 / 73,480 (85.5%) |
| of which recovered by the research-agent pass | 159 |
| of which recovered by the patient pass | 61 |
| of which recovered from the Wayback Machine | 1,660 |
| images with caption and paragraphs from MINT-1T-HTML | 73,478 / 73,480 (72 GB read in 1 h 31 min) |
| test rows with an image | 8,934 / 10,497 |
| test queries with every candidate | 1,090 / 1,969 |

| Why the rest failed | Images | Share |
|---|---|---|
| blocked: 403, 401, 402, or an `X-Robots-Tag` directive | 6,313 | 8.6% |
| gone: 404, 410, or the host no longer resolves, and no snapshot | 2,277 | 3.1% |
| server or network: 5xx, timeouts, TLS or connection errors | 1,310 | 1.8% |
| not usable: not an image, undecodable, or above the size cap | 635 | 0.9% |
| rate limited: 429 after every retry | 136 | 0.2% |

Keep the downloader's concurrency low: DNS saturation looks like link rot (see `docs/BUILD.md`). The
paper's baselines were computed with every image available (`docs/RESULTS.md`); re-run a couple of them
on your own rebuild before comparing.

## Repository layout

```
data/        release tables, MANIFEST.json, the CC BY-SA 3.0 license of the queries table
docs/        SCHEMA.md (data contract), BUILD.md (rebuild steps), KNOWN_GAPS.md, RESULTS.md (paper tables)
prompts/     the query-generation prompt
src/wire/    the toolkit: build, fetch, mint, documents, verify, assemble
tests/       unit tests, including the release-table invariants
```

## License

The relevance labels, splits, MINT index, code and documentation are released under the Creative Commons
Attribution-NonCommercial 4.0 International license (`LICENSE`). Please cite the paper when you use them.
`data/wire_queries.parquet` derives from Google's Natural Questions (CC BY-SA 3.0) and is released under
the Creative Commons Attribution-ShareAlike 3.0 Unported license (`data/LICENSE-wire_queries.txt`), with
attribution to Natural Questions (Kwiatkowski et al., TACL 2019). Images, captions and page text are
third-party content that the toolkit fetches from their original sources and from the public
MINT-1T-HTML corpus (Awadalla et al., NeurIPS 2024); they are not redistributed here and keep their own
terms.

<!-- After the paper is published, add here:
This code is being released solely for academic and scientific reproducibility purposes, in support of the
methods and findings described in the associated publication. Pull requests are not being accepted in order
to maintain the code exactly as it was used in the paper.
-->

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for how to report a security issue.

## Citation

The paper is under review; a citation entry will be added on acceptance.
