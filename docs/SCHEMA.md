# WIRE data contract

Two directory layouts matter: the **release** (`data/`, redistributed) and the **rebuilt dataset**
(`BUILD_DIR`, produced locally by `wire-build`, never redistributed).

## Release tables (`data/`)

Schemas are pinned in `src/wire/schema.py`; `MANIFEST.json` carries sha256 and row counts.

### `wire_queries.parquet` — 13,120 rows, one per query

| column | type | meaning |
|---|---|---|
| `query_id` | string | `sha1(query)[:16]` |
| `query` | string | the image-seeking query (an LLM rewrite of a Natural Questions q/a pair) |
| `nq_key` | string | `key` of the source example in `lucadiliello/naturalquestionsshortqa`; every key is verified, i.e. the query equals the saved generator output for that key |
| `split` | string | `train`, `val` or `test` |

### `wire_images.parquet` — 73,480 rows, one per image

| column | type | meaning |
|---|---|---|
| `image_id` | string | `sha256(image_url)[:32]`; the file name under `images/` |
| `image_url` | string | original web URL of the image, the only locator shipped |
| `page_url` | string, nullable | URL of the web page the image was found on (null for 2 images) |

### `wire_labels.parquet` — 77,250 rows, one per (query, image)

| column | type | meaning |
|---|---|---|
| `query_id` | string | FK to `wire_queries` |
| `image_id` | string | FK to `wire_images` |
| `label` | string | consolidated graded label: `highly relevant`, `somewhat relevant`, `irrelevant` |
| `safe_for_eval` | bool | false when the judgments were a hard disagreement; such rows never enter val/test |
| `split` | string | `train`, `val` or `test`; a query's rows all share its split |

### `wire_mint.parquet` — 106,049 rows, one per (image, MINT document)

Where the image's page sits in the public corpus `mlfoundations/MINT-1T-HTML`. An image may have been
seen in several documents (crawls of the same page, or pages that share the image); `mint_rank` 0 is the
one the annotation pipeline listed first and the one `wire-mint` reads.

| column | type | meaning |
|---|---|---|
| `image_id` | string | FK to `wire_images` |
| `mint_rank` | int32 | 0 for the primary document, then 1, 2, ... |
| `mint_file` | string | path of the parquet file in the hub repo, e.g. `data_v1_0/shard_012_part_003.parquet` |
| `mint_row` | int64 | row of the document in that file |
| `mint_image_idx` | int32 | slot of the image in the document's interleaved `texts` / `images` lists |
| `mint_num_slots` | int32 | length of those lists; `wire-mint` checks it before trusting the slot |
| `mint_doc_idx` | int64 | row of the document in the internal copy the pipeline used (provenance only) |
| `page_url` | string | `url` of the document |

### `wire_mint_pages.parquet` — 122,204 rows, one per needed parquet data page

The MINT-1T-HTML files have a single row group and no page index, so this table is what lets a rebuild
fetch a row without the file. Only the `texts` and `metadata` pages that hold the primary (`mint_rank` 0)
rows of `wire_mint` are listed: 72 GB in total (51 GB of paragraphs, 21 GB of metadata), about 1 MB per
document.

| column | type | meaning |
|---|---|---|
| `mint_file` | string | as above |
| `column` | string | `texts` (paragraphs) or `metadata` (per-image alt text, the caption) |
| `codec` | string | compression of the column chunk (`SNAPPY`) |
| `offset` | int64 | byte offset of the page header in the file |
| `size` | int64 | bytes of header plus body |
| `first_row` | int64 | index of the first record that starts in the page |
| `num_rows` | int64 | records that start in the page |
| `continues` | bool | the page opens inside a record begun on the previous page |

Grades used by the metrics: `highly relevant` = 2, `somewhat relevant` = 1, `irrelevant` = 0.
Binary relevance: positive iff `highly relevant`. A query is *answerable* iff it has at least one
`highly relevant` candidate.

## Rebuilt dataset (`BUILD_DIR`)

```
BUILD_DIR/
  wire_queries.parquet  wire_images.parquet  wire_labels.parquet   # copies of the release tables
  images/{image_id}.jpg          # fetched bytes as served (any raster format; decode with PIL)
  fetch_log.parquet              # one row per image: outcome, recovered_by, archive_outcome, http_status, content_type, bytes, md5, format, width, height, error, attempts
  fetch_report.json              # counts by outcome, class, HTTP status, scheme and host
  wire_queries_full.parquet      # query_id, query, nq_key, question, answers (list<string>), context  — local only
  wire_documents.parquet         # one row per image: image_id, page_url, page_status, image_found, caption, page_title, prev_text, next_text, doc_text
  wire_dataset/                  # optional, from wire-assemble: datasets.DatasetDict with an Image column
  BUILD_MANIFEST.json
```

`caption` is the image's alt text: from the MINT-1T-HTML document's `metadata` when written by
`wire-mint`, from the `<img alt>` attribute when written by `wire-documents`; empty when the page has
none. `doc_text` is `prev_text + "\n" + next_text` (empty parts dropped): the nearest non-empty paragraph
before and after the image in its document. Written by `wire-mint` (`page_status` = `mint`, from the
MINT-1T-HTML row; `page_title` is null) or by `wire-documents` (from the live page; `page_title` is
the page `<title>`, `image_found` tells how the image was located in the markup).
