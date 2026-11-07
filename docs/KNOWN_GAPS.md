# Known gaps

Facts established while preparing the release that limit what a rebuild can reproduce, and counts that
differ from the paper.

## Counts

| Statement in the paper | Note |
|---|---|
| 99,089 tuples / 13,260 queries (abstract) vs 96,566 / 13,255 (splits table) | Both are right for different populations. 99,089 rows minus one duplicate (query, image) pair gives 99,088 labelled pairs; 2,523 hard-disagreement rows of val/test queries are excluded from the splits, which removes 5 queries entirely. The public release is smaller still (see the README). |
| Label-resolution counts 59,912 + 28,823 + 9,284 | They sum to 98,019; candidates with four retained judgments (655) and with none (415) complete the total. |
| "images at least 200x200 px" | The retrieval filter was on the smaller side being above 60 px; 4.1% of candidates have a smaller side between 101 and 199 px. |
| "994 different-document queries" and "36% same-document" | With the primary page of each candidate: 520 of 1,566 answerable test queries (33.2%) have positives and negatives on the same page; 1,046 have all candidates on distinct pages. |

## Reproducibility limits

- **Published zero-shot numbers were computed over all splits together** (13,255 queries, 91,738-image
  pool), with every image available, and included the 2,523 hard-disagreement rows of val/test queries.
  A rebuild scores a smaller pool with fewer distractors and fewer positives.
- **Fetched images cannot be byte-verified against the annotated ones.** Every image was re-encoded to
  JPEG before annotation, so the release carries no per-image checksum; a rebuild is checked by URL
  identity (`image_id = sha256(image_url)[:32]`) and by its own fetch log.
- **Link rot.** On 2026-11-06, 62,809 of the 73,480 image URLs (85.5%) could be fetched, 1,660 of them
  only from the Wayback Machine. Of the 10,671 failures, 6,313 are refusals (403/401/402 or an
  `X-Robots-Tag` directive), 2,277 are gone with no archived snapshot (404/410 or a host that no longer
  resolves), 1,310 are server or network errors, 635 serve something that is not a decodable image, and
  136 stayed rate limited. `wire-fetch-report` breaks a rebuild's own failures down the same way.
- **Page context comes from MINT-1T-HTML, not from a shipped snapshot.** The pages the images were
  annotated in are rows of the public `mlfoundations/MINT-1T-HTML` (`data_v1_0`); the release points at
  them (`wire_mint.parquet`) and at the parquet pages that hold them (`wire_mint_pages.parquet`), and
  `wire-mint` reads those pages (72 GB in total for the 73,478 images that have a page; 2 images have
  none). Every pointer was verified against the annotated record (identical image list and text layout)
  when the release was built. The corpus files are pinned to one hub revision; if they are ever
  rewritten, `wire-mint` reports `mint_layout_mismatch` rather than returning the wrong paragraph.
- **Page titles are not in MINT-1T-HTML.** `page_title` is null for MINT-sourced rows; the `T` input of
  the paper's reranking experiments needs the live-page crawl (`wire-documents`), which recovers only
  the pages still online.
- **Results vary with the rebuild.** The pool is whatever the crawl recovered, and captions and
  paragraphs are read from the public corpus at build time, so numbers computed on a rebuild shift from
  the published ones (`docs/RESULTS.md`). Re-run a couple of the baselines on your own rebuild before
  comparing.

## Provenance notes

- 4 queries coincide verbatim with a Natural Questions question (television-episode and cast titles),
  and 228 coincide with an NQ short answer (mostly person names); both are consequences of the
  generation prompt, which asks for the key visual concept of the pair.
- 284 queries match more than one Natural Questions example because the dataset contains repeated
  questions with distinct keys; the lexicographically smallest key is released.
