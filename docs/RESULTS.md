# Baselines reported in the paper

Numbers from the paper *WIRE: A Large Human-Annotated Resource for Multimodal Image Retrieval*. They were
computed on the annotated set the paper describes (13,255 queries and 91,738 distinct images across the
three splits, 1,566 answerable test queries), with every image available. The public release is smaller
(see the README) and web content decays, so a rebuild scores a different pool; re-run a couple of these
baselines on your own rebuild before comparing.

## Protocol

Two kinds of evaluation. **Zero-shot retrieval** treats the three splits as one test set with no
fine-tuning: for each query, a model ranks every distinct image of the splits by cosine similarity to
the query, and R@10 is the primary metric. Only the images of a query's own BM25 pool were judged, so a
relevant image retrieved from outside the pool counts as a miss: zero-shot recall is a lower bound.
**Fine-tuned reranking** trains on the train split and evaluates on the answerable test queries (those
with at least one Highly Relevant candidate); each model reranks only the query's judged candidate pool
(about 6.6 images per query), and P@1 is the primary metric. Relevance is binary: only *Highly Relevant*
counts as positive. The Oracle ranks all positives above all negatives.

Modes: **I** scores the image alone (query-image cosine similarity); **C+I** adds the alt-text caption,
by mean fusion of the image and caption embeddings for dual encoders (and Jina) or by joint encoding for
the other VLM embedders; **C+I+P** adds the paragraph adjacent to the image on its page; **T** is the
page title.

## Zero-shot retrieval

All numbers are percentages. Tier A: dual-encoder retrievers; Tier B: joint VLM-as-embedder models.

| Model | I R@10 | I R@50 | C+I R@10 | C+I R@50 |
|---|---|---|---|---|
| *Tier A* | | | | |
| CLIP ViT-L/14@336 (427M) | 38.7 | 55.1 | 42.5 | 56.2 |
| OpenCLIP ViT-bigG/14 (2.5B) | 41.1 | 57.0 | 47.2 | 60.6 |
| MetaCLIP-2 ViT-H/14 (1.8B) | 41.6 | 58.8 | 42.6 | 55.7 |
| EVA02-CLIP L/14@336 (428M) | 38.5 | 54.6 | 45.5 | 59.2 |
| EVA02-CLIP E+/14 (5.1B) | 40.4 | 56.2 | 49.3 | 63.7 |
| SigLIP-2 ViT-g/16@384 (1.9B) | 41.3 | 57.7 | 42.1 | 55.0 |
| DFN5B CLIP ViT-H/14@378 (987M) | 33.2 | 49.2 | 40.2 | 53.5 |
| *Tier B* | | | | |
| Qwen3-VL-Embedding-8B (8.1B) | 32.0 | 47.6 | 54.0 | 68.9 |
| GME-Qwen2-VL 7B (7.7B) | 31.7 | 48.5 | 53.7 | 69.4 |
| Jina-Embeddings v5-omni-small (1.7B) | 28.3 | 42.6 | 52.4 | 66.1 |
| Oracle | 100.0 | 100.0 | 100.0 | 100.0 |

The best models reach about 54 R@10 (Qwen3-VL-Embedding-8B 54.0, GME-Qwen2-VL 7B 53.7), far below the
oracle: WIRE is far from saturated. Captions matter: adding them improves R@10 for nearly every model,
marginally for some dual encoders (SigLIP-2 +0.8, MetaCLIP-2 +1.0) and massively for the strongest
joint models (Qwen3-VL 32.0 to 54.0). In image-only mode the Tier A dual encoders (38-42 R@10) beat the
Tier B joint embedders (28-32); captions reverse the order, lifting Tier B to 52-54, past the best
Tier A result (EVA02-CLIP E+ 49.3): the image embeddings of the Tier B models are weaker than those of
dedicated contrastive dual encoders, but they combine caption and image far more effectively than mean
fusion. Within Tier A, scale yields diminishing returns and data curation matters (DFN5B underperforms
comparably sized OpenCLIP variants). Jina-v5 is reported with mean fusion, which beats its native joint
encoding by 22 R@10.

## Fine-tuned reranking

Answerable WIRE test queries (1,566). The bottom block restricts the evaluation to the 994 queries whose
candidates come from different parent documents.

| Setting | P@1 | MAP | MRR | NDCG | R@5 |
|---|---|---|---|---|---|
| *EVA02-CLIP L/14@336 (428M), fine-tuned on WIRE* | | | | | |
| I | 42.9 | 58.2 | 63.2 | 70.8 | 81.7 |
| C+I | 46.0 | 60.2 | 65.3 | 72.4 | 82.8 |
| C+I+P (weights 0.4 / 0.4 / 0.2) | **47.1** | **60.3** | **65.6** | **72.5** | 82.3 |
| *MiniLM-L-12-v2 (33M) cross-encoder, fine-tuned on WIRE* | | | | | |
| C | 43.0 | 57.3 | 62.7 | 70.3 | 78.5 |
| C+T+P | 43.5 | 57.4 | 63.0 | 70.4 | 78.4 |
| *MiniLM, 994 different-document queries only* | | | | | |
| C | 40.4 | 57.0 | 61.1 | 70.4 | 80.1 |
| C+T | **43.6** | **58.1** | **62.9** | **71.0** | 80.2 |
| Oracle | 100.0 | 100.0 | 100.0 | 100.0 | 99.8 |

EVA02-CLIP L/14@336 (a dual encoder mapping text and images into a shared 768-dimensional space) was
trained for 2 epochs with a symmetric InfoNCE loss (batch size 128, in-batch negatives); under C+I and
C+I+P the similarity of the query with each component is averaged with the weights given. MiniLM-L-12-v2
was used as a text-only cross-encoder, trained with binary cross-entropy for 2 epochs (batch size 128) on
the caption alone (C) or the caption with the page title and adjacent paragraph (C+T+P).

Combining caption and image (C+I) improves over image-only by 3.1 P@1: the two signals are
complementary, the image encoder capturing semantics that noisy captions miss and captions supplying
lexical cues absent from images. The adjacent paragraph adds a further 1.1 P@1. For the cross-encoder,
adding title and paragraph to the caption leaves P@1 essentially flat (43.0 to 43.5), for a structural
reason: 36% of answerable queries have positives and negatives drawn from the same document, so title and
paragraph are near-identical between sibling candidates and cannot discriminate. Restricted to the 994
queries whose candidates come from different documents, adding the title gives +3.2 P@1. Document
context carries relevance information, but exploiting it needs models that attend to the context of
the specific image rather than to a page-level text blob.
