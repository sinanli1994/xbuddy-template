# Resume Retrieval Eval

Measures whether JobBuddy can find the **right evidence** in a resume — the part of
the system that has to work before anything it *says* about that evidence can be
trusted. Retrieval quality is measured here on its own, separately from answer quality.

```bash
uv run python evals/resume_retrieval/run_resume_retrieval_eval.py --check-labels              # free
uv run python evals/resume_retrieval/run_resume_retrieval_eval.py                             # free: BM25
uv run python evals/resume_retrieval/run_resume_retrieval_eval.py --embeddings                # free once cached
uv run python evals/resume_retrieval/run_resume_retrieval_eval.py --embeddings --allow-paid   # first run only
```

`--embeddings` adds dense retrieval (`text-embedding-3-small`, 1536 dims) through a disk
cache in `.embedding_cache/` (git-ignored). Without `--allow-paid`, a cache miss stops the
run and prints what it would cost — it never spends silently. The whole corpus plus every
query is **117 texts, 7,744 tokens, about $0.00015**, and a warm rerun makes no calls.

## The corpus

Three synthetic resumes in `corpus/`, deliberately different in shape:

| Resume | Tokens | Shape it exercises |
|---|---|---|
| `backend_to_ai` | ~840, 2 pages | ALL-CAPS headings, **no blank lines** between entries (how PDFs usually extract), a `TALKS` heading that maps to Publications |
| `data_analyst` | ~410, 1 page | Title Case headings with colons, blank lines between entries |
| `teacher_to_ux` | ~700, 2 pages | Mixed headings (`Portfolio`, `Certifications & Training`), a paragraph-style entry, and one entry **over the hard maximum** so the split path runs |

Technologies and themes overlap on purpose — PostgreSQL appears in three places in one
resume, dashboards in two in another — so ranking has hard negatives to get wrong.

## Labels: content anchors, not chunk ids

Each query lists the **evidence** a good retriever should surface, as short verbatim
phrases from the resume. A retrieved chunk is relevant to an evidence item if its own
text contains the anchor (case- and whitespace-insensitive).

Chunk ids change whenever the chunker does. Anchors don't — so one set of labels scores
every chunking strategy fairly, and a strategy that cuts an anchor in half pays for it
as a miss (reported as `unreach`).

`--check-labels` enforces that every anchor occurs **exactly once** in its resume.

## Metrics

- **Recall@k** — over evidence items, not chunks. Two overlapping windows holding the
  same anchor count as one item found, so duplicating text cannot inflate recall. A
  query with two evidence items can score 0.5 at k=1.
- **MRR** — reciprocal rank of the first chunk holding any evidence item.
- **tok@3** — tokens a caller would put in the prompt by taking the top 3. Recall@k is
  only comparable across strategies alongside this: bigger chunks buy recall with context.

Queries are split into `lexical` and `semantic` **automatically**: a query is lexical
if it shares a stemmed term with any line holding its evidence. Nobody chooses the split,
so it cannot be tuned to flatter one retriever.

## Strategies compared

| Strategy | What it is |
|---|---|
| `section` | Production default. Section-aware; small entries merged toward ~120 tokens |
| `entry` | Section-aware, one entry per chunk — measures what merging costs |
| `window-250` | The production fallback: 250-token windows, 40 overlap |
| `window-80` | Fixed windows at roughly section-aware granularity |

`window-80` exists because a one-page resume is only 2–4 windows of 250 tokens, so
`window-250` reaches high Recall@3 by returning most of the resume. Holding granularity
roughly constant isolates *where boundaries fall* from *how big chunks are*.

`positional` ranks chunks in document order, ignoring the query — the chance floor
every real retriever must beat.

## Stage 1 results (BM25, free)

| strategy | chunks | tok/chunk | R@1 | R@3 | R@5 | MRR | tok@3 |
|---|---|---|---|---|---|---|---|
| section | 9.7 | 70 | 0.42 | 0.79 | 0.83 | 0.63 | 244 |
| entry | 10.7 | 64 | 0.40 | 0.77 | 0.81 | 0.60 | 219 |
| window-250 | 3.7 | 214 | 0.62 | 0.92 | 0.92 | 0.80 | 468 |
| window-80 | 13.0 | 70 | 0.48 | 0.73 | 0.77 | 0.64 | 183 |
| *section, positional floor* | 9.7 | 70 | 0.00 | 0.44 | 0.73 | 0.27 | 218 |

BM25 on `section`, by query kind: lexical (n=19) R@1 0.53 / MRR 0.76; **semantic (n=7)
R@1 0.14 / MRR 0.29**. Keyword search barely finds evidence phrased differently from the
question — the gap Stage 2's embeddings have to close.

## Stage 2 results (embeddings)

32 queries (19 lexical, 13 semantic), 37 evidence items. `text-embedding-3-small`, cosine.

### Overall (n=32)

| strategy | retriever | R@1 | R@3 | R@5 | MRR | tok@3 |
|---|---|---|---|---|---|---|
| section | bm25 | 0.41 | 0.73 | 0.77 | 0.59 | 222 |
| section | embedding | 0.52 | 0.80 | 0.92 | 0.72 | 278 |
| **section −summary** | **embedding** | **0.58** | **0.86** | **0.95** | **0.76** | 306 |
| entry | embedding | 0.52 | 0.80 | 0.88 | 0.71 | 253 |
| window-80 | embedding | 0.58 | 0.81 | 0.91 | 0.75 | 210 |
| window-250 | embedding | 0.38 | 0.88 | 1.00 | 0.67 | 596 |

### Semantic (paraphrased, n=13) and lexical (n=19), embedding

| strategy | sem R@1 | sem R@3 | sem MRR | lex R@1 | lex R@3 | lex MRR |
|---|---|---|---|---|---|---|
| section | 0.46 | 0.69 | 0.62 | 0.55 | 0.87 | 0.78 |
| section −summary | 0.54 | 0.81 | 0.68 | 0.61 | 0.89 | 0.82 |
| window-80 | 0.65 | 0.81 | 0.79 | 0.53 | 0.82 | 0.72 |
| *bm25, section* | *0.23* | *0.46* | *0.35* | *0.53* | *0.92* | *0.76* |

### What the eval can and cannot distinguish

Paired bootstrap over queries (95% interval; `*` = excludes zero):

| comparison | overall MRR diff | semantic | W/L/T |
|---|---|---|---|
| window-80 − section | +0.030 [−0.115, +0.172] | +0.161 [−0.074, +0.397] | 9/8/15 |
| **section −summary − section** | **+0.044 [+0.008, +0.092]\*** | **+0.055 [+0.006, +0.136]\*** | **7/0/25** |
| window-80 − section −summary | −0.014 [−0.163, +0.129] | +0.106 [−0.141, +0.353] | 9/10/13 |
| embedding − bm25 (window-80) | +0.198 [+0.016, +0.381]\* | +0.516 [+0.228, +0.776]\* | 15/7/10 |

- **Embeddings fix paraphrases.** Semantic MRR 0.35 → 0.62–0.79 depending on chunking,
  winning 8–10 of 13; lexical queries are unchanged (BM25 ≈ embeddings there).
- **Section-aware and fixed 80-token windows are statistically tied.** They fail on
  *different* queries: section-aware dilutes one relevant bullet inside a 190–230-token
  job chunk; windows cut a bullet away from its entry title and mix sections. The flagship
  query — "evidence of deploying an AI application to production" — ranks 1st section-aware
  and 5th under window-80.
- **Excluding the Summary chunk is the only significant effect.** It helps 7 queries and
  hurts none. The summary paraphrases the whole resume, so it attracts matches (top-1 for
  4 queries under embeddings, 8 under BM25) while holding none of the specific evidence.
  Excluding the header as well adds nothing.

### Top-K (embedding, section −summary)

| K | R@K | semantic R@K | tokens | share of resume |
|---|---|---|---|---|
| 1 | 0.58 | 0.54 | 101 | 16% |
| **3** | **0.86** | **0.81** | **306** | **48%** |
| 5 | 0.95 | 0.96 | 472 | 73% |

K=5 returns nearly three quarters of a short resume — at that point retrieval is barely
selecting. K=3 recovers 86% of evidence from under half of it.

### Decision

**Section-aware chunking, Summary excluded from retrieval, K=3.** Not because section-aware
won — it tied — but because the one effect this eval could distinguish (excluding the
Summary) needs to know where the Summary is, and it keeps evidence attributable to a
section and an entry. window-80 is the stronger point estimate on paraphrases
(+0.106 MRR, interval spanning zero); the specific failure it fixes — dilution in large
entries — is the next experiment, to be judged on held-out queries rather than the 32
that revealed it.
