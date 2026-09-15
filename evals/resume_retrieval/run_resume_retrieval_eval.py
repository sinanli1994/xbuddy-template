"""Resume retrieval eval: chunking strategies x retrievers, over labelled evidence.

    uv run python evals/resume_retrieval/run_resume_retrieval_eval.py --check-labels         # free
    uv run python evals/resume_retrieval/run_resume_retrieval_eval.py                        # free: BM25 only
    uv run python evals/resume_retrieval/run_resume_retrieval_eval.py --embeddings           # free if cached
    uv run python evals/resume_retrieval/run_resume_retrieval_eval.py --embeddings --allow-paid
    uv run python evals/resume_retrieval/run_resume_retrieval_eval.py --embeddings --detail section

Retrieval quality is measured on its own, separately from anything JobBuddy says
with the evidence it retrieves.

`--embeddings` adds dense retrieval (text-embedding-3-small, 1536 dims) through a
disk cache. Without `--allow-paid` a cache miss stops the run and prints what it
would cost; it never spends silently. Only embedding calls are made — no chat model.

Chunking strategies:
  section     production default: section-aware, small entries merged to ~120 tokens
  entry       section-aware, one entry per chunk (what merging costs or buys)
  window-80   fixed windows at roughly section-aware granularity
  window-250  the production fallback: 250-token windows, 40 overlap

A one-page resume is only 2-4 windows of 250 tokens, so window-250 reaches high
recall by returning most of the resume; the tok@k and "% resume" columns show it.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rr_dataset import (
    QUERIES,
    RESUMES,
    LabeledQuery,
    contains_anchor,
    evidence_line,
    load_resume,
    normalize_for_match,
)
from rr_metrics import (
    QueryResult,
    mean,
    recall_at_k,
    reciprocal_rank,
    relevant_chunks,
    tokens_at_k,
)
from rr_retrievers import BM25Retriever, EmbeddingRetriever, PositionalRetriever, Retriever, terms
from rr_stats import paired_bootstrap

from agents.xbuddy.resume.chunking import chunk_fixed_window, chunk_text
from agents.xbuddy.resume.models import ChunkedResume, ResumeSection
from agents.xbuddy.resume.tokens import count_tokens

STRATEGIES = {
    "section": lambda text: chunk_text(text),
    "entry": lambda text: chunk_text(text, merge_below=0),
    "window-80": lambda text: chunk_fixed_window(text, window_tokens=80, overlap_tokens=15),
    "window-250": lambda text: chunk_fixed_window(text),
}

# Eval-only ablations: chunks in these sections are removed from the candidate set
# before ranking. The chunker is untouched; only what retrieval may return changes.
ABLATIONS: dict[str, frozenset[ResumeSection]] = {
    "section": frozenset(),
    "section -summary": frozenset({ResumeSection.SUMMARY}),
    "section -summary -header": frozenset({ResumeSection.SUMMARY, ResumeSection.HEADER}),
}

# text-embedding-3-small list price, USD per million input tokens.
EMBEDDING_USD_PER_MILLION = 0.02


def query_kind(query: LabeledQuery) -> str:
    """`lexical` if the query shares a stemmed term with any evidence line."""
    query_terms = set(terms(query.query))
    for anchor in query.evidence:
        if query_terms & set(terms(evidence_line(query.resume, anchor))):
            return "lexical"
    return "semantic"


def chunk_all(strategy: str) -> dict[str, ChunkedResume]:
    return {name: STRATEGIES[strategy](load_resume(name)) for name in RESUMES}


def evaluate(
    strategy: str, retriever: Retriever, exclude: frozenset[ResumeSection] = frozenset()
) -> list[QueryResult]:
    chunked = chunk_all(strategy)
    results = []
    for query in QUERIES:
        chunks = chunked[query.resume].chunks
        candidates = [i for i, c in enumerate(chunks) if c.section not in exclude]
        local = retriever.rank(query.query, [chunks[i].content for i in candidates])
        ranking = [candidates[j] for j in local]
        results.append(
            QueryResult(
                query=query,
                ranking=ranking,
                # Relevance is judged on the chunk's own text, over *all* chunks: an
                # anchor that only an excluded chunk holds is a real miss, not a skip.
                # A continuation's prefix repeats its entry's title and must not count.
                relevant=relevant_chunks(query, [c.text for c in chunks]),
                retrieved_tokens=tuple(chunks[i].token_count for i in ranking),
            )
        )
    return results


def summarize(results: list[QueryResult]) -> dict[str, float]:
    resume_tokens = {name: count_tokens(load_resume(name)) for name in RESUMES}
    summary: dict[str, float] = {
        "MRR": mean([reciprocal_rank(r) for r in results]),
        "unreachable": sum(r.unreachable for r in results),
        "n": len(results),
    }
    for k in (1, 3, 5):
        summary[f"R@{k}"] = mean([recall_at_k(r, k) for r in results])
        summary[f"tok@{k}"] = mean([tokens_at_k(r, k) for r in results])
        summary[f"frac@{k}"] = mean([tokens_at_k(r, k) / resume_tokens[r.query.resume] for r in results])
    return summary


def by_kind(results: list[QueryResult], kind: str | None) -> list[QueryResult]:
    return results if kind is None else [r for r in results if query_kind(r.query) == kind]


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def check_labels(verbose: bool = True) -> int:
    """Every anchor must occur exactly once in its resume; ids must be unique."""
    problems: list[str] = []
    ids = [q.id for q in QUERIES]
    if len(ids) != len(set(ids)):
        problems.append("duplicate query ids")
    for query in QUERIES:
        if query.resume not in RESUMES:
            problems.append(f"{query.id}: unknown resume {query.resume!r}")
            continue
        if not query.evidence:
            problems.append(f"{query.id}: no evidence")
        text = normalize_for_match(load_resume(query.resume))
        for anchor in query.evidence:
            count = text.count(normalize_for_match(anchor))
            if count != 1:
                problems.append(f"{query.id}: anchor occurs {count}x: {anchor!r}")

    if verbose:
        kinds = [query_kind(q) for q in QUERIES]
        print(f"  queries: {len(QUERIES)} across {len(RESUMES)} resumes "
              f"({', '.join(f'{r}={sum(q.resume == r for q in QUERIES)}' for r in RESUMES)})")
        print(f"  evidence items: {sum(len(q.evidence) for q in QUERIES)}")
        print(f"  kinds: lexical={kinds.count('lexical')} semantic={kinds.count('semantic')}")
        print("  anchors reachable (contained whole in >=1 chunk) per strategy:")
        for strategy in STRATEGIES:
            chunked = chunk_all(strategy)
            total = reachable = 0
            for query in QUERIES:
                for anchor in query.evidence:
                    total += 1
                    reachable += any(contains_anchor(c.text, anchor) for c in chunked[query.resume].chunks)
            print(f"    {strategy:11} {reachable}/{total}")

    for problem in problems:
        print(f"  [FAIL] {problem}")
    if verbose:
        print(f"\n  label check: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


@dataclass
class EmbeddingRun:
    retriever: EmbeddingRetriever
    store: object  # JsonEmbeddingStore
    unique_chunks: int
    unique_queries: int
    paid_texts: int
    paid_tokens: int


def prepare_embeddings(allow_paid: bool) -> EmbeddingRun | None:
    """Warm the cache for every text the eval will embed, in as few calls as possible.

    Returns None (after explaining) when texts are missing and paying is not allowed.
    """
    from rr_embedding_cache import JsonEmbeddingStore

    from agents.xbuddy.resume.embeddings import CachedEmbedder, EmbeddingCacheMiss, embedding_key
    from core.llm import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

    store = JsonEmbeddingStore(EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)

    def embed_batch(texts: list[str]) -> list[list[float]]:
        from core.llm import get_embeddings

        return get_embeddings().embed_documents(texts)

    embedder = CachedEmbedder(
        embed_batch,
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        cache=store,
        allow_calls=allow_paid,
    )

    chunk_texts = sorted({c.content for s in STRATEGIES for r in chunk_all(s).values() for c in r.chunks})
    query_texts = sorted({q.query for q in QUERIES})

    def key(text: str) -> str:
        return embedding_key(text, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS)

    try:
        embedder.embed(chunk_texts + query_texts)  # one batch for everything missing
    except EmbeddingCacheMiss as miss:
        cost = miss.tokens / 1_000_000 * EMBEDDING_USD_PER_MILLION
        print(f"\n  {miss.missing} text(s) are not cached ({miss.tokens} tokens, about ${cost:.5f}).")
        print("  Rerun with --allow-paid to embed them. Nothing was sent.")
        return None
    finally:
        if embedder.stats.embedded:
            store.save()

    embedder.allow_calls = False  # everything is warm; any later miss is a bug
    return EmbeddingRun(
        retriever=EmbeddingRetriever(embedder),
        store=store,
        unique_chunks=len({key(t) for t in chunk_texts}),
        unique_queries=len({key(t) for t in query_texts}),
        paid_texts=embedder.stats.embedded,
        paid_tokens=embedder.stats.embedded_tokens,
    )


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def _row(label: str, name: str, s: dict[str, float]) -> str:
    return (f"  {label:24} {name:10} {s['R@1']:5.2f} {s['R@3']:5.2f} {s['R@5']:5.2f} "
            f"{s['MRR']:5.2f} {s['tok@3']:6.0f}")


def _header(title: str) -> None:
    print(f"\n  {title}")
    print(f"  {'strategy':24} {'retriever':10} {'R@1':>5} {'R@3':>5} {'R@5':>5} {'MRR':>5} {'tok@3':>6}")
    print("  " + "-" * 64)


def print_strategy_tables(retrievers: list[Retriever]) -> None:
    sizes = {}
    for strategy in STRATEGIES:
        chunks = [c for r in chunk_all(strategy).values() for c in r.chunks]
        sizes[strategy] = (len(chunks) / len(RESUMES), sum(c.token_count for c in chunks) / len(chunks))
    print("\n  chunk shape: " + "  ".join(
        f"{s}={n:.1f}/resume x {t:.0f} tok" for s, (n, t) in sizes.items()))

    results = {(s, r.name): evaluate(s, r) for s in STRATEGIES for r in retrievers}
    for title, kind in (("OVERALL", None), ("LEXICAL queries", "lexical"), ("SEMANTIC (paraphrased) queries", "semantic")):
        n = len(by_kind(next(iter(results.values())), kind))
        _header(f"{title} (n={n})")
        for strategy in STRATEGIES:
            for retriever in retrievers:
                print(_row(strategy, retriever.name, summarize(by_kind(results[(strategy, retriever.name)], kind))))


def _summary_pressure(results: list[QueryResult], strategy: str) -> tuple[int, int]:
    """(# queries whose top chunk is Summary, # where a Summary chunk outranks the first hit)."""
    chunked = chunk_all(strategy)
    top1 = ahead = 0
    for r in results:
        sections = [c.section for c in chunked[r.query.resume].chunks]
        wanted = frozenset().union(*r.relevant)
        if r.ranking and sections[r.ranking[0]] is ResumeSection.SUMMARY:
            top1 += 1
        for index in r.ranking:
            if index in wanted:
                break
            if sections[index] is ResumeSection.SUMMARY:
                ahead += 1
                break
    return top1, ahead


def print_summary_ablation(retrievers: list[Retriever]) -> None:
    print("\n  SUMMARY ABLATION (section-aware chunks; eval-only exclusion, chunker unchanged)")
    print(f"  {'variant':24} {'retriever':10} {'R@1':>5} {'R@3':>5} {'MRR':>5} "
          f"{'semR@1':>6} {'semMRR':>6} {'sum@1':>5} {'sum<hit':>7}")
    print("  " + "-" * 82)
    for retriever in retrievers:
        for label, exclude in ABLATIONS.items():
            results = evaluate("section", retriever, exclude)
            s = summarize(results)
            sem = summarize(by_kind(results, "semantic"))
            top1, ahead = _summary_pressure(results, "section")
            print(f"  {label:24} {retriever.name:10} {s['R@1']:5.2f} {s['R@3']:5.2f} {s['MRR']:5.2f} "
                  f"{sem['R@1']:6.2f} {sem['MRR']:6.2f} {top1:5d} {ahead:7d}")


def print_top_k(retriever: Retriever) -> None:
    print(f"\n  TOP-K ({retriever.name}): recall, context cost, and share of the resume returned")
    print(f"  {'strategy':24} {'K':>2} {'R@K':>5} {'semR@K':>6} {'tok@K':>6} {'% resume':>8}")
    print("  " + "-" * 56)
    variants = [(s, frozenset()) for s in STRATEGIES] + [("section", ABLATIONS["section -summary"])]
    for strategy, exclude in variants:
        results = evaluate(strategy, retriever, exclude)
        s = summarize(results)
        sem = summarize(by_kind(results, "semantic"))
        label = strategy + (" -summary" if exclude else "")
        for k in (1, 3, 5):
            print(f"  {label:24} {k:2d} {s[f'R@{k}']:5.2f} {sem[f'R@{k}']:6.2f} "
                  f"{s[f'tok@{k}']:6.0f} {100 * s[f'frac@{k}']:7.0f}%")


def print_paired(embedding: Retriever) -> None:
    """Which differences the eval can actually distinguish, query by query."""
    configs = {
        "section": ("section", frozenset(), embedding),
        "section -summary": ("section", ABLATIONS["section -summary"], embedding),
        "window-80": ("window-80", frozenset(), embedding),
        "bm25 section -summary": ("section", ABLATIONS["section -summary"], BM25Retriever()),
        "bm25 window-80": ("window-80", frozenset(), BM25Retriever()),
    }
    rr = {
        name: {r.query.id: reciprocal_rank(r) for r in evaluate(strategy, retriever, exclude)}
        for name, (strategy, exclude, retriever) in configs.items()
    }
    kinds = {q.id: query_kind(q) for q in QUERIES}
    pairs = [
        ("window-80", "section"),
        ("section -summary", "section"),
        ("window-80", "section -summary"),
        ("section -summary", "bm25 section -summary"),
        ("window-80", "bm25 window-80"),
    ]
    print("\n  PAIRED MRR DIFFERENCE (a - b), 95% bootstrap interval over queries; * = interval excludes 0")
    print(f"  {'a':>17} vs {'b':<22} {'subset':9} {'diff':>6}  {'95% interval':>17}  W/L/T")
    print("  " + "-" * 88)
    for a, b in pairs:
        for subset in (None, "semantic", "lexical"):
            ids = [q for q in rr[a] if subset is None or kinds[q] == subset]
            res = paired_bootstrap([rr[a][q] for q in ids], [rr[b][q] for q in ids])
            mark = "*" if res.significant else " "
            print(f"  {a:>17} vs {b:<22} {subset or 'overall':9} {res.mean:+6.3f}  "
                  f"[{res.low:+.3f}, {res.high:+.3f}]{mark} {res.wins}/{res.losses}/{res.ties}")


def print_detail(strategy: str, retriever: Retriever) -> None:
    chunked = chunk_all(strategy)
    print(f"\n  per-query detail: {retriever.name} x {strategy}")
    for result in evaluate(strategy, retriever):
        q = result.query
        chunks = chunked[q.resume].chunks
        first_hit = next(
            (rank for rank, i in enumerate(result.ranking, 1) if any(i in rel for rel in result.relevant)),
            None,
        )
        top = result.ranking[0] if result.ranking else None
        top_label = f"#{top} {chunks[top].content.splitlines()[0][:50]}" if top is not None else "(nothing)"
        print(f"    {q.id:4} {query_kind(q)[:3]} first-hit={first_hit or '-'!s:>2}  "
              f"R@3={recall_at_k(result, 3):.2f}  top: {top_label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-labels", action="store_true", help="validate the labels and exit")
    parser.add_argument("--embeddings", action="store_true", help="add dense retrieval (cached)")
    parser.add_argument("--allow-paid", action="store_true", help="let cache misses call the embedding API")
    parser.add_argument("--detail", choices=list(STRATEGIES), help="per-query ranks for one strategy")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.check_labels:
        return check_labels()
    if check_labels() != 0:
        return 1

    retrievers: list[Retriever] = [PositionalRetriever(), BM25Retriever()]
    if args.embeddings:
        if args.allow_paid:
            from dotenv import find_dotenv, load_dotenv

            load_dotenv(find_dotenv())
        run = prepare_embeddings(args.allow_paid)
        if run is None:
            return 2
        cost = run.paid_tokens / 1_000_000 * EMBEDDING_USD_PER_MILLION
        print(f"\n  embeddings: {run.unique_chunks} unique chunk texts + {run.unique_queries} unique queries; "
              f"cache held {run.store.loaded} before this run")
        print(f"  this run embedded {run.paid_texts} text(s), {run.paid_tokens} tokens, about ${cost:.5f}")
        retrievers.append(run.retriever)

    print_strategy_tables(retrievers)
    print_summary_ablation(retrievers[1:])
    print_top_k(retrievers[-1])
    if args.embeddings:
        print_paired(retrievers[-1])
    if args.detail:
        print_detail(args.detail, retrievers[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
