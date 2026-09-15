"""Retrieval metrics over content-anchor labels. Pure; no model, no I/O.

Recall@k is measured over **evidence items**, not chunks. Two overlapping windows
that both contain the same anchor are one piece of evidence found, not two — so a
strategy cannot raise its recall by duplicating text across chunks.
"""

from dataclasses import dataclass

from rr_dataset import LabeledQuery, contains_anchor


@dataclass(frozen=True)
class QueryResult:
    query: LabeledQuery
    ranking: list[int]  # chunk indices, best first; may be shorter than the resume
    # For each evidence item, the chunk indices containing it. Empty means no chunk
    # contains the anchor whole — the chunker cut through it.
    relevant: tuple[frozenset[int], ...]
    retrieved_tokens: tuple[int, ...]  # token_count of each ranked chunk, same order

    @property
    def unreachable(self) -> int:
        return sum(1 for chunks in self.relevant if not chunks)


def relevant_chunks(query: LabeledQuery, chunk_texts: list[str]) -> tuple[frozenset[int], ...]:
    return tuple(
        frozenset(i for i, text in enumerate(chunk_texts) if contains_anchor(text, anchor))
        for anchor in query.evidence
    )


def recall_at_k(result: QueryResult, k: int) -> float:
    top = set(result.ranking[:k])
    found = sum(1 for chunks in result.relevant if chunks & top)
    return found / len(result.relevant)


def reciprocal_rank(result: QueryResult) -> float:
    """1 / rank of the first chunk holding any evidence item; 0 if none is retrieved."""
    wanted = frozenset().union(*result.relevant)
    for rank, index in enumerate(result.ranking, start=1):
        if index in wanted:
            return 1.0 / rank
    return 0.0


def tokens_at_k(result: QueryResult, k: int) -> int:
    """Context a caller would pay for by taking the top k."""
    return sum(result.retrieved_tokens[:k])


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
