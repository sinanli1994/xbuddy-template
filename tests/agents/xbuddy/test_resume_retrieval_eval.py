"""The resume retrieval eval's interface: labels, metrics, and the keyword baseline.

Stage 1 locks these before any embedding is computed, so a Stage 2 change to the
retriever is measured against labels and metrics that did not move with it.
"""

import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[3] / "evals" / "resume_retrieval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from rr_dataset import QUERIES, RESUMES, LabeledQuery
from rr_metrics import (
    QueryResult,
    recall_at_k,
    reciprocal_rank,
    relevant_chunks,
    tokens_at_k,
)
from rr_retrievers import BM25Retriever, PositionalRetriever, light_stem
from run_resume_retrieval_eval import check_labels, evaluate, query_kind, summarize

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def test_the_labels_are_valid(capsys):
    """Every anchor occurs exactly once in its resume, and ids are unique."""
    assert check_labels() == 0


def test_the_dataset_is_the_agreed_size():
    """Stage 2 target: ~31-33 queries. Small on purpose; growth should be deliberate."""
    assert 31 <= len(QUERIES) <= 33
    assert {q.resume for q in QUERIES} == set(RESUMES)


def test_both_query_kinds_are_represented():
    """Stage 2 target: ~12-14 paraphrased queries. With the 7 Stage 1 had, each one
    moved the semantic score by 0.14 and the embedding comparison was noise-bound."""
    kinds = [query_kind(q) for q in QUERIES]
    assert 12 <= kinds.count("semantic") <= 14
    assert kinds.count("lexical") >= 12


@pytest.mark.parametrize("strategy", ["section", "entry", "window-250", "window-80"])
def test_every_evidence_item_is_reachable(strategy):
    """An anchor no chunk contains whole would be an unanswerable query."""
    assert summarize(evaluate(strategy, PositionalRetriever()))["unreachable"] == 0


# --------------------------------------------------------------------------
# Metric arithmetic
# --------------------------------------------------------------------------


QUERY = LabeledQuery("x", "backend_to_ai", "q", ("alpha", "beta"), "experience")


def result(ranking, chunk_texts):
    return QueryResult(
        query=QUERY,
        ranking=ranking,
        relevant=relevant_chunks(QUERY, chunk_texts),
        retrieved_tokens=tuple(10 for _ in ranking),
    )


def test_recall_counts_evidence_items_not_chunks():
    r = result([0, 1, 2], ["alpha here", "alpha again", "beta there"])
    assert recall_at_k(r, 1) == 0.5
    assert recall_at_k(r, 2) == 0.5  # two chunks with the same anchor are one item
    assert recall_at_k(r, 3) == 1.0


def test_reciprocal_rank_uses_the_first_relevant_chunk():
    assert reciprocal_rank(result([2, 0], ["x", "y", "beta"])) == 1.0
    assert reciprocal_rank(result([0, 1, 2], ["x", "y", "alpha"])) == pytest.approx(1 / 3)
    assert reciprocal_rank(result([0, 1], ["x", "y", "alpha"])) == 0.0


def test_an_anchor_cut_in_half_is_unreachable():
    r = result([0, 1], ["...alp", "ha... beta"])
    assert r.unreachable == 1
    assert recall_at_k(r, 2) == 0.5


def test_anchor_matching_ignores_case_and_whitespace():
    r = result([0], ["ALPHA\n  and BETA"])
    assert recall_at_k(r, 1) == 1.0


def test_tokens_at_k_sums_the_top_k():
    assert tokens_at_k(result([0, 1, 2], ["a", "b", "c"]), 2) == 20


# --------------------------------------------------------------------------
# Keyword baseline
# --------------------------------------------------------------------------


def test_light_stemming_meets_plurals_and_verb_forms():
    assert light_stem("dashboards") == light_stem("dashboard")
    assert light_stem("modelling") == light_stem("modelled")
    assert light_stem("process") == "process"  # "ss" is not a plural
    assert light_stem("uses") == "uses"  # too short to strip


def test_the_retriever_tokenizer_actually_stems():
    """Testing `light_stem` alone would pass even if nothing called it."""
    from rr_retrievers import terms

    assert terms("Tableau dashboards") == terms("tableau dashboard")
    assert terms("modelling") == terms("modelled")


def test_bm25_ranks_only_chunks_sharing_a_term():
    """A chunk with no shared term is not "retrieved" by document order."""
    ranking = BM25Retriever().rank("kafka pipeline", ["nothing relevant", "a kafka pipeline", "pipeline"])
    assert ranking == [1, 2]


def test_bm25_beats_the_query_blind_floor():
    """If it didn't, the queries would not be measuring retrieval at all."""
    bm25 = summarize(evaluate("section", BM25Retriever()))
    floor = summarize(evaluate("section", PositionalRetriever()))
    assert bm25["MRR"] > floor["MRR"] + 0.2
    assert bm25["R@1"] > floor["R@1"]
