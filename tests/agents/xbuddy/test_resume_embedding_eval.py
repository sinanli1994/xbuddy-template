"""The embedding half of the retrieval eval, run entirely offline.

Uses a deterministic bag-of-words double in place of the API, so these pin the
eval's *plumbing* — the disk cache, cold-versus-warm equivalence, the Summary
exclusion, and the embedding retriever — not the quality of real embeddings. The
autouse guard in tests/conftest.py fails the run if anything reaches the API.
"""

import hashlib
import re
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[3] / "evals" / "resume_retrieval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from rr_embedding_cache import JsonEmbeddingStore
from rr_retrievers import EmbeddingRetriever
from rr_stats import paired_bootstrap
from run_resume_retrieval_eval import ABLATIONS, chunk_all, evaluate, summarize

from agents.xbuddy.resume.embeddings import CachedEmbedder, rank_by_cosine
from agents.xbuddy.resume.models import ResumeSection

DIMS = 48


class FakeEmbedder:
    def __init__(self) -> None:
        self.sent = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.sent += len(texts)
        out = []
        for text in texts:
            vector = [0.0] * DIMS
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.sha256(word.encode()).digest()
                # Non-integer weights, so float32 rounding in the disk store is exercised.
                vector[digest[0] % DIMS] += 1.0 + digest[1] / 997
            out.append(vector)
        return out


def cached(store, fake=None, allow_calls=True) -> CachedEmbedder:
    return CachedEmbedder(fake or FakeEmbedder(), model="test-model", dimensions=DIMS, cache=store,
                          allow_calls=allow_calls)


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------


def test_the_store_round_trips_through_disk(tmp_path):
    store = JsonEmbeddingStore("test-model", DIMS, directory=tmp_path)
    cached(store).embed(["alpha beta", "gamma"])
    store.save()

    reloaded = JsonEmbeddingStore("test-model", DIMS, directory=tmp_path)
    assert reloaded.loaded == 2
    assert dict(reloaded) == dict(store)


def test_the_value_returned_on_write_is_the_value_loaded_later(tmp_path):
    """The store rounds to float32 on write. If the first run saw unrounded vectors
    and later runs saw rounded ones, a near-tie could rank differently between them."""
    store = JsonEmbeddingStore("test-model", DIMS, directory=tmp_path)
    first = cached(store).embed(["alpha beta gamma"])[0]
    store.save()
    later = cached(JsonEmbeddingStore("test-model", DIMS, directory=tmp_path), allow_calls=False).embed(
        ["alpha beta gamma"]
    )[0]
    assert first == later


def test_cold_and_warm_caches_rank_identically(tmp_path):
    query = "kafka pipeline latency"
    chunks = ["built a kafka pipeline", "reduced latency", "wrote docs", "kafka latency pipeline tuning"]

    cold_store = JsonEmbeddingStore("test-model", DIMS, directory=tmp_path)
    cold = cached(cold_store)
    cold_vectors = cold.embed([query, *chunks])
    cold_store.save()

    fake = FakeEmbedder()
    warm = cached(JsonEmbeddingStore("test-model", DIMS, directory=tmp_path), fake=fake, allow_calls=False)
    warm_vectors = warm.embed([query, *chunks])

    assert fake.sent == 0
    assert rank_by_cosine(cold_vectors[0], cold_vectors[1:]) == rank_by_cosine(warm_vectors[0], warm_vectors[1:])


def test_a_cache_file_for_another_model_is_refused(tmp_path):
    store = JsonEmbeddingStore("model-a", DIMS, directory=tmp_path)
    cached(store).embed(["x"])
    store.save()
    (tmp_path / f"model-b-{DIMS}.json").write_text((tmp_path / f"model-a-{DIMS}.json").read_text())
    with pytest.raises(ValueError):
        JsonEmbeddingStore("model-b", DIMS, directory=tmp_path)


def test_saving_leaves_no_temporary_file(tmp_path):
    store = JsonEmbeddingStore("test-model", DIMS, directory=tmp_path)
    cached(store).embed(["x"])
    store.save()
    assert [p.name for p in tmp_path.iterdir()] == [f"test-model-{DIMS}.json"]


# --------------------------------------------------------------------------
# Embedding retriever and the Summary ablation
# --------------------------------------------------------------------------


def test_the_embedding_retriever_ranks_the_matching_chunk_first():
    retriever = EmbeddingRetriever(cached({}))
    ranking = retriever.rank("kafka pipeline", ["taught maths", "built a kafka pipeline", "wrote docs"])
    assert ranking[0] == 1
    assert sorted(ranking) == [0, 1, 2]  # dense retrieval ranks every candidate


def test_excluded_sections_are_never_returned():
    retriever = EmbeddingRetriever(cached({}))
    exclude = ABLATIONS["section -summary -header"]
    chunked = chunk_all("section")
    for result in evaluate("section", retriever, exclude):
        sections = [c.section for c in chunked[result.query.resume].chunks]
        assert all(sections[i] not in exclude for i in result.ranking)


def test_exclusion_maps_ranks_back_to_the_original_chunk_indices():
    """Ranking happens over the filtered list; results must still point at the right
    chunks, or relevance would be judged against the wrong text."""
    retriever = EmbeddingRetriever(cached({}))
    chunked = chunk_all("section")
    plain = {r.query.id: r for r in evaluate("section", retriever)}
    for result in evaluate("section", retriever, ABLATIONS["section -summary"]):
        kept = [i for i in plain[result.query.id].ranking
                if chunked[result.query.resume].chunks[i].section is not ResumeSection.SUMMARY]
        assert result.ranking == kept


def test_an_anchor_only_an_excluded_chunk_holds_counts_as_a_miss():
    """Excluding a section must not quietly drop the queries it would have answered."""
    retriever = EmbeddingRetriever(cached({}))
    results = evaluate("section", retriever, frozenset({ResumeSection.EXPERIENCE}))
    assert summarize(results)["R@5"] < summarize(evaluate("section", retriever))["R@5"]


def test_the_whole_eval_runs_offline_through_the_embedding_path():
    fake = FakeEmbedder()
    retriever = EmbeddingRetriever(cached({}, fake=fake))
    for strategy in ("section", "entry", "window-80", "window-250"):
        s = summarize(evaluate(strategy, retriever))
        assert {"R@1", "R@3", "R@5", "MRR", "tok@3", "frac@5"} <= set(s)
        assert 0.0 <= s["R@1"] <= s["R@3"] <= s["R@5"] <= 1.0
    assert fake.sent > 0  # the double, not the API, did the embedding


# --------------------------------------------------------------------------
# Paired comparison
# --------------------------------------------------------------------------


def test_identical_configurations_are_indistinguishable():
    res = paired_bootstrap([1.0, 0.5, 0.33], [1.0, 0.5, 0.33])
    assert (res.mean, res.low, res.high) == (0.0, 0.0, 0.0)
    assert (res.wins, res.losses, res.ties) == (0, 0, 3)
    assert not res.significant


def test_a_consistent_improvement_is_significant():
    a = [1.0] * 20
    b = [0.5] * 15 + [1.0] * 5
    res = paired_bootstrap(a, b)
    assert res.significant and res.low > 0
    assert (res.wins, res.losses, res.ties) == (15, 0, 5)


def test_a_mixed_result_is_not():
    """Wins on some queries, losses on others: the shape the chunking comparison had."""
    a = [1.0, 0.2, 1.0, 0.33, 1.0, 0.25, 0.5, 1.0]
    b = [0.33, 1.0, 0.25, 1.0, 0.5, 1.0, 1.0, 0.33]
    assert not paired_bootstrap(a, b).significant


def test_the_bootstrap_is_seeded():
    a, b = [1.0, 0.5, 0.2, 0.33], [0.5, 0.5, 1.0, 0.2]
    assert paired_bootstrap(a, b) == paired_bootstrap(a, b)


def test_unpaired_scores_are_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.5])
