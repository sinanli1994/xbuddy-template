"""The resume store adapter: payload contract, scoping, and honest failure.

Offline. A fake supabase client records exactly what would be sent; the autouse
guards fail any test that reaches the real client or the embedding API.
"""

import logging
import math
import struct
from datetime import datetime

import pytest

from agents.xbuddy.resume.models import ResumeSection
from agents.xbuddy.resume.store import (
    MAX_MATCH_COUNT,
    ChunkRecord,
    ResumeStore,
    ResumeStoreError,
    build_match_payload,
    build_replace_payload,
    check_scope,
    serialize_vector,
)

D = 8  # small vectors: the adapter is dimension-agnostic; 1536 is pinned elsewhere


class Response:
    def __init__(self, data):
        self.data = data


class Query:
    """Records a table query's filters and returns canned rows."""

    def __init__(self, rows, log):
        self.rows, self.log = rows, log

    def select(self, columns):
        self.log.append(("select", columns))
        return self

    def eq(self, column, value):
        self.log.append(("eq", column, value))
        return self

    def limit(self, n):
        self.log.append(("limit", n))
        return self

    def execute(self):
        return Response(self.rows)


class FakeClient:
    def __init__(self, rpc_data=None, table_rows=None, fail=None):
        self.rpc_calls: list[tuple[str, dict]] = []
        self.table_log: list[tuple] = []
        self.rpc_data = rpc_data if rpc_data is not None else []
        self.table_rows = table_rows if table_rows is not None else []
        self.fail = fail

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        client = self

        class Call:
            def execute(self):
                if client.fail:
                    raise client.fail
                return Response(client.rpc_data)

        return Call()

    def table(self, name):
        self.table_log.append(("table", name))
        if self.fail:
            raise self.fail
        return Query(self.table_rows, self.table_log)


def chunk(i=0, section=ResumeSection.EXPERIENCE, vector=None):
    return ChunkRecord(
        chunk_index=i, section=section, content=f"[Experience] item {i}", token_count=4, page=1,
        embedding=vector or [0.1 * (i + 1)] * D,
    )


def store(client) -> ResumeStore:
    return ResumeStore(client, dimensions=D)


async def replace(s, chunks=None, **overrides):
    kwargs = {
        "user_id": 7, "thread_id": "thread-1", "filename": "cv.pdf", "content_sha256": "a" * 64,
        "page_count": 1, "embedding_model": "text-embedding-3-small", "candidate_facts": None,
        "chunks": chunks or [chunk(0), chunk(1)],
    }
    kwargs.update(overrides)
    return await s.replace(**kwargs)


STORED_ROW = {"document_id": "11111111-1111-1111-1111-111111111111", "chunk_count": 2,
              "created_at": "2026-09-11T12:00:00+00:00"}


def match_row(i=2, section="experience", similarity=0.9):
    return {"id": f"chunk-{i}", "document_id": "doc-1", "chunk_index": i, "section": section,
            "content": f"[Experience] item {i}", "token_count": 4, "page": 1, "similarity": similarity}


# --------------------------------------------------------------------------
# Vector serialisation
# --------------------------------------------------------------------------


def test_vectors_serialise_to_pgvector_text_form():
    assert serialize_vector([1.0, -0.5, 0.0, 2.0], 4) == "[1,-0.5,0,2]"


def test_serialisation_round_trips_float32_exactly():
    """pgvector stores float32; nine significant digits are enough to recover it."""
    values = [0.123456789, -3.4028234e38, 1e-7, 0.1, 2 / 3, math.pi, -1.17549435e-38, 12345.678]
    text = serialize_vector(values, len(values))
    parsed = [float(x) for x in text[1:-1].split(",")]
    as_f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
    assert [as_f32(p) for p in parsed] == [as_f32(v) for v in values]


def test_a_wrong_length_vector_is_refused_before_sending():
    with pytest.raises(ValueError, match="8-dimension"):
        serialize_vector([0.1] * 7, 8)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_refused(bad):
    with pytest.raises(ValueError, match="NaN or infinity"):
        serialize_vector([0.1] * 7 + [bad], 8)


# --------------------------------------------------------------------------
# Scope: always both halves
# --------------------------------------------------------------------------


@pytest.mark.parametrize("user_id", [0, -1, True, "7", None, 1.5])
def test_a_bad_user_id_is_refused(user_id):
    with pytest.raises(ValueError, match="user_id"):
        check_scope(user_id, "thread-1")


@pytest.mark.parametrize("thread_id", ["", "   ", None, 7])
def test_a_bad_thread_id_is_refused(thread_id):
    with pytest.raises(ValueError, match="thread_id"):
        check_scope(7, thread_id)


def test_every_payload_carries_both_scope_halves():
    replace_payload = build_replace_payload(
        user_id=7, thread_id="thread-1", filename="cv.pdf", content_sha256="a" * 64, page_count=1,
        embedding_model="m", dimensions=D, candidate_facts=None, chunks=[chunk()],
    )
    match_payload = build_match_payload(user_id=7, thread_id="thread-1", query_embedding=[0.1] * D, dimensions=D, k=3)
    for payload in (replace_payload, match_payload):
        assert payload["p_user_id"] == 7 and payload["p_thread_id"] == "thread-1"


# --------------------------------------------------------------------------
# Payload shapes
# --------------------------------------------------------------------------


def test_replace_payload_shape():
    payload = build_replace_payload(
        user_id=7, thread_id="thread-1", filename="cv.pdf", content_sha256="a" * 64, page_count=2,
        embedding_model="text-embedding-3-small", dimensions=D,
        candidate_facts={"current_role": "Engineer"}, chunks=[chunk(0, ResumeSection.SKILLS), chunk(1)],
    )
    assert payload["p_candidate_facts"] == {"current_role": "Engineer"}
    first = payload["p_chunks"][0]
    assert set(first) == {"chunk_index", "section", "content", "token_count", "page", "embedding"}
    assert first["section"] == "skills"  # the enum's value, lower case, as the CHECK expects
    assert first["embedding"].startswith("[") and first["embedding"].count(",") == D - 1


def test_replace_payload_refuses_no_chunks_and_duplicate_indices():
    base = {"user_id": 7, "thread_id": "t", "filename": "f", "content_sha256": "a" * 64, "page_count": 1,
            "embedding_model": "m", "dimensions": D, "candidate_facts": None}
    with pytest.raises(ValueError, match="at least one chunk"):
        build_replace_payload(**base, chunks=[])
    with pytest.raises(ValueError, match="unique"):
        build_replace_payload(**base, chunks=[chunk(0), chunk(0)])


@pytest.mark.parametrize("k", [1, 3, 5, MAX_MATCH_COUNT])
def test_k_is_propagated_unchanged(k):
    payload = build_match_payload(user_id=7, thread_id="t", query_embedding=[0.1] * D, dimensions=D, k=k)
    assert payload["p_match_count"] == k


@pytest.mark.parametrize("k", [0, -1, MAX_MATCH_COUNT + 1, True, 3.0])
def test_k_out_of_range_is_refused_rather_than_clamped(k):
    with pytest.raises(ValueError, match="k must be"):
        build_match_payload(user_id=7, thread_id="t", query_embedding=[0.1] * D, dimensions=D, k=k)


# --------------------------------------------------------------------------
# replace — the upload path, which must fail loudly
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replace_calls_the_rpc_once_and_returns_the_document():
    client = FakeClient(rpc_data=[STORED_ROW])
    stored = await replace(store(client))
    assert [name for name, _ in client.rpc_calls] == ["replace_resume"]
    assert stored.document_id == STORED_ROW["document_id"] and stored.chunk_count == 2
    assert isinstance(stored.created_at, datetime)


@pytest.mark.asyncio
async def test_replace_is_one_call_so_the_swap_is_atomic():
    """Delete-old and insert-new happen inside one RPC — one transaction. The adapter
    must never split them into separate requests that could half-succeed."""
    client = FakeClient(rpc_data=[STORED_ROW])
    await replace(store(client))
    assert len(client.rpc_calls) == 1 and client.table_log == []


@pytest.mark.asyncio
async def test_a_database_error_on_replace_raises():
    client = FakeClient(fail=RuntimeError("connection reset by peer at 10.0.0.1"))
    with pytest.raises(ResumeStoreError) as caught:
        await replace(store(client))
    assert "10.0.0.1" not in str(caught.value)  # no connection detail in the message


@pytest.mark.asyncio
async def test_missing_credentials_on_replace_raise():
    client = FakeClient(fail=ValueError("Supabase credentials not configured"))
    with pytest.raises(ResumeStoreError):
        await replace(store(client))


@pytest.mark.parametrize("rows", [[], [STORED_ROW, STORED_ROW]])
@pytest.mark.asyncio
async def test_replace_must_return_exactly_one_row(rows):
    with pytest.raises(ResumeStoreError, match="expected 1"):
        await replace(store(FakeClient(rpc_data=rows)))


@pytest.mark.asyncio
async def test_a_chunk_count_mismatch_is_an_error_not_a_success():
    """The deployed function disagreeing with what was sent means it is not the
    function this adapter was written against."""
    with pytest.raises(ResumeStoreError, match="chunk count"):
        await replace(store(FakeClient(rpc_data=[{**STORED_ROW, "chunk_count": 1}])))


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_is_none_when_nothing_is_on_file():
    assert await store(FakeClient(table_rows=[])).status(user_id=7, thread_id="thread-1") is None


@pytest.mark.asyncio
async def test_status_filters_by_both_user_and_thread():
    client = FakeClient(table_rows=[])
    await store(client).status(user_id=7, thread_id="thread-1")
    filters = {entry[1:] for entry in client.table_log if entry[0] == "eq"}
    assert filters == {("user_id", 7), ("thread_id", "thread-1")}
    assert ("table", "resume_documents") in client.table_log


@pytest.mark.asyncio
async def test_status_never_selects_resume_text():
    client = FakeClient(table_rows=[])
    await store(client).status(user_id=7, thread_id="thread-1")
    columns = next(entry[1] for entry in client.table_log if entry[0] == "select")
    assert "content" not in columns.replace("content_sha256", "")


@pytest.mark.asyncio
async def test_status_parses_a_row():
    row = {"document_id": "doc-1", "filename": "cv.pdf", "page_count": 2, "chunk_count": 9,
           "embedding_model": "text-embedding-3-small", "content_sha256": "a" * 64,
           "created_at": "2026-09-11T12:00:00+00:00", "candidate_facts": {"current_role": "Engineer"}}
    status = await store(FakeClient(table_rows=[row])).status(user_id=7, thread_id="thread-1")
    assert status.chunk_count == 9 and status.candidate_facts == {"current_role": "Engineer"}


@pytest.mark.asyncio
async def test_status_failure_raises():
    with pytest.raises(ResumeStoreError):
        await store(FakeClient(fail=RuntimeError("down"))).status(user_id=7, thread_id="thread-1")


# --------------------------------------------------------------------------
# match
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_match_sends_the_query_vector_and_k():
    client = FakeClient(rpc_data=[match_row()])
    await store(client).match(user_id=7, thread_id="thread-1", query_embedding=[0.2] * D, k=3)
    name, params = client.rpc_calls[0]
    assert name == "match_resume_chunks"
    assert params["p_match_count"] == 3 and params["p_query_embedding"] == serialize_vector([0.2] * D, D)


@pytest.mark.asyncio
async def test_match_parses_rows_into_typed_chunks():
    chunks = await store(FakeClient(rpc_data=[match_row(2), match_row(5, "skills", 0.7)])).match(
        user_id=7, thread_id="thread-1", query_embedding=[0.2] * D, k=3
    )
    assert [(c.chunk_index, c.section) for c in chunks] == [(2, ResumeSection.EXPERIENCE), (5, ResumeSection.SKILLS)]
    assert chunks[0].similarity == 0.9


@pytest.mark.asyncio
async def test_a_summary_row_is_dropped_and_reported(caplog):
    """The RPC excludes Summary. If one arrives, the deployed function is wrong —
    the adapter says so rather than feeding it to the conversation."""
    rows = [match_row(1, "summary", 0.99), match_row(2)]
    with caplog.at_level(logging.ERROR):
        chunks = await store(FakeClient(rpc_data=rows)).match(
            user_id=7, thread_id="t", query_embedding=[0.2] * D, k=3
        )
    assert [c.section for c in chunks] == [ResumeSection.EXPERIENCE]
    assert "Summary chunk returned" in caplog.text


@pytest.mark.asyncio
async def test_match_never_returns_more_than_k():
    rows = [match_row(i) for i in range(6)]
    chunks = await store(FakeClient(rpc_data=rows)).match(user_id=7, thread_id="t", query_embedding=[0.2] * D, k=3)
    assert len(chunks) == 3


@pytest.mark.asyncio
async def test_match_failure_raises_at_the_store_layer():
    """The store is honest; degrading is the retrieval helper's decision."""
    with pytest.raises(ResumeStoreError):
        await store(FakeClient(fail=RuntimeError("down"))).match(
            user_id=7, thread_id="t", query_embedding=[0.2] * D, k=3
        )


@pytest.mark.asyncio
async def test_a_malformed_row_raises():
    with pytest.raises(ResumeStoreError, match="unexpected shape"):
        await store(FakeClient(rpc_data=[{"id": "x"}])).match(
            user_id=7, thread_id="t", query_embedding=[0.2] * D, k=3
        )


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_real_client_is_unreachable_in_tests(no_real_resume_store):
    with pytest.raises(ResumeStoreError):
        await ResumeStore(dimensions=D).status(user_id=7, thread_id="thread-1")
    assert no_real_resume_store == ["resume store client"]
    no_real_resume_store.clear()


def test_the_production_store_uses_1536_dimensions():
    assert ResumeStore(FakeClient()).dimensions == 1536
