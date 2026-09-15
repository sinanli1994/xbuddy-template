"""POST /resume and POST /resume/status.

The real pipeline runs — pypdf extraction, section-aware chunking, the embedding
batch, the store adapter — with only the embedder, the store, and the candidate
model replaced. So "never report indexed unless persistence succeeded" is tested
through the endpoint, not assumed from a unit test.
"""

import logging
import sys
from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

import service.service as svc
from agents.xbuddy.resume.store import ResumeStatus, ResumeStoreError, StoredResume
from service import app

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "agents" / "xbuddy"))
from resume_pdf import encrypted_pdf, image_only_pdf, text_pdf

RESUME_TEXT = (ROOT / "evals" / "resume_retrieval" / "corpus" / "backend_to_ai.txt").read_text(encoding="utf-8")
PDF = text_pdf(RESUME_TEXT)
OWNER, INTRUDER, THREAD = 4242, 5151, "resume-thread"
D = 1536


@pytest.fixture(autouse=True)
def no_rate_limiting(monkeypatch):
    """These tests make more than RATE_LIMIT_EXPENSIVE uploads; one test below
    turns the limiter back on to check it applies."""
    monkeypatch.setattr(svc.limiter, "enabled", False)


@pytest.fixture(autouse=True)
def no_real_supabase(monkeypatch):
    from agents.xbuddy.resume import store as store_module

    def blocked():
        raise AssertionError("resume endpoint test reached for the real Supabase client")

    monkeypatch.setattr(store_module, "_default_client", blocked)


class FakeStore:
    instances: ClassVar[list["FakeStore"]] = []
    fail_replace = False
    fail_status = False
    on_file: ResumeStatus | None = None

    def __init__(self, *args, **kwargs):
        self.replaced: list[dict] = []
        FakeStore.instances.append(self)

    async def replace(self, **kwargs):
        self.replaced.append(kwargs)
        if FakeStore.fail_replace:
            raise ResumeStoreError("resume replace failed")
        return StoredResume(document_id="doc-777", chunk_count=len(kwargs["chunks"]),
                            created_at="2026-09-11T12:00:00+00:00")

    async def status(self, *, user_id, thread_id):
        if FakeStore.fail_status:
            raise ResumeStoreError("resume status failed")
        return FakeStore.on_file


@pytest.fixture
def pipeline(monkeypatch):
    """The real pipeline with fakes at its three external edges."""
    FakeStore.instances, FakeStore.fail_replace, FakeStore.fail_status, FakeStore.on_file = [], False, False, None
    monkeypatch.setattr("agents.xbuddy.resume.ingestion.ResumeStore", FakeStore)
    monkeypatch.setattr(svc, "ResumeStore", FakeStore)
    embeds: list[list[str]] = []

    def embed(texts):
        embeds.append(list(texts))
        return [[0.01 * (i + 1)] + [0.0] * (D - 1) for i, _ in enumerate(texts)]

    monkeypatch.setattr("agents.xbuddy.resume.ingestion._default_embed_batch", embed)
    candidates = AsyncMock(return_value={"current_role": "Senior Backend Engineer", "years_experience": 8,
                                         "highest_education": "BASc", "work_history": ["a"]})
    monkeypatch.setattr(svc, "extract_background_candidates", candidates)
    return {"embeds": embeds, "candidates": candidates}


def agent_owned_by(user_id):
    agent = Mock()

    async def aget_state(config=None):
        snapshot = Mock()
        snapshot.values = {"user_id": user_id} if user_id is not None else {}
        return snapshot

    agent.aget_state = aget_state
    return agent


def upload(data=PDF, *, filename="Jordan CV.pdf", content_type="application/pdf", user_id=OWNER,
           thread_id=THREAD, owner=OWNER, headers=None, path="/resume"):
    with patch.object(svc, "get_agent", Mock(return_value=agent_owned_by(owner))):
        return TestClient(app).post(
            path, files={"file": (filename, data, content_type)},
            data={"user_id": str(user_id), "thread_id": thread_id}, headers=headers or {},
        )


def status_of(*, user_id=OWNER, thread_id=THREAD, owner=OWNER, path="/resume/status"):
    with patch.object(svc, "get_agent", Mock(return_value=agent_owned_by(owner))):
        return TestClient(app).post(path, json={"user_id": user_id, "thread_id": thread_id})


# --------------------------------------------------------------------------
# Success
# --------------------------------------------------------------------------


def test_an_upload_is_indexed_and_reports_metadata_only(pipeline):
    response = upload()
    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {"indexed": True, "document_id": "doc-777", "filename": "Jordan CV.pdf",
                    "page_count": 2, "chunk_count": 12, "indexed_at": "2026-09-11T12:00:00Z"}


def test_the_response_never_contains_resume_text_or_candidates(pipeline):
    text = upload().text
    for leak in ("Northwind", "Kafka", "Senior Backend Engineer", "BASc", "candidate", "content"):
        assert leak not in text


def test_the_prefixed_route_behaves_the_same(pipeline):
    assert upload(path="/xbuddy/resume").json()["chunk_count"] == 12


def test_candidates_are_extracted_from_the_same_text_and_stored(pipeline):
    upload()
    (text,), _ = pipeline["candidates"].call_args
    assert "Northwind Logistics" in text
    stored = FakeStore.instances[-1].replaced[0]
    assert stored["candidate_facts"]["current_role"] == "Senior Backend Engineer"
    assert (stored["user_id"], stored["thread_id"]) == (OWNER, THREAD)


def test_no_candidates_still_indexes(pipeline):
    pipeline["candidates"].return_value = None
    assert upload().status_code == 200
    assert FakeStore.instances[-1].replaced[0]["candidate_facts"] is None


def test_every_chunk_is_embedded_in_one_batch(pipeline):
    upload()
    assert len(pipeline["embeds"]) == 1 and len(pipeline["embeds"][0]) == 12


def test_a_new_conversation_may_attach_a_resume_before_its_first_message(pipeline):
    assert upload(owner=None).status_code == 200


# --------------------------------------------------------------------------
# Refusals — useful 4xx, nothing indexed
# --------------------------------------------------------------------------


def test_a_non_pdf_is_refused(pipeline):
    response = upload(b"just some text", filename="cv.txt", content_type="text/plain")
    assert response.status_code == 415 and response.json()["detail"]["code"] == "not_pdf"
    assert FakeStore.instances == []


def test_a_pdf_named_file_that_is_not_a_pdf_is_refused(pipeline):
    response = upload(b"not really a pdf", filename="cv.pdf")
    assert response.status_code == 415 and response.json()["detail"]["code"] == "not_pdf"


def test_a_file_over_two_megabytes_is_refused(pipeline):
    response = upload(b"%PDF-1.4\n" + b"0" * (2 * 1024 * 1024))
    assert response.status_code == 413 and response.json()["detail"]["code"] == "too_large"
    assert pipeline["candidates"].await_count == 0


def test_an_image_only_pdf_is_a_clear_422(pipeline):
    response = upload(image_only_pdf())
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "no_text"
    assert "scanned" in response.json()["detail"]["message"].lower()
    assert pipeline["candidates"].await_count == 0 and FakeStore.instances == []


def test_a_password_protected_pdf_is_refused(pipeline):
    response = upload(encrypted_pdf(RESUME_TEXT, user_password="hunter2"))
    assert response.status_code == 422 and response.json()["detail"]["code"] == "encrypted"


def test_a_damaged_pdf_is_refused(pipeline):
    response = upload(b"%PDF-1.4\ngarbage that is not a pdf\n%%EOF")
    assert response.status_code == 422 and response.json()["detail"]["code"] == "unreadable"


def test_an_empty_file_is_refused(pipeline):
    assert upload(b"").status_code == 422


@pytest.mark.parametrize("user_id", ["0", "-3", "abc"])
def test_a_bad_user_id_is_refused(pipeline, user_id):
    assert upload(user_id=user_id).status_code == 422


def test_a_blank_thread_is_refused(pipeline):
    assert upload(thread_id="   ").status_code == 422


# --------------------------------------------------------------------------
# Failures after the file is accepted: explicit, never "indexed"
# --------------------------------------------------------------------------


def test_a_database_failure_is_not_reported_as_success(pipeline):
    FakeStore.fail_replace = True
    response = upload()
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "persistence_failed"
    assert "indexed" not in response.json()


def test_an_embedding_failure_is_not_reported_as_success(pipeline, monkeypatch):
    def broken(texts):
        raise RuntimeError("openai 503")

    monkeypatch.setattr("agents.xbuddy.resume.ingestion._default_embed_batch", broken)
    response = upload()
    assert response.status_code == 502 and response.json()["detail"]["code"] == "embedding_failed"
    assert FakeStore.instances == [] or FakeStore.instances[-1].replaced == []


# --------------------------------------------------------------------------
# Ownership and auth
# --------------------------------------------------------------------------


def test_uploading_into_someone_elses_conversation_is_404(pipeline):
    response = upload(user_id=INTRUDER, owner=OWNER)
    assert response.status_code == 404
    assert FakeStore.instances == [] and pipeline["candidates"].await_count == 0


def test_upload_requires_the_bearer_token(pipeline, auth_secret):
    assert upload().status_code == 401
    assert upload(headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert upload(headers={"Authorization": f"Bearer {auth_secret}"}).status_code == 200


def test_status_requires_the_bearer_token(pipeline, auth_secret):
    with patch.object(svc, "get_agent", Mock(return_value=agent_owned_by(OWNER))):
        client = TestClient(app)
        assert client.post("/resume/status", json={"user_id": OWNER, "thread_id": THREAD}).status_code == 401


def test_upload_is_rate_limited_as_expensive(pipeline, monkeypatch):
    monkeypatch.setattr(svc.limiter, "enabled", True)
    svc.limiter.reset()
    try:
        codes = [upload().status_code for _ in range(11)]
    finally:
        svc.limiter.reset()
    assert codes[:10] == [200] * 10 and codes[10] == 429


# --------------------------------------------------------------------------
# /resume/status
# --------------------------------------------------------------------------


def test_status_without_a_resume(pipeline):
    assert status_of().json() == {"has_resume": False, "filename": None, "page_count": None,
                                  "chunk_count": None, "indexed_at": None}


def test_status_with_a_resume_never_includes_content_or_candidates(pipeline):
    FakeStore.on_file = ResumeStatus(document_id="doc-777", filename="cv.pdf", page_count=2, chunk_count=12,
                                     embedding_model="text-embedding-3-small", content_sha256="a" * 64,
                                     created_at="2026-09-11T12:00:00+00:00",
                                     candidate_facts={"current_role": "Senior Backend Engineer"})
    response = status_of()
    assert response.json() == {"has_resume": True, "filename": "cv.pdf", "page_count": 2,
                               "chunk_count": 12, "indexed_at": "2026-09-11T12:00:00Z"}
    assert "Senior Backend Engineer" not in response.text and "doc-777" not in response.text


def test_status_for_someone_elses_conversation_is_404(pipeline):
    assert status_of(user_id=INTRUDER, owner=OWNER).status_code == 404


def test_a_status_failure_is_an_error_not_no_resume(pipeline):
    FakeStore.fail_status = True
    response = status_of()
    assert response.status_code == 502 and response.json()["detail"]["code"] == "status_unavailable"


def test_status_validates_its_input(pipeline):
    assert status_of(user_id=0).status_code == 422
    assert status_of(thread_id="").status_code == 422


# --------------------------------------------------------------------------
# Privacy of the request log
# --------------------------------------------------------------------------


def test_the_request_log_never_contains_the_upload(pipeline, caplog):
    """`setup_logging` raises service.service to WARNING by default, which hides the
    middleware — and would make every leak assertion here pass vacuously. So the
    capture is taken at that logger's level, and proven to see the middleware first."""
    with caplog.at_level(logging.INFO, logger="service.service"):
        upload()
    logged = caplog.text
    assert "FRONTEND_REQUEST: POST" in logged, "capture is not seeing the request middleware"
    assert "multipart upload; not logged" in logged
    for leak in ("Jordan CV.pdf", "Northwind", "%PDF", "JORDAN AVERY"):
        assert leak not in logged


def test_the_leak_check_would_catch_a_logged_body(pipeline, caplog):
    """Control: the same PDF bytes sent as a non-multipart body ARE logged by the
    middleware's raw-body fallback — and the same capture sees them. Without this,
    the test above could pass because it sees nothing rather than because nothing
    leaked."""
    with caplog.at_level(logging.INFO, logger="service.service"):
        # The endpoint rejects the binary body; only what the middleware logged matters.
        TestClient(app, raise_server_exceptions=False).post(
            "/resume/status", content=PDF[:600], headers={"Content-Type": "application/octet-stream"}
        )
    assert "%PDF" in caplog.text and "JORDAN AVERY" in caplog.text
