"""Startup warm-up: the first request must not pay the provider-stack import.

Production evidence (Fly, shared-cpu-1x, after a restart): `import core.llm` took
~9.7 s and stalled the event loop for all of it, and the first `/resume/status` paid
it because `ResumeStore()` imports `core.llm` lazily. These tests pin the fix:
the lifespan does that work once, in a worker thread, before the app is ready —
without calling a model, touching data, or special-casing health.
"""

import ast
import inspect
import logging
import sys
import textwrap
import threading
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import service.service as svc
from integrations.supabase import supabase_client as sb
from service import warmup

URL = "https://warmup-test.supabase.co"


class RecordingClient:
    """Stands in for the Supabase client; records any data access at all."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append(name)
            return self

        return record


@pytest.fixture
def supabase_factory(monkeypatch):
    """A fresh client singleton and a counting `create_client`."""
    monkeypatch.setattr(sb, "_supabase_client", None)
    client = RecordingClient()
    created: list[str] = []

    def create_client(url, key):
        created.append(url)
        return client

    monkeypatch.setattr(sb, "create_client", create_client)
    return created, client


@pytest.fixture
def resume_rag_on(monkeypatch):
    monkeypatch.setattr(sb.settings, "SUPABASE_URL", URL)
    monkeypatch.setattr(sb.settings, "SUPABASE_SECRET_KEY", SecretStr("sb_secret_test"))
    monkeypatch.setattr(sb.settings, "SUPABASE_SERVICE_ROLE_KEY", None)


@pytest.fixture
def resume_rag_off(monkeypatch):
    monkeypatch.setattr(sb.settings, "SUPABASE_URL", None)
    monkeypatch.setattr(sb.settings, "SUPABASE_SECRET_KEY", None)
    monkeypatch.setattr(sb.settings, "SUPABASE_SERVICE_ROLE_KEY", None)


@pytest.fixture
def postgres_lifespan(monkeypatch):
    """The Postgres branch of the lifespan with its database pieces stubbed."""
    monkeypatch.setattr(svc.settings, "DATABASE_TYPE", svc.DatabaseType.POSTGRES)
    monkeypatch.setattr(svc.settings, "USE_SUPABASE_REALTIME", False)
    monkeypatch.setattr(svc.pg_manager, "setup", AsyncMock())
    monkeypatch.setattr(svc.pg_manager, "cleanup", AsyncMock())
    monkeypatch.setattr(svc.pg_manager, "get_saver", Mock(return_value=Mock(name="saver")))
    monkeypatch.setattr(svc.pg_manager, "get_store", Mock(return_value=Mock(name="store")))
    monkeypatch.setattr(svc, "get_all_agent_info", Mock(return_value=[]))


# --------------------------------------------------------------------------
# 1. The lifespan invokes the seam, off the event loop, before readiness
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_lifespan_warms_dependencies_before_the_app_is_ready(postgres_lifespan, monkeypatch):
    ran_on: list[int] = []
    warm = Mock(side_effect=lambda: ran_on.append(threading.get_ident()))
    monkeypatch.setattr(svc, "warm_request_dependencies", warm)

    async with svc.lifespan(svc.app):
        # Inside the context the app is serving: the warm-up has already finished.
        warm.assert_called_once_with()

    assert ran_on and ran_on[0] != threading.get_ident(), "warm-up must run in a worker thread"


@pytest.mark.asyncio
async def test_the_warm_up_runs_before_the_database_is_set_up(postgres_lifespan, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(svc, "warm_request_dependencies", Mock(side_effect=lambda: order.append("warm")))
    monkeypatch.setattr(svc.pg_manager, "setup", AsyncMock(side_effect=lambda: order.append("database")))

    async with svc.lifespan(svc.app):
        pass

    assert order == ["warm", "database"]


# --------------------------------------------------------------------------
# 2–3. The first request re-runs nothing; the warm-up is idempotent
# --------------------------------------------------------------------------


def test_the_first_request_reuses_the_warmed_client(resume_rag_on, supabase_factory):
    created, client = supabase_factory

    warmup.warm_request_dependencies()
    first_request_client = sb.get_supabase_client()  # what ResumeStore's first query calls

    assert created == [URL]
    assert first_request_client is client


def test_warming_twice_builds_one_client_and_imports_once(resume_rag_on, supabase_factory):
    created, _ = supabase_factory

    warmup.warm_request_dependencies()
    provider_module = sys.modules["core.llm"]
    warmup.warm_request_dependencies()

    assert created == [URL]
    assert sys.modules["core.llm"] is provider_module  # imported once, never re-executed


def test_the_warm_up_really_imports_the_provider_stack(monkeypatch):
    """The test process already has core.llm loaded (the root conftest's embedding
    guard imports it), which would hide a warm-up that imports nothing. Unload it
    for this test only; monkeypatch restores the original module object afterwards,
    both in sys.modules and on the package, so no other test sees a second copy."""
    import core
    import core.llm as original

    monkeypatch.setattr(core, "llm", original)
    monkeypatch.delitem(sys.modules, "core.llm")

    warmup._import_provider_stack()

    assert "core.llm" in sys.modules


def test_after_the_warm_up_the_resume_store_imports_nothing_new(resume_rag_on, supabase_factory):
    """ResumeStore() lazily imports core.llm; once warm, constructing it adds no module."""
    warmup.warm_request_dependencies()
    from agents.xbuddy.resume.store import ResumeStore

    before = set(sys.modules)
    ResumeStore()
    assert set(sys.modules) - before == set()


# --------------------------------------------------------------------------
# 4–5. No model, no embedding, no data
# --------------------------------------------------------------------------


def test_the_warm_up_calls_no_model_and_no_embedding(resume_rag_on, supabase_factory, monkeypatch):
    import core.llm

    def forbidden(*args, **kwargs):
        raise AssertionError("startup warm-up must not construct a model or embedding client")

    for name in ("get_model", "get_embeddings", "ChatOpenAI", "OpenAIEmbeddings", "AzureChatOpenAI"):
        monkeypatch.setattr(core.llm, name, forbidden)

    report = warmup.warm_request_dependencies()
    assert report.supabase_client == "ready"


def test_the_warm_up_reads_and_writes_no_data(resume_rag_on, supabase_factory, monkeypatch):
    _, client = supabase_factory
    from agents.xbuddy.resume.store import ResumeStore

    for method in ("status", "replace", "match"):
        monkeypatch.setattr(ResumeStore, method, Mock(side_effect=AssertionError(f"{method} called at startup")))

    warmup.warm_request_dependencies()

    assert client.calls == [], f"startup touched Supabase data: {client.calls}"


# --------------------------------------------------------------------------
# Failure semantics: required import fails loudly; optional Supabase does not
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_broken_provider_import_fails_startup(postgres_lifespan, monkeypatch):
    def broken():
        raise ImportError("provider stack unavailable")

    monkeypatch.setattr(warmup, "_import_provider_stack", broken)

    with pytest.raises(ImportError, match="provider stack unavailable"):
        async with svc.lifespan(svc.app):
            pass
    svc.pg_manager.setup.assert_not_awaited()


def test_unconfigured_resume_rag_builds_no_client(resume_rag_off, supabase_factory):
    created, _ = supabase_factory

    report = warmup.warm_request_dependencies()

    assert report.supabase_client == "not_configured"
    assert created == []


@pytest.mark.asyncio
async def test_a_supabase_client_failure_does_not_block_startup(
    postgres_lifespan, resume_rag_on, supabase_factory, monkeypatch, caplog
):
    monkeypatch.setattr(sb, "create_client", Mock(side_effect=RuntimeError("secret-looking-detail")))

    with caplog.at_level(logging.WARNING, logger="service.warmup"):
        async with svc.lifespan(svc.app):
            pass  # startup completed and the app served

    assert warmup.warm_request_dependencies().supabase_client == "failed"
    assert "RuntimeError" in caplog.text
    assert "secret-looking-detail" not in caplog.text


# --------------------------------------------------------------------------
# 7. Health is not faked or special-cased
# --------------------------------------------------------------------------


def _identifiers(source: str) -> set[str]:
    """Names the code actually uses — not docstrings or comments."""
    tree = ast.parse(textwrap.dedent(source))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.alias, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def test_health_does_not_depend_on_the_warm_up():
    """Neither side refers to the other in code (the warm-up's docstring may
    describe the health probe it protects; that is prose, not a dependency)."""
    assert not any("warm" in name.lower() for name in _identifiers(inspect.getsource(svc.health_check)))
    assert not any("health" in name.lower() for name in _identifiers(inspect.getsource(warmup)))


def test_health_answers_the_same_without_running_the_lifespan():
    # No context manager: the lifespan (and so the warm-up) never runs.
    response = TestClient(svc.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
