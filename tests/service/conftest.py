from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import SecretStr

import service.service as service_module
from service import app

# The secret the `auth_secret` fixture installs. A fixed value, so a test can send it.
TEST_AUTH_SECRET = "test-auth-secret"


@pytest.fixture(autouse=True)
def deterministic_auth(request, monkeypatch):
    """Ordinary service tests must not depend on the developer's `.env`.

    `core/settings.py` builds `settings = Settings()` at import time with
    `env_file=find_dotenv()`, and `service.service` binds that one instance.
    `verify_bearer` reads `settings.AUTH_SECRET` on **every request**, so the moment a
    real `AUTH_SECRET` appeared in `.env` for the Fly deployment, eighteen tests that
    send no bearer token started getting 401 — nothing in the application had changed.

    So this pins the authentication state of the object production code actually reads,
    rather than mutating `os.environ` (which the already-constructed singleton would
    never re-read) or reloading the module (which would break every `patch(
    "service.service...")` in the suite by swapping the objects underneath them).

    Tests that mean to exercise authentication opt out by requesting `auth_secret`, and
    the ones that patch `service.service.settings` wholesale — `mock_settings`, and
    `test_security_boundary`'s `invoke_agent` — are unaffected either way, because they
    replace the object this fixture touched.

    The `auth_secret` early-return below is insurance rather than the mechanism: pytest
    runs autouse fixtures before explicitly-requested ones of the same scope, so
    `auth_secret` would win on ordering regardless. A teeth check confirmed that
    removing the branch changes no result. It stays because relying on implicit
    ordering is exactly the kind of thing that breaks silently if either fixture's
    scope changes later.
    """
    if "auth_secret" in request.fixturenames:
        return
    monkeypatch.setattr(service_module.settings, "AUTH_SECRET", None)


@pytest.fixture
def auth_secret(monkeypatch) -> str:
    """Turn authentication **on** with a known secret, and return it.

    Requesting this suppresses `deterministic_auth`, so a test that wants a live
    401/200 boundary gets one it configured itself instead of inheriting whatever
    happens to be in `.env`.
    """
    monkeypatch.setattr(service_module.settings, "AUTH_SECRET", SecretStr(TEST_AUTH_SECRET))
    return TEST_AUTH_SECRET


@pytest.fixture
def test_client():
    """Fixture to create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_agent():
    """Fixture to create a mock agent that can be configured for different test scenarios."""
    agent_mock = AsyncMock()
    agent_mock.ainvoke = AsyncMock(
        return_value=[("values", {"messages": [AIMessage(content="Test response")]})]
    )
    agent_mock.get_state = Mock()  # Default empty mock for get_state

    # `/invoke` reads the post-run snapshot to build its section metadata and the
    # public completion projection, so `aget_state` has to return something whose
    # `.values` is a real mapping. Left unconfigured, AsyncMock hands back an
    # un-awaited coroutine and any `.get()` on it raises — which surfaced as a 500
    # from every auth test the moment /invoke started calling `.get()`.
    async def _aget_state(config=None):
        snapshot = Mock()
        snapshot.values = {}
        return snapshot

    agent_mock.aget_state = _aget_state
    with patch("service.service.get_agent", Mock(return_value=agent_mock)):
        yield agent_mock


@pytest.fixture
def mock_settings(mock_env):
    """Fixture to ensure settings are clean for each test."""
    with patch("service.service.settings") as mock_settings:
        yield mock_settings


@pytest.fixture
def mock_httpx():
    """Patch httpx.stream and httpx.get to use our test client."""

    with TestClient(app) as client:

        def mock_stream(method: str, url: str, **kwargs):
            # Strip the base URL since TestClient expects just the path
            path = url.replace("http://0.0.0.0", "")
            return client.stream(method, path, **kwargs)

        def mock_get(url: str, **kwargs):
            # Strip the base URL since TestClient expects just the path
            path = url.replace("http://0.0.0.0", "")
            return client.get(path, **kwargs)

        with patch("httpx.stream", mock_stream), patch("httpx.get", mock_get):
            yield
