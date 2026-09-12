"""PR 6 Stage 4: the production security boundary.

Three small controls, all offline:

* a deployed process refuses to start without `AUTH_SECRET`
* CORS is an explicit allowlist, never `*`
* the two expensive LLM entrypoints are rate limited by client IP

Deliberately small. There is no per-user identity here — authorization is one
shared bearer token, and the rate limit is cost protection rather than a quota.
"""

import re
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from core.settings import Settings
from service import app
from service.service import client_ip_key, limiter

SERVICE_SRC = Path(__file__).resolve().parents[2] / "src" / "service" / "service.py"
API_KEY = {"OPENAI_API_KEY": SecretStr("sk-test"), "_env_file": None}


# ==========================================================================
# A. Production auth safety
# ==========================================================================


def test_local_mode_without_auth_secret_still_starts():
    """Developer convenience is preserved: MODE unset means not production."""
    settings = Settings(**API_KEY)
    assert settings.AUTH_SECRET is None
    assert settings.is_production() is False


def test_dev_mode_without_auth_secret_still_starts():
    settings = Settings(MODE="dev", **API_KEY)
    assert settings.is_dev() is True
    assert settings.is_production() is False


@pytest.mark.parametrize("mode", ["prod", "production", "PRODUCTION", " Prod "])
def test_production_without_auth_secret_fails_configuration(mode):
    """Fail fast rather than serve an unprotected API."""
    with pytest.raises(ValueError, match="AUTH_SECRET is required"):
        Settings(MODE=mode, **API_KEY)


def test_production_with_auth_secret_starts():
    settings = Settings(MODE="production", AUTH_SECRET=SecretStr("s3cret"), **API_KEY)
    assert settings.is_production() is True


def test_production_is_opt_in_not_the_absence_of_dev():
    """MODE is unset by default, so "not dev" would make every local run fail."""
    for mode in (None, "", "staging", "test"):
        assert Settings(MODE=mode, **API_KEY).is_production() is False


# ==========================================================================
# B. Bearer authentication
# ==========================================================================


@pytest.fixture
def invoke_agent():
    """An agent whose /invoke path completes, so auth is what decides the status."""
    agent = Mock()
    agent.ainvoke = AsyncMock(
        return_value=[("values", {"messages": [AIMessage(content="hi")]})]
    )

    async def aget_state(config=None):
        snapshot = Mock()
        snapshot.values = {}
        return snapshot

    agent.aget_state = aget_state
    return agent


def call_invoke(invoke_agent, headers=None, secret=None):
    with (
        patch("service.service.get_agent", Mock(return_value=invoke_agent)),
        patch("service.service.settings") as mocked,
    ):
        mocked.AUTH_SECRET = SecretStr(secret) if secret else None
        mocked.USE_SUPABASE_REALTIME = False
        return TestClient(app).post(
            "/invoke", json={"message": "hi", "user_id": 1}, headers=headers or {}
        )


def test_no_token_is_401_when_auth_is_enabled(invoke_agent):
    assert call_invoke(invoke_agent, secret="right").status_code == 401


def test_wrong_token_is_401(invoke_agent):
    response = call_invoke(
        invoke_agent, headers={"Authorization": "Bearer wrong"}, secret="right"
    )
    assert response.status_code == 401


def test_correct_token_is_allowed(invoke_agent):
    response = call_invoke(
        invoke_agent, headers={"Authorization": "Bearer right"}, secret="right"
    )
    assert response.status_code == 200


def test_no_auth_secret_allows_everything(invoke_agent):
    """The local bypass, unchanged."""
    assert call_invoke(invoke_agent).status_code == 200


def test_the_secret_is_compared_in_constant_time():
    """`!=` on a secret leaks length and prefix through timing."""
    source = SERVICE_SRC.read_text(encoding="utf-8")
    assert "secrets.compare_digest(http_auth.credentials, auth_secret)" in source
    assert "http_auth.credentials != auth_secret" not in source


def test_a_rejected_credential_is_never_echoed(invoke_agent):
    response = call_invoke(
        invoke_agent,
        headers={"Authorization": "Bearer super-secret-value"},
        secret="right",
    )
    assert response.status_code == 401
    assert "super-secret-value" not in response.text


def test_the_logging_middleware_masks_the_authorization_header():
    source = SERVICE_SRC.read_text(encoding="utf-8")
    assert "'authorization'" in source.lower()


def test_health_remains_public():
    """No token, and it must still answer."""
    response = TestClient(app).get("/health")
    assert response.status_code != 401


# ==========================================================================
# C. CORS
# ==========================================================================


def test_the_wildcard_origin_is_gone():
    source = SERVICE_SRC.read_text(encoding="utf-8")
    assert 'allow_origins=["*"]' not in source
    assert "allow_origins=settings.cors_allow_origins" in source


def test_credentials_are_disabled():
    """No cookie or browser-credential flow exists; the token is attached
    server-side by the frontend's own route handlers."""
    source = SERVICE_SRC.read_text(encoding="utf-8")
    assert "allow_credentials=False" in source
    assert "allow_credentials=True" not in source


def test_the_localhost_development_origin_is_the_default_outside_production():
    assert Settings(**API_KEY).cors_allow_origins == ["http://localhost:3000"]


def test_production_falls_back_to_no_origin_at_all():
    """PR 6 Stage 5B. The dev default must not follow the service into production.

    `env_ignore_empty=True` means a deployment cannot clear this by setting
    `CORS_ALLOW_ORIGINS=""` — an empty env value reads as unset. So the guarantee has
    to come from the mode-aware default, or a deployed API would advertise
    `http://localhost:3000` as an allowed origin while its config said otherwise.
    """
    production = Settings(MODE="production", AUTH_SECRET=SecretStr("t"), **API_KEY)
    assert production.cors_allow_origins == []


def test_origins_come_from_the_environment_as_a_comma_separated_list():
    settings = Settings(
        CORS_ALLOW_ORIGINS="http://localhost:3000, https://app.example.com", **API_KEY
    )
    assert settings.cors_allow_origins == [
        "http://localhost:3000",
        "https://app.example.com",
    ]


def test_blank_entries_are_dropped_and_empty_never_becomes_wildcard():
    """Whatever the mode, an unusable setting resolves to a concrete allowlist and
    never to `["*"]`. In production that allowlist is empty; locally it falls back to
    the dev server, which is a convenience and not a security boundary."""
    production = {"MODE": "production", "AUTH_SECRET": SecretStr("t")}
    assert Settings(CORS_ALLOW_ORIGINS="  ,  ", **production, **API_KEY).cors_allow_origins == []
    assert Settings(CORS_ALLOW_ORIGINS="", **production, **API_KEY).cors_allow_origins == []
    assert "*" not in Settings(CORS_ALLOW_ORIGINS="  ,  ", **API_KEY).cors_allow_origins


def test_no_deployment_url_is_hardcoded():
    source = SERVICE_SRC.read_text(encoding="utf-8")
    assert "vercel.app" not in source.replace(
        "# frontend reaches this API through its own Next.js route handlers", ""
    )


def cors_app(origins: list[str]) -> TestClient:
    """A minimal app wired the way service.py wires CORS, for header assertions."""
    from fastapi.middleware.cors import CORSMiddleware

    isolated = FastAPI()
    isolated.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @isolated.get("/probe")
    def probe():
        return {"ok": True}

    return TestClient(isolated)


def test_a_configured_origin_receives_allow_origin():
    client = cors_app(["http://localhost:3000"])
    response = client.get("/probe", headers={"Origin": "http://localhost:3000"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_an_unconfigured_origin_receives_no_allow_origin():
    client = cors_app(["http://localhost:3000"])
    response = client.get("/probe", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in response.headers


def test_credentials_header_is_not_advertised():
    client = cors_app(["http://localhost:3000"])
    response = client.get("/probe", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-credentials" not in response.headers


# ==========================================================================
# D. Rate limiting — the key function
# ==========================================================================


def fake_request(headers: dict | None = None, host: str | None = "127.0.0.1") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/invoke",
        "headers": [
            (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
        ],
        "client": (host, 12345) if host else None,
    }
    return Request(scope)


def test_fly_client_ip_is_preferred_when_present():
    request = fake_request({"Fly-Client-IP": "203.0.113.7"}, host="10.0.0.1")
    assert client_ip_key(request) == "203.0.113.7"


def test_local_fallback_uses_the_direct_peer():
    assert client_ip_key(fake_request(host="127.0.0.1")) == "127.0.0.1"


def test_x_forwarded_for_alone_is_not_trusted():
    """Any client can send it; trusting it would let one caller mint a fresh
    bucket per request and bypass the limit entirely."""
    request = fake_request({"X-Forwarded-For": "1.2.3.4"}, host="127.0.0.1")
    assert client_ip_key(request) == "127.0.0.1"


def test_x_forwarded_for_cannot_override_fly_client_ip():
    request = fake_request(
        {"Fly-Client-IP": "203.0.113.7", "X-Forwarded-For": "1.2.3.4"}, host="10.0.0.1"
    )
    assert client_ip_key(request) == "203.0.113.7"


def test_a_blank_fly_header_falls_back():
    request = fake_request({"Fly-Client-IP": "   "}, host="127.0.0.1")
    assert client_ip_key(request) == "127.0.0.1"


def test_a_missing_client_is_handled():
    assert client_ip_key(fake_request(host=None)) == "unknown"


def test_the_key_function_never_reads_x_forwarded_for():
    source = SERVICE_SRC.read_text(encoding="utf-8")
    key_fn = source[source.index("def client_ip_key") : source.index("limiter = Limiter")]
    assert "X-Forwarded-For" not in key_fn.replace(
        "`X-Forwarded-For` is **never** consulted", ""
    )


# ==========================================================================
# D. Rate limiting — the routes
# ==========================================================================


def limited_client(invoke_agent, monkeypatch, limit: str = "2/minute"):
    """Rebind the route limits to a small value and reset the limiter's storage."""
    limiter.reset()
    monkeypatch.setattr(limiter, "_default_limits", [], raising=False)
    return invoke_agent


@pytest.fixture(autouse=True)
def _clear_limiter():
    limiter.reset()
    yield
    limiter.reset()


def hammer(path: str, agent, body: dict, times: int, ip: str = "198.51.100.5"):
    statuses = []
    with (
        patch("service.service.get_agent", Mock(return_value=agent)),
        patch("service.service.settings") as mocked,
    ):
        mocked.AUTH_SECRET = None
        mocked.USE_SUPABASE_REALTIME = False
        client = TestClient(app)
        for _ in range(times):
            response = client.post(path, json=body, headers={"Fly-Client-IP": ip})
            statuses.append(response.status_code)
    return statuses


def test_invoke_is_rate_limited(invoke_agent):
    """The configured default is 10/minute; the 11th call from one IP is refused."""
    statuses = hammer("/invoke", invoke_agent, {"message": "hi", "user_id": 1}, times=12)
    assert 429 in statuses, statuses
    assert statuses.count(200) <= 10


def test_stream_is_rate_limited(invoke_agent):
    async def astream(**kwargs):
        yield ("updates", {"generate_reply": {"messages": [AIMessage(content="x")]}})

    invoke_agent.astream = astream
    statuses = hammer("/stream", invoke_agent, {"message": "hi", "user_id": 1}, times=12)
    assert 429 in statuses, statuses


def test_history_is_not_rate_limited(invoke_agent):
    """A cheap checkpoint read. Deliberately unthrottled in PR 6."""
    async def aget_state(config=None):
        snapshot = Mock()
        snapshot.values = {"user_id": 1, "messages": []}
        return snapshot

    invoke_agent.aget_state = aget_state
    statuses = hammer(
        "/history", invoke_agent, {"thread_id": "t-1", "user_id": 1}, times=25
    )
    assert 429 not in statuses, statuses


def test_separate_client_ips_get_separate_buckets(invoke_agent):
    first = hammer("/invoke", invoke_agent, {"message": "hi", "user_id": 1}, times=11,
                   ip="198.51.100.1")
    assert 429 in first
    second = hammer("/invoke", invoke_agent, {"message": "hi", "user_id": 1}, times=1,
                    ip="198.51.100.2")
    assert second == [200], second


def test_the_limit_is_applied_before_any_agent_work(invoke_agent):
    """429 must cost nothing. The agent is never invoked for a refused request."""
    with (
        patch("service.service.get_agent", Mock(return_value=invoke_agent)) as resolver,
        patch("service.service.settings") as mocked,
    ):
        mocked.AUTH_SECRET = None
        mocked.USE_SUPABASE_REALTIME = False
        client = TestClient(app)
        body = {"message": "hi", "user_id": 1}
        headers = {"Fly-Client-IP": "198.51.100.9"}
        for _ in range(10):
            client.post("/invoke", json=body, headers=headers)
        calls_before = resolver.call_count
        invocations_before = invoke_agent.ainvoke.await_count

        refused = client.post("/invoke", json=body, headers=headers)

    assert refused.status_code == 429
    assert resolver.call_count == calls_before, "the agent was resolved for a refused request"
    assert invoke_agent.ainvoke.await_count == invocations_before, "the model was invoked"


# ==========================================================================
# D. Rate limiting — configuration
# ==========================================================================


def test_the_default_limit_is_conservative():
    assert Settings(**API_KEY).RATE_LIMIT_EXPENSIVE == "10/minute"


def test_the_limit_is_environment_overridable():
    assert Settings(RATE_LIMIT_EXPENSIVE="3/minute", **API_KEY).RATE_LIMIT_EXPENSIVE == "3/minute"


def test_both_expensive_route_forms_carry_the_limit():
    """Prefixed and bare forms share one decorated handler, so both are covered."""
    source = SERVICE_SRC.read_text(encoding="utf-8")
    decorated = re.findall(r"@limiter\.limit\(settings\.RATE_LIMIT_EXPENSIVE\)", source)
    # /resume joined in Phase 8: one upload makes a structured-extraction model call
    # and an embedding call, so it is as expensive as a chat turn.
    assert len(decorated) == 3, "expected exactly /invoke, /stream and /resume to be limited"
    resume_block = source[source.index('@router.post("/{agent_id}/resume")') :][:300]
    assert "@limiter.limit(settings.RATE_LIMIT_EXPENSIVE)" in resume_block
    status_block = source[source.index('@router.post("/{agent_id}/resume/status")') :][:300]
    assert "@limiter.limit" not in status_block

    # The /{agent_id}/stream decorator is written across several lines, so anchor
    # on the path literal rather than a single-line decorator string.
    invoke_block = source[source.index('@router.post("/{agent_id}/invoke")') :][:700]
    stream_block = source[source.index('"/{agent_id}/stream"') :][:1200]
    assert "@limiter.limit(settings.RATE_LIMIT_EXPENSIVE)" in invoke_block
    assert "@limiter.limit(settings.RATE_LIMIT_EXPENSIVE)" in stream_block
    assert "async def invoke(" in invoke_block
    assert "async def stream(" in stream_block


def test_history_carries_no_limit_decorator():
    source = SERVICE_SRC.read_text(encoding="utf-8")
    history_block = source[source.index('@router.post("/{agent_id}/history")') :][:500]
    assert "@limiter.limit" not in history_block
