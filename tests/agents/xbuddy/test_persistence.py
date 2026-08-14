"""Stage 4A tests: Supabase key resolution, write contract, and content rendering.

Everything here runs against fakes — no live project is contacted. The write
contract is asserted at the payload level, because the two defects this stage
fixes (a missing `on_conflict` target and an `agent_id` that read paths filter
out) are both invisible unless you inspect what is actually sent.
"""

import logging

import pytest
from pydantic import SecretStr

from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import SectionState, XBuddyData
from agents.xbuddy.persistence import (
    AGENT_ID,
    persist_section,
    render_section_content,
)

FAKE_SECRET = "sb_secret_NEVER_LOG_THIS_VALUE_12345"
FAKE_LEGACY = "eyJlegacy_service_role_NEVER_LOG_67890"


class FakeTable:
    """Records the upsert payload and kwargs for assertion."""

    def __init__(self, store: dict, fail: Exception | None = None):
        self.store = store
        self.fail = fail

    def upsert(self, payload, **kwargs):
        if self.fail is not None:
            raise self.fail
        self.store["payloads"].append(payload)
        self.store["kwargs"].append(kwargs)
        # Emulate the unique constraint: one logical row per conflict target.
        key = (payload["user_id"], payload["thread_id"], payload["section_id"])
        self.store["rows"][key] = payload
        return self

    def execute(self):
        return type("Result", (), {"data": [self.store["payloads"][-1]]})()


class FakeSupabaseClient:
    """Stands in for integrations.supabase.SupabaseClient's real method."""

    def __init__(self, fail: Exception | None = None, construct_error: Exception | None = None):
        if construct_error is not None:
            raise construct_error
        self.calls: list[dict] = []
        self.store: dict = {"payloads": [], "kwargs": [], "rows": {}}
        self._fail = fail

    def save_section_state(self, **kwargs):
        """Mirrors the real signature, then delegates to the real upsert shape."""
        self.calls.append(kwargs)
        if self._fail is not None:
            return {"success": False, "error": str(self._fail)}
        table = FakeTable(self.store)
        table.upsert(
            {
                "user_id": kwargs["user_id"],
                "thread_id": kwargs["thread_id"],
                "agent_id": kwargs["agent_id"],
                "section_id": kwargs["section_id"],
                "content": kwargs["content"],
                "plain_text": kwargs["plain_text"],
                "status": kwargs["status"],
                "satisfaction_status": kwargs["satisfaction_status"],
                "updated_at": "now()",
            },
            on_conflict="user_id,thread_id,section_id",
        ).execute()
        return {"success": True, "data": [kwargs]}


@pytest.fixture
def fake_client(monkeypatch):
    from agents.xbuddy import persistence

    client = FakeSupabaseClient()
    monkeypatch.setattr(persistence, "_client", lambda: client)
    return client


def in_progress(section: SectionID = SectionID.CAREER_GOAL) -> SectionState:
    return SectionState(section_id=section, status=SectionStatus.IN_PROGRESS)


# --------------------------------------------------------------------------
# 1. Key resolution
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_client_singleton(monkeypatch):
    """get_supabase_client caches a module singleton; clear it between tests."""
    import integrations.supabase.supabase_client as module

    monkeypatch.setattr(module, "_supabase_client", None)


def test_new_secret_key_wins_over_legacy(monkeypatch):
    from core.settings import settings
    from integrations.supabase.supabase_client import _resolve_backend_key

    monkeypatch.setattr(settings, "SUPABASE_SECRET_KEY", SecretStr(FAKE_SECRET))
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", SecretStr(FAKE_LEGACY))

    value, source = _resolve_backend_key()

    assert value == FAKE_SECRET
    assert source == "SUPABASE_SECRET_KEY"


def test_legacy_service_role_key_is_the_fallback(monkeypatch):
    from core.settings import settings
    from integrations.supabase.supabase_client import _resolve_backend_key

    monkeypatch.setattr(settings, "SUPABASE_SECRET_KEY", None)
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", SecretStr(FAKE_LEGACY))

    value, source = _resolve_backend_key()

    assert value == FAKE_LEGACY
    assert source == "SUPABASE_SERVICE_ROLE_KEY"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_new_key_falls_through_to_legacy(monkeypatch, blank):
    """env_ignore_empty handles "" from .env, but a whitespace value must not win."""
    from core.settings import settings
    from integrations.supabase.supabase_client import _resolve_backend_key

    monkeypatch.setattr(settings, "SUPABASE_SECRET_KEY", SecretStr(blank))
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", SecretStr(FAKE_LEGACY))

    value, source = _resolve_backend_key()

    assert value == FAKE_LEGACY
    assert source == "SUPABASE_SERVICE_ROLE_KEY"


def test_missing_keys_resolve_to_none(monkeypatch):
    from core.settings import settings
    from integrations.supabase.supabase_client import _resolve_backend_key

    monkeypatch.setattr(settings, "SUPABASE_SECRET_KEY", None)
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", None)

    assert _resolve_backend_key() == (None, None)


def test_missing_credentials_fail_safely_with_a_helpful_message(monkeypatch):
    from core.settings import settings
    from integrations.supabase.supabase_client import get_supabase_client

    monkeypatch.setattr(settings, "SUPABASE_URL", None)
    monkeypatch.setattr(settings, "SUPABASE_SECRET_KEY", None)
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", None)

    with pytest.raises(ValueError, match="SUPABASE_SECRET_KEY"):
        get_supabase_client()


def test_new_key_fields_are_declared_on_settings():
    """extra="ignore" means an undeclared name in .env is silently dropped."""
    from core.settings import Settings

    for name in (
        "SUPABASE_SECRET_KEY",
        "SUPABASE_PUBLISHABLE_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_ANON_KEY",
    ):
        assert name in Settings.model_fields, f"{name} not declared; .env would ignore it"


def test_no_secret_value_is_logged(monkeypatch, caplog):
    """Only the variable *name* may appear in logs, never the key."""
    from core.settings import settings
    from integrations.supabase.supabase_client import get_supabase_client

    monkeypatch.setattr(settings, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(settings, "SUPABASE_SECRET_KEY", SecretStr(FAKE_SECRET))

    created: dict = {}
    monkeypatch.setattr(
        "integrations.supabase.supabase_client.create_client",
        lambda url, key: created.setdefault("key", key) or object(),
    )

    with caplog.at_level(logging.DEBUG):
        get_supabase_client()

    # The key did reach create_client...
    assert created["key"] == FAKE_SECRET
    # ...but never the log stream.
    assert FAKE_SECRET not in caplog.text
    assert "SUPABASE_SECRET_KEY" in caplog.text, "the source name should be logged"


def test_no_secret_value_in_the_credentials_error(monkeypatch):
    from core.settings import settings
    from integrations.supabase.supabase_client import get_supabase_client

    monkeypatch.setattr(settings, "SUPABASE_URL", None)
    monkeypatch.setattr(settings, "SUPABASE_SECRET_KEY", SecretStr(FAKE_SECRET))

    with pytest.raises(ValueError) as exc:
        get_supabase_client()

    assert FAKE_SECRET not in str(exc.value)


# --------------------------------------------------------------------------
# 2. render_section_content
# --------------------------------------------------------------------------


def test_empty_section_renders_empty_placeholders():
    """`{}` satisfies the column's JSONB NOT NULL without a migration."""
    content = render_section_content(SectionID.CAREER_GOAL, XBuddyData())

    assert content.content == {}
    assert content.plain_text == ""


def test_populated_section_renders_deterministically():
    data = XBuddyData(
        target_roles=["SRE", "Platform Engineer"],
        career_goal_summary="Move into platform work",
        target_timeline="3 months",
    )
    first = render_section_content(SectionID.CAREER_GOAL, data)
    second = render_section_content(SectionID.CAREER_GOAL, data)

    assert first == second, "rendering must be deterministic"
    assert first.plain_text == (
        "Target role(s): SRE, Platform Engineer\n"
        "Career goal: Move into platform work\n"
        "Timeline: 3 months"
    )
    assert first.content["type"] == "doc"
    assert len(first.content["content"]) == 3
    assert first.content["content"][0] == {
        "type": "paragraph",
        "content": [{"type": "text", "text": "Target role(s): SRE, Platform Engineer"}],
    }


def test_render_includes_only_the_sections_own_fields():
    """A Career Goal row must not carry Background or Skills data."""
    data = XBuddyData(
        target_roles=["SRE"], current_role="QA Analyst", strengths=["testing"]
    )
    content = render_section_content(SectionID.CAREER_GOAL, data)

    assert "SRE" in content.plain_text
    assert "QA Analyst" not in content.plain_text
    assert "testing" not in content.plain_text


def test_render_skips_unfilled_fields():
    content = render_section_content(
        SectionID.CAREER_GOAL, XBuddyData(target_timeline="3 months")
    )
    assert content.plain_text == "Timeline: 3 months"
    assert len(content.content["content"]) == 1


def test_render_rejects_an_unknown_section():
    with pytest.raises(ValueError, match="Unknown section_id"):
        render_section_content("not_a_section", XBuddyData())


# --------------------------------------------------------------------------
# 3. persist_section — payload contract
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_payload_shape_is_exact(fake_client):
    data = XBuddyData(target_roles=["SRE"], target_timeline="3 months")
    section = SectionState(
        section_id=SectionID.CAREER_GOAL,
        status=SectionStatus.DONE,
        satisfaction_status="satisfied",
    )

    assert await persist_section(7, "t-1", SectionID.CAREER_GOAL, section, data) is True

    call = fake_client.calls[-1]
    assert call["user_id"] == 7
    assert call["thread_id"] == "t-1"
    assert call["section_id"] == "career_goal"
    assert call["status"] == "done"
    assert call["satisfaction_status"] == "satisfied"
    assert call["plain_text"] == "Target role(s): SRE\nTimeline: 3 months"
    assert call["content"]["type"] == "doc"


@pytest.mark.asyncio
async def test_agent_id_is_xbuddy(fake_client):
    """The old "founder-buddy" default wrote rows the read paths filter out."""
    await persist_section(7, "t-1", SectionID.CAREER_GOAL, in_progress(), XBuddyData())

    assert fake_client.calls[-1]["agent_id"] == "xbuddy"
    assert AGENT_ID == "xbuddy"


@pytest.mark.asyncio
async def test_section_id_is_the_canonical_string(fake_client):
    """The column is TEXT and stores SectionID.value."""
    for section in SectionID:
        await persist_section(7, "t-1", section, in_progress(section), XBuddyData())

    sent = [call["section_id"] for call in fake_client.calls]
    assert sent == [section.value for section in SectionID]
    assert all(isinstance(value, str) for value in sent)


@pytest.mark.asyncio
async def test_a_string_section_id_is_accepted(fake_client):
    assert await persist_section(7, "t-1", "background", in_progress(), XBuddyData()) is True
    assert fake_client.calls[-1]["section_id"] == "background"


@pytest.mark.asyncio
async def test_empty_content_is_persisted_as_empty_placeholders(fake_client):
    await persist_section(7, "t-1", SectionID.CAREER_GOAL, in_progress(), XBuddyData())

    call = fake_client.calls[-1]
    assert call["content"] == {}
    assert call["plain_text"] == ""


class RecordingRoot:
    """Stubs the postgrest root so the REAL save_section_state can be exercised."""

    def __init__(self):
        self.table_name: str | None = None
        self.payload: dict | None = None
        self.kwargs: dict | None = None

    def table(self, name):
        self.table_name = name
        return self

    def upsert(self, payload, **kwargs):
        self.payload = payload
        self.kwargs = kwargs
        return self

    def execute(self):
        return type("Result", (), {"data": [self.payload]})()


def real_client_with_stub() -> tuple:
    """A real SupabaseClient with its postgrest root stubbed.

    `__new__` bypasses `__init__`, which would demand live credentials.
    """
    from integrations.supabase.supabase_client import SupabaseClient

    client = SupabaseClient.__new__(SupabaseClient)
    root = RecordingRoot()
    client.client = root
    return client, root


def test_real_save_section_state_names_the_exact_conflict_target():
    """Behavioural guard on the defect this stage fixes.

    Without `on_conflict` the target defaults to the UUID primary key; since no
    `id` is sent, a repeat write INSERTs and violates
    UNIQUE(user_id, thread_id, section_id).
    """
    client, root = real_client_with_stub()

    client.save_section_state(
        user_id=7,
        thread_id="t-1",
        section_id="career_goal",
        content={},
        plain_text="",
        status="in_progress",
    )

    assert root.table_name == "section_states"
    assert root.kwargs == {"on_conflict": "user_id,thread_id,section_id"}


def test_real_save_section_state_defaults_agent_id_to_xbuddy():
    """The old "founder-buddy" default wrote rows the read paths filter out."""
    client, root = real_client_with_stub()

    client.save_section_state(
        user_id=7,
        thread_id="t-1",
        section_id="career_goal",
        content={},
        plain_text="",
        status="in_progress",
    )

    assert root.payload["agent_id"] == "xbuddy"


def test_real_save_section_state_payload_columns_match_the_schema():
    """Every key must be a real column, or PostgREST rejects the whole write."""
    client, root = real_client_with_stub()

    client.save_section_state(
        user_id=7,
        thread_id="t-1",
        section_id="career_goal",
        content={"type": "doc"},
        plain_text="text",
        status="done",
        satisfaction_status="satisfied",
        agent_id="xbuddy",
    )

    assert set(root.payload) == {
        "user_id",
        "thread_id",
        "agent_id",
        "section_id",
        "content",
        "plain_text",
        "status",
        "satisfaction_status",
        "updated_at",
    }


def test_real_save_section_state_returns_failure_dict_on_error():
    """It swallows exceptions into a result dict; persist_section reads that."""
    from integrations.supabase.supabase_client import SupabaseClient

    class Boom:
        def table(self, name):
            raise RuntimeError("42P10 there is no unique constraint")

    client = SupabaseClient.__new__(SupabaseClient)
    client.client = Boom()

    result = client.save_section_state(
        user_id=7,
        thread_id="t-1",
        section_id="career_goal",
        content={},
        plain_text="",
        status="done",
    )

    assert result["success"] is False
    assert "42P10" in result["error"]


# --------------------------------------------------------------------------
# 4. Idempotency and failure
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writing_the_same_turn_twice_is_logically_idempotent(fake_client):
    data = XBuddyData(target_roles=["SRE"])
    section = SectionState(section_id=SectionID.CAREER_GOAL, status=SectionStatus.DONE)

    assert await persist_section(7, "t-1", SectionID.CAREER_GOAL, section, data) is True
    assert await persist_section(7, "t-1", SectionID.CAREER_GOAL, section, data) is True

    assert len(fake_client.calls) == 2, "both writes were attempted"
    # ...but the conflict target collapses them to one logical row.
    assert len(fake_client.store["rows"]) == 1
    assert fake_client.store["kwargs"][-1] == {
        "on_conflict": "user_id,thread_id,section_id"
    }


@pytest.mark.asyncio
async def test_different_sections_produce_different_rows(fake_client):
    await persist_section(7, "t-1", SectionID.CAREER_GOAL, in_progress(), XBuddyData())
    await persist_section(
        7, "t-1", SectionID.BACKGROUND, in_progress(SectionID.BACKGROUND), XBuddyData()
    )

    assert len(fake_client.store["rows"]) == 2


@pytest.mark.asyncio
async def test_sync_client_exception_returns_false(monkeypatch):
    from agents.xbuddy import persistence

    def boom():
        raise RuntimeError("connection reset")

    monkeypatch.setattr(persistence, "_client", boom)

    result = await persist_section(
        7, "t-1", SectionID.CAREER_GOAL, in_progress(), XBuddyData()
    )
    assert result is False


@pytest.mark.asyncio
async def test_missing_credentials_return_false_not_an_exception(monkeypatch):
    """The client factory raises ValueError when unconfigured — must degrade."""
    from agents.xbuddy import persistence

    def unconfigured():
        raise ValueError("Supabase credentials not configured.")

    monkeypatch.setattr(persistence, "_client", unconfigured)

    assert (
        await persist_section(7, "t-1", SectionID.CAREER_GOAL, in_progress(), XBuddyData())
        is False
    )


@pytest.mark.asyncio
async def test_unsuccessful_result_returns_false(monkeypatch):
    from agents.xbuddy import persistence

    client = FakeSupabaseClient(fail=RuntimeError("42P10 no unique constraint"))
    monkeypatch.setattr(persistence, "_client", lambda: client)

    assert (
        await persist_section(7, "t-1", SectionID.CAREER_GOAL, in_progress(), XBuddyData())
        is False
    )


@pytest.mark.asyncio
async def test_unknown_section_returns_false_without_touching_the_client(fake_client):
    assert (
        await persist_section(7, "t-1", "not_a_section", in_progress(), XBuddyData()) is False
    )
    assert fake_client.calls == []


@pytest.mark.asyncio
async def test_persist_section_does_not_itself_disclose_a_secret(monkeypatch, caplog):
    """What this module controls: the return value and its own log message.

    `persist_section` returns a bare bool, so no credential can travel back to a
    caller. Its formatted message names only the section and thread.

    The honest caveat: it uses `logger.exception`, so if an *upstream* library
    embeds a key in its exception text that text lands in the traceback. Supabase
    errors do not do this today, and dropping the traceback would cost real
    debuggability, so the trade is accepted rather than hidden — this test pins
    the half we own.
    """
    from agents.xbuddy import persistence

    def leaky():
        raise RuntimeError(f"auth failed for key {FAKE_SECRET}")

    monkeypatch.setattr(persistence, "_client", leaky)

    with caplog.at_level(logging.DEBUG):
        result = await persist_section(
            7, "t-1", SectionID.CAREER_GOAL, in_progress(), XBuddyData()
        )

    # A bool cannot carry credential text.
    assert result is False
    assert isinstance(result, bool)

    # Our own formatted messages mention neither the key nor anything sensitive.
    own_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "agents.xbuddy.persistence"
    ]
    assert own_messages, "expected persist_section to log the failure"
    for message in own_messages:
        assert FAKE_SECRET not in message
    assert any("career_goal" in message for message in own_messages)


# --------------------------------------------------------------------------
# 5. Stage boundary
# --------------------------------------------------------------------------


def test_persistence_is_wired_into_memory_updater():
    """Stage 5 wired the layer in; Stage 4A only shipped it.

    Behavioural coverage of the wiring lives in test_persistence_retry.py — this
    just pins that the node reaches for the layer at all, so the two files cannot
    drift into testing an unused module.
    """
    from pathlib import Path

    source = Path("src/agents/xbuddy/nodes/memory_updater.py").read_text(encoding="utf-8")
    assert "persist_section" in source
    assert "persistence_pending" in source


def test_stage_6_and_7_behaviour_is_absent():
    """Remaining PR 4 boundary: no restart-restore, no eval harness in the node."""
    from pathlib import Path

    source = Path("src/agents/xbuddy/nodes/memory_updater.py").read_text(encoding="utf-8")
    # Supabase is not a restore path: the checkpoint stays primary for a live thread.
    assert "get_section_states" not in source
    assert "aupdate_state" not in source


def test_persistence_module_does_not_import_a_node():
    """The layer stays callable from anywhere without dragging the graph in."""
    from pathlib import Path

    source = Path("src/agents/xbuddy/persistence.py").read_text(encoding="utf-8")
    assert "from .nodes" not in source
    assert "import nodes" not in source
