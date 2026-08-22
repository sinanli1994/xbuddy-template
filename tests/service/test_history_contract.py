"""PR 6 Stage 3: the /history contract.

Agent-aware, async-correct, and scoped to the requesting user. Offline throughout:
the checkpointer is faked, and the two fakes deliberately mimic the two savers this
service actually configures.

What this endpoint is *not*: a state dump. It returns the transcript and the two
identifiers, and nothing else — progress and completion belong to `/invoke`'s
`CompletionState` projection.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from schema import ChatHistory, ChatHistoryInput
from service import app
from service.service import load_chat_history

THREAD = "7bcc7cc1-99d7-4b1d-bdb5-e6f90ed44de6"
OWNER = 7
INTRUDER = 99

TRANSCRIPT = [
    HumanMessage(content="I need a new job"),
    AIMessage(content="What role are you targeting?"),
    HumanMessage(content="Senior SRE"),
    AIMessage(content="Good. When would you like to move?"),
]

# Everything a completed JobBuddy thread holds that must never reach the response.
NOISY_STATE = {
    "user_id": OWNER,
    "messages": TRANSCRIPT,
    "user_data": {"salary_expectation": "80k EUR", "target_roles": ["Senior SRE"]},
    "section_states": {"career_goal": {"status": "done", "content": "draft"}},
    "final_output": "# Your job search strategy\n\n## Your Action Plan\n",
    "finished": True,
    "should_generate_final_output": True,
    "router_directive": "next",
    "persistence_pending": ["career_goal"],
    "final_output_pending": "write",
    "short_memory": ["internal window"],
    "last_error": "something internal",
    "context_packet": {"system_prompt": "internal prompt"},
    "agent_output": {"reply": "internal"},
}


class AsyncSqliteStyleAgent:
    """A saver that implements only the async API.

    `AsyncSqliteSaver` is what runs locally. Calling the synchronous `get_state`
    against it is the bug Stage 3 fixed, so this fake raises if anyone tries.
    """

    def __init__(self, values: dict):
        self._values = values
        self.aget_state_calls: list[dict] = []

    async def aget_state(self, config):
        self.aget_state_calls.append(dict(config.get("configurable", {})))
        snapshot = Mock()
        snapshot.values = self._values
        return snapshot

    def get_state(self, config):  # pragma: no cover - must never be called
        raise AssertionError("sync get_state is not supported by an async saver")


class AsyncPostgresStyleAgent(AsyncSqliteStyleAgent):
    """Production's saver. Same public surface, so the same code path must serve it.

    Modelled as a subclass on purpose: if `load_chat_history` ever grew a
    saver-specific branch, one of these two fakes would stop working.
    """


def call(agent, thread_id=THREAD, user_id=OWNER, agent_id="xbuddy"):
    """Invoke the shared implementation directly, with `get_agent` patched."""
    import asyncio

    with patch("service.service.get_agent", Mock(return_value=agent)):
        return asyncio.run(load_chat_history(agent_id, thread_id, user_id))


def post(agent, body: dict, path: str = "/history"):
    with patch("service.service.get_agent", Mock(return_value=agent)):
        return TestClient(app).post(path, json=body)


# --------------------------------------------------------------------------
# Request/response contract
# --------------------------------------------------------------------------


def test_the_request_requires_both_identifiers():
    assert set(ChatHistoryInput.model_fields) == {"thread_id", "user_id"}
    for field in ChatHistoryInput.model_fields.values():
        assert field.is_required()


def test_a_request_without_user_id_is_rejected():
    response = post(AsyncSqliteStyleAgent(NOISY_STATE), {"thread_id": THREAD})
    assert response.status_code == 422


def test_the_response_carries_only_the_transcript_and_identifiers():
    assert set(ChatHistory.model_fields) == {"thread_id", "user_id", "messages"}


# --------------------------------------------------------------------------
# Route forms and agent resolution
# --------------------------------------------------------------------------


def test_the_default_agent_route_resolves_xbuddy():
    agent = AsyncSqliteStyleAgent(NOISY_STATE)
    with patch("service.service.get_agent", Mock(return_value=agent)) as resolver:
        response = TestClient(app).post(
            "/history", json={"thread_id": THREAD, "user_id": OWNER}
        )
    assert response.status_code == 200
    assert resolver.call_args.args[0] == "xbuddy"


def test_the_explicit_agent_route_passes_the_requested_agent_through():
    """Not DEFAULT_AGENT — the path parameter must reach the registry.

    Asserted with a **non-default** id on purpose. `DEFAULT_AGENT` is itself
    "xbuddy", so checking that the resolver received "xbuddy" cannot tell
    pass-through from a hardcoded default: substituting DEFAULT_AGENT in the
    implementation passed that version of this test.
    """
    agent = AsyncSqliteStyleAgent(NOISY_STATE)
    with patch("service.service.get_agent", Mock(return_value=agent)) as resolver:
        response = TestClient(app).post(
            "/some-other-agent/history", json={"thread_id": THREAD, "user_id": OWNER}
        )
    assert response.status_code == 200
    assert resolver.call_args.args[0] == "some-other-agent"


def test_the_shared_impl_resolves_whatever_agent_it_is_given():
    """Direct call, so nothing about routing can mask a hardcoded default.

    Does not go through the `call` helper: that helper installs its own `get_agent`
    patch, which shadows an outer one and leaves the observed Mock uncalled.
    """
    import asyncio

    agent = AsyncSqliteStyleAgent(NOISY_STATE)
    with patch("service.service.get_agent", Mock(return_value=agent)) as resolver:
        asyncio.run(load_chat_history("another-agent", THREAD, OWNER))
    assert resolver.call_args.args[0] == "another-agent"


def test_the_default_route_still_uses_the_default_agent():
    """The bare path must not require an agent id."""
    from agents import DEFAULT_AGENT

    agent = AsyncSqliteStyleAgent(NOISY_STATE)
    with patch("service.service.get_agent", Mock(return_value=agent)) as resolver:
        TestClient(app).post("/history", json={"thread_id": THREAD, "user_id": OWNER})
    assert resolver.call_args.args[0] == DEFAULT_AGENT


def test_an_unknown_agent_is_a_404():
    """`get_agent` raises KeyError off the registry dict; the endpoint translates it."""
    with patch("service.service.get_agent", Mock(side_effect=KeyError("nope"))):
        response = TestClient(app).post(
            "/not-a-real-agent/history", json={"thread_id": THREAD, "user_id": OWNER}
        )
    assert response.status_code == 404
    assert "Unknown agent" in response.json()["detail"]


def test_the_shared_implementation_takes_the_agent_as_a_parameter():
    """A hardcoded DEFAULT_AGENT inside the shared function would make the explicit
    route a lie. The signature is the guard."""
    import inspect

    assert "agent_id" in inspect.signature(load_chat_history).parameters


# --------------------------------------------------------------------------
# Async state access
# --------------------------------------------------------------------------


def test_state_is_read_through_aget_state():
    agent = AsyncSqliteStyleAgent(NOISY_STATE)
    call(agent)
    assert len(agent.aget_state_calls) == 1


def test_the_thread_configuration_matches_invoke_and_stream():
    agent = AsyncSqliteStyleAgent(NOISY_STATE)
    call(agent)
    configurable = agent.aget_state_calls[0]
    assert configurable == {"thread_id": THREAD, "user_id": OWNER}


def test_the_sqlite_style_saver_is_served():
    result = call(AsyncSqliteStyleAgent(NOISY_STATE))
    assert [message.content for message in result.messages] == [
        message.content for message in TRANSCRIPT
    ]


def test_the_postgres_style_saver_uses_the_same_public_path():
    """Production runs DATABASE_TYPE=postgres; the endpoint must not care."""
    result = call(AsyncPostgresStyleAgent(NOISY_STATE))
    assert [message.content for message in result.messages] == [
        message.content for message in TRANSCRIPT
    ]


def test_a_saver_failure_becomes_a_500_without_leaking_details():
    class Exploding(AsyncSqliteStyleAgent):
        async def aget_state(self, config):
            raise RuntimeError("connection string: postgres://user:secret@host/db")

    with patch("service.service.get_agent", Mock(return_value=Exploding({}))):
        response = TestClient(app).post(
            "/history", json={"thread_id": THREAD, "user_id": OWNER}
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Unexpected error"
    assert "secret" not in response.text


# --------------------------------------------------------------------------
# User scoping
# --------------------------------------------------------------------------


def test_the_owner_receives_the_transcript_in_order():
    result = call(AsyncSqliteStyleAgent(NOISY_STATE), user_id=OWNER)
    assert [(m.type, m.content) for m in result.messages] == [
        ("human", "I need a new job"),
        ("ai", "What role are you targeting?"),
        ("human", "Senior SRE"),
        ("ai", "Good. When would you like to move?"),
    ]


def test_a_different_user_cannot_read_the_thread():
    """404 rather than 403: 403 would confirm the thread exists."""
    with pytest.raises(Exception) as excinfo:
        call(AsyncSqliteStyleAgent(NOISY_STATE), user_id=INTRUDER)
    assert getattr(excinfo.value, "status_code", None) == 404


def test_the_intruder_response_contains_no_transcript():
    response = post(
        AsyncSqliteStyleAgent(NOISY_STATE),
        {"thread_id": THREAD, "user_id": INTRUDER},
    )
    assert response.status_code == 404
    for message in TRANSCRIPT:
        assert message.content not in response.text


def test_a_checkpoint_with_no_stored_user_id_is_denied():
    """Deny by default: ownership cannot be established, so nothing is shown.

    initialize_node always writes user_id, so this only fires for a checkpoint
    written by something else.
    """
    state = {key: value for key, value in NOISY_STATE.items() if key != "user_id"}
    response = post(AsyncSqliteStyleAgent(state), {"thread_id": THREAD, "user_id": OWNER})
    assert response.status_code == 404


def test_a_string_user_id_in_state_does_not_match_an_int_request():
    """Strict equality, no coercion — a type mismatch is not an ownership proof."""
    state = {**NOISY_STATE, "user_id": str(OWNER)}
    response = post(AsyncSqliteStyleAgent(state), {"thread_id": THREAD, "user_id": OWNER})
    assert response.status_code == 404


# --------------------------------------------------------------------------
# Empty / nonexistent threads
# --------------------------------------------------------------------------


def test_an_unknown_thread_returns_an_empty_transcript():
    """Preserves the previous behaviour rather than inventing a 404 for new threads."""
    result = call(AsyncSqliteStyleAgent({}))
    assert result.messages == []
    assert result.thread_id == THREAD
    assert result.user_id == OWNER


def test_a_thread_with_state_but_no_messages_returns_an_empty_list():
    result = call(AsyncSqliteStyleAgent({"user_id": OWNER}))
    assert result.messages == []


def test_a_none_values_snapshot_is_handled():
    class NullState(AsyncSqliteStyleAgent):
        async def aget_state(self, config):
            snapshot = Mock()
            snapshot.values = None
            return snapshot

    assert call(NullState({})).messages == []


# --------------------------------------------------------------------------
# No internal state may leak
# --------------------------------------------------------------------------


def test_no_graph_state_field_reaches_the_response():
    response = post(AsyncSqliteStyleAgent(NOISY_STATE), {"thread_id": THREAD, "user_id": OWNER})
    assert response.status_code == 200
    body = response.text

    for leaked in (
        "salary_expectation",
        "section_states",
        "final_output",
        "finished",
        "should_generate_final_output",
        "router_directive",
        "persistence_pending",
        "final_output_pending",
        "short_memory",
        "last_error",
        "context_packet",
        "agent_output",
        "system_prompt",
        "Your job search strategy",
    ):
        assert leaked not in body, f"{leaked} leaked into /history"


def test_the_response_json_has_exactly_three_top_level_keys():
    response = post(AsyncSqliteStyleAgent(NOISY_STATE), {"thread_id": THREAD, "user_id": OWNER})
    assert set(response.json()) == {"thread_id", "user_id", "messages"}


def test_messages_are_public_chat_messages_only():
    """No tool internals, no additional_kwargs dumps."""
    response = post(AsyncSqliteStyleAgent(NOISY_STATE), {"thread_id": THREAD, "user_id": OWNER})
    for message in response.json()["messages"]:
        assert set(message) <= {
            "type",
            "content",
            "tool_calls",
            "tool_call_id",
            "run_id",
            "response_metadata",
            "custom_data",
        }
        assert message["type"] in {"human", "ai", "tool", "custom"}
