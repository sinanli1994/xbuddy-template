"""Checkpoint-restored values reach the public API as enums *or* as plain strings.

A live `/invoke` returned 500 with `'str' object has no attribute 'value'`. The graph
ran, the model was called, Postgres stored the checkpoint and `/history` read it back
fine — the exception was thrown afterwards, building the public response, on a bare
`.value` access against a value that came back from the checkpoint as a string.

The domain layer already treats both forms as normal: `router_node` does
`SectionID(state.get("current_section", ...))`, and `coerce_section_state`'s docstring
says outright that "deserialized checkpoints can hand back plain dicts". Only the
service boundary assumed the typed form.

Every service test missed it because `mock_agent.aget_state` returns `values = {}`, so
`if "current_section" in state.values` is False and the whole active-section block is
skipped. These tests supply a Postgres-style snapshot instead, in each representation
a restore can actually produce.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage

from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import SectionState

CANONICAL = ["career_goal", "background", "job_preferences", "skill_assessment", "action_plan"]

# The three shapes a checkpoint restore can hand back for one section entry.
#   typed      - the in-process form, what tests used to assume
#   string     - enum fields deserialized to their plain values
#   dict       - the pydantic model itself deserialized to a mapping
FORMS = ["typed", "string", "dict"]


def snapshot_values(form: str, status: str = "in_progress") -> dict:
    """A Postgres-style state snapshot in the requested representation."""
    if form == "typed":
        current = SectionID.CAREER_GOAL
        entry = SectionState(
            section_id=SectionID.CAREER_GOAL, status=SectionStatus(status)
        )
    elif form == "string":
        current = SectionID.CAREER_GOAL.value
        entry = SectionState(section_id=SectionID.CAREER_GOAL, status=SectionStatus(status))
        # the model survives but its enum field came back as a bare string
        object.__setattr__(entry, "status", status)
    elif form == "dict":
        current = SectionID.CAREER_GOAL.value
        entry = {"section_id": "career_goal", "status": status}
    else:  # pragma: no cover - guard against a typo in FORMS
        raise AssertionError(form)

    return {
        "current_section": current,
        "section_states": {"career_goal": entry},
        "should_generate_final_output": False,
        "final_output": None,
        "user_id": 999,
        "messages": [],
    }


@pytest.fixture(autouse=True)
def no_rate_limiting(monkeypatch):
    """These tests make far more than `RATE_LIMIT_EXPENSIVE` (10/minute) calls to
    /invoke, so without this the later parametrisations get a real 429 and fail for a
    reason that has nothing to do with enum normalization — while passing in isolation.

    Scoped to this file: the rate limiter stays live everywhere else.
    """
    import service.service as svc

    monkeypatch.setattr(svc.limiter, "enabled", False)


@pytest.fixture
def agent_with_state():
    """A mock agent whose `aget_state` returns a caller-supplied snapshot."""

    def build(values: dict):
        agent = AsyncMock()
        agent.ainvoke = AsyncMock(
            return_value=[("values", {"messages": [AIMessage(content="Test response")]})]
        )
        agent.get_state = Mock()

        async def _aget_state(config=None):
            snapshot = Mock()
            snapshot.values = values
            # `_handle_input` iterates `state.tasks` looking for interrupts before the
            # graph runs; a bare Mock is not iterable.
            snapshot.tasks = []
            return snapshot

        agent.aget_state = _aget_state
        return agent

    return build


# --------------------------------------------------------------------------
# /invoke - the endpoint that actually returned 500
# --------------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
def test_invoke_builds_section_metadata_for_every_restored_form(
    form, agent_with_state, test_client
):
    """The live failure: a bare `.value` on a string 500s here."""
    agent = agent_with_state(snapshot_values(form))
    with patch("service.service.get_agent", Mock(return_value=agent)):
        response = test_client.post(
            "/invoke", json={"message": "hi", "thread_id": "t-1", "user_id": 999}
        )

    assert response.status_code == 200, response.text
    section = response.json()["output"]["custom_data"]["section"]
    assert section["name"] == "Career Goal"
    assert section["status"] == "in_progress"
    assert section["database_id"] == 1


@pytest.mark.parametrize("form", FORMS)
def test_invoke_completion_projection_survives_every_form(form, agent_with_state, test_client):
    agent = agent_with_state(snapshot_values(form))
    with patch("service.service.get_agent", Mock(return_value=agent)):
        response = test_client.post(
            "/invoke", json={"message": "hi", "thread_id": "t-2", "user_id": 999}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [s["id"] for s in body["sections"]] == CANONICAL
    assert body["sections"][0]["status"] == "in_progress"
    assert all(set(s) == {"id", "name", "status"} for s in body["sections"])
    assert body["collection_complete"] is False
    assert body["artifact_available"] is False


@pytest.mark.parametrize("status", ["pending", "in_progress", "done"])
@pytest.mark.parametrize("form", FORMS)
def test_public_status_values_are_preserved_exactly(status, form, agent_with_state, test_client):
    """Normalization must not rename anything the frontend switches on."""
    agent = agent_with_state(snapshot_values(form, status=status))
    with patch("service.service.get_agent", Mock(return_value=agent)):
        response = test_client.post(
            "/invoke", json={"message": "hi", "thread_id": "t-3", "user_id": 999}
        )

    assert response.status_code == 200, response.text
    assert response.json()["output"]["custom_data"]["section"]["status"] == status
    assert response.json()["sections"][0]["status"] == status


# --------------------------------------------------------------------------
# /stream - the same block, where the bug is silent instead of loud
# --------------------------------------------------------------------------


def sse_events(raw: str) -> list[dict]:
    import json

    events = []
    for line in raw.splitlines():
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            events.append(json.loads(line[6:]))
    return events


@pytest.mark.parametrize("form", FORMS)
def test_stream_emits_the_section_event_for_every_restored_form(
    form, agent_with_state, test_client
):
    """`/stream`'s section block is wrapped in `except Exception: logger.error(...)`, so
    the same bug drops the event silently rather than failing — the frontend just stops
    getting section updates and nothing surfaces."""
    agent = agent_with_state(snapshot_values(form))

    async def _astream(*args, **kwargs):
        yield ("values", {"messages": [AIMessage(content="Test response")]})

    agent.astream = _astream

    with patch("service.service.get_agent", Mock(return_value=agent)):
        response = test_client.post(
            "/stream", json={"message": "hi", "thread_id": "t-4", "user_id": 999}
        )

    assert response.status_code == 200, response.text
    sections = [e for e in sse_events(response.text) if e.get("type") == "section"]
    assert sections, f"no section event emitted for the {form!r} form"
    assert sections[0]["content"]["name"] == "Career Goal"
    assert sections[0]["content"]["status"] == "in_progress"


# --------------------------------------------------------------------------
# The helper's own contract
# --------------------------------------------------------------------------


def test_enum_value_normalizes_both_representations():
    from service.service import enum_value

    assert enum_value(SectionID.CAREER_GOAL) == "career_goal"
    assert enum_value("career_goal") == "career_goal"
    assert enum_value(SectionStatus.DONE) == "done"
    assert enum_value("done") == "done"


def test_enum_value_refuses_arbitrary_objects():
    """A silent `str(value)` fallback would turn a real bug into a plausible-looking
    string like `<object object at 0x...>` and ship it to the frontend."""
    from service.service import enum_value

    for bad in (object(), 42, None, ["career_goal"]):
        with pytest.raises(TypeError):
            enum_value(bad)


def test_section_status_reads_every_entry_shape():
    from service.service import section_status

    assert section_status(None) == "pending"
    assert section_status({"status": "done"}) == "done"
    assert section_status({}) == "pending"
    assert (
        section_status(SectionState(section_id=SectionID.CAREER_GOAL, status=SectionStatus.DONE))
        == "done"
    )


def test_no_unguarded_value_access_remains_on_checkpoint_state():
    """Pins the fix. Both endpoints must go through the helper, so `/invoke` and
    `/stream` cannot diverge on the same representation issue again."""
    import inspect

    import service.service as svc

    source = inspect.getsource(svc)
    assert "current_section_enum.value" not in source
    assert "section_state.status.value" not in source
    assert source.count("enum_value(current_section_enum)") == 2, "one per endpoint"
    assert source.count("section_status(section_state)") == 2
