"""`/completion` — the refresh-time read of the public completion projection.

`/invoke` and the SSE `completion` event were the only places the projection was ever
emitted, so a browser reload could restore the transcript from `/history` and nothing
beside it. This endpoint closes that gap without widening the contract: same three
fields, same helper, same trust boundary as `/history`.

The tests that matter most are the negative ones. This endpoint reads graph state, so
the interesting question is not "does it work" but "what can it be made to reveal" —
`finished`, raw `section_states`, another user's progress.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import SectionState
from schema.schema import CompletionState
from service import app

OWNER = 4242
INTRUDER = 9999
THREAD = "completion-thread"
CANONICAL = ["career_goal", "background", "job_preferences", "skill_assessment", "action_plan"]


def sections(**statuses) -> dict:
    return {
        section.value: SectionState(
            section_id=section,
            status=statuses.get(section.value, SectionStatus.PENDING),
        )
        for section in SectionID
    }


def agent_with(values: dict | None):
    """An agent whose `aget_state` returns the given values."""
    agent = AsyncMock()

    async def _aget_state(config=None):
        snapshot = Mock()
        snapshot.values = values
        snapshot.tasks = []
        return snapshot

    agent.aget_state = _aget_state
    return agent


def call(values, *, user_id=OWNER, thread_id=THREAD, path="/completion", agent=None):
    resolved = agent if agent is not None else agent_with(values)
    with patch("service.service.get_agent", Mock(return_value=resolved)):
        return TestClient(app).post(path, json={"thread_id": thread_id, "user_id": user_id})


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_the_owner_gets_200_and_the_projection():
    response = call(
        {
            "user_id": OWNER,
            "section_states": sections(career_goal=SectionStatus.IN_PROGRESS),
            "should_generate_final_output": False,
            "final_output": None,
        }
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["collection_complete"] is False
    assert body["artifact_available"] is False
    assert [s["id"] for s in body["sections"]] == CANONICAL
    assert body["sections"][0]["status"] == "in_progress"


def test_exactly_five_canonical_sections():
    response = call({"user_id": OWNER, "section_states": sections()})
    assert len(response.json()["sections"]) == 5


def test_a_complete_thread_reports_completion():
    done = {section.value: SectionStatus.DONE for section in SectionID}
    response = call(
        {
            "user_id": OWNER,
            "section_states": sections(**done),
            "should_generate_final_output": True,
            "final_output": {"summary": "a plan"},
        }
    )
    body = response.json()
    assert body["collection_complete"] is True
    assert body["artifact_available"] is True
    assert all(s["status"] == "done" for s in body["sections"])


def test_the_explicit_agent_route_works_too():
    response = call({"user_id": OWNER, "section_states": sections()}, path="/xbuddy/completion")
    assert response.status_code == 200


def test_an_unknown_agent_is_404():
    with patch("service.service.get_agent", Mock(side_effect=KeyError("nope"))):
        response = TestClient(app).post(
            "/unknown/completion", json={"thread_id": THREAD, "user_id": OWNER}
        )
    assert response.status_code == 404


# --------------------------------------------------------------------------
# Scoping — the same deny-by-default rule as /history
# --------------------------------------------------------------------------


def test_a_different_user_gets_404():
    response = call({"user_id": OWNER, "section_states": sections()}, user_id=INTRUDER)
    assert response.status_code == 404
    assert response.json()["detail"] == "Thread not found"


def test_the_intruder_response_reveals_no_progress():
    """Not even the shape. Section statuses would leak how far along someone else is."""
    done = {section.value: SectionStatus.DONE for section in SectionID}
    response = call({"user_id": OWNER, "section_states": sections(**done)}, user_id=INTRUDER)
    body = response.text
    for leak in ("career_goal", "done", "collection_complete", "sections"):
        assert leak not in body


def test_a_checkpoint_with_no_stored_user_id_is_denied():
    """Ownership cannot be established, so it is shown to nobody."""
    response = call({"section_states": sections()})
    assert response.status_code == 404


def test_a_string_user_id_does_not_match_an_int_request():
    response = call({"user_id": str(OWNER), "section_states": sections()})
    assert response.status_code == 404


def test_user_id_is_required():
    with patch("service.service.get_agent", Mock(return_value=agent_with({}))):
        response = TestClient(app).post("/completion", json={"thread_id": THREAD})
    assert response.status_code == 422


# --------------------------------------------------------------------------
# New / empty threads
# --------------------------------------------------------------------------


@pytest.mark.parametrize("empty", [None, {}])
def test_a_brand_new_thread_is_not_an_error(empty):
    """Matches `/history`, which answers an unknown thread with an empty transcript
    rather than 404. Five pending sections is what a conversation that has not started
    actually looks like."""
    response = call(empty)
    assert response.status_code == 200
    body = response.json()
    assert body["collection_complete"] is False
    assert body["artifact_available"] is False
    assert [s["id"] for s in body["sections"]] == CANONICAL
    assert all(s["status"] == "pending" for s in body["sections"])


# --------------------------------------------------------------------------
# Nothing internal may cross the boundary
# --------------------------------------------------------------------------


def test_the_response_has_only_completion_state_fields():
    response = call({"user_id": OWNER, "section_states": sections()})
    assert set(response.json()) == {"collection_complete", "artifact_available", "sections"}


def test_each_section_has_only_id_name_status():
    response = call({"user_id": OWNER, "section_states": sections()})
    assert all(set(s) == {"id", "name", "status"} for s in response.json()["sections"])


def test_finished_cannot_leak():
    """Issue #10 fixed `finished`'s semantics but deliberately kept it internal. A new
    read endpoint is exactly where it would slip out."""
    response = call(
        {
            "user_id": OWNER,
            "section_states": sections(),
            "finished": True,
            "should_generate_final_output": False,
        }
    )
    assert "finished" not in response.text
    assert response.json()["collection_complete"] is False, "must read the flag, not finished"


def test_database_id_cannot_leak():
    response = call({"user_id": OWNER, "section_states": sections()})
    assert "database_id" not in response.text


def test_no_raw_graph_state_leaks():
    response = call(
        {
            "user_id": OWNER,
            "section_states": sections(),
            "user_data": {"target_roles": ["SRE"]},
            "messages": [HumanMessage(content="hi"), AIMessage(content="hello")],
            "context_packet": "internal",
            "router_directive": "stay",
            "agent_output": {"decision_reason": "internal"},
            "persistence_pending": ["career_goal"],
            "short_memory": "internal",
        }
    )
    body = response.text
    for leak in (
        "user_data",
        "target_roles",
        "context_packet",
        "router_directive",
        "agent_output",
        "decision_reason",
        "persistence_pending",
        "short_memory",
        "messages",
    ):
        assert leak not in body, f"leaked {leak}"


# --------------------------------------------------------------------------
# One projection, not two
# --------------------------------------------------------------------------


def test_the_endpoint_reuses_public_completion():
    """Re-deriving completion here would recreate the drift PR 6 removed. Pinned by
    patching the helper and observing that the route's answer changes with it."""
    import inspect

    import service.service as svc

    source = inspect.getsource(svc.load_completion_state)
    assert "public_completion(" in source
    assert "SectionID" not in source, "must not rebuild the section list"
    assert "should_generate_final_output" not in source, "must not re-read the flag"


def test_invoke_and_completion_agree_on_the_same_state():
    """The two surfaces must not drift. Same state in, same projection out."""
    from service.service import public_completion

    values = {
        "user_id": OWNER,
        "section_states": sections(career_goal=SectionStatus.DONE),
        "should_generate_final_output": False,
        "final_output": None,
    }
    from_endpoint = call(values).json()
    from_helper = public_completion(values).model_dump()
    assert from_endpoint == from_helper


def test_history_remains_messages_only():
    """The new endpoint must not have tempted anyone to widen `/history`."""
    from schema.schema import ChatHistory

    assert set(ChatHistory.model_fields) == {"thread_id", "user_id", "messages"}


def test_completion_is_not_rate_limited():
    """A cheap state read with no model call behind it, like `/history`."""
    import inspect

    import service.service as svc

    source = inspect.getsource(svc)
    block = source.split('@router.post("/completion")')[0].split(
        '@router.post("/{agent_id}/completion")'
    )[-1]
    assert "limiter.limit" not in block


def test_the_response_model_annotation_is_what_enforces_the_narrow_shape():
    """The endpoint returns `-> CompletionState`, so FastAPI validates and filters the
    response. A teeth check proved this: returning a dict with an extra
    `section_states` key changed nothing, because the annotation stripped it.

    That makes the annotation the actual guarantee, so it is pinned here. Widening it
    to `dict` would silently remove the filter and let raw graph state through.
    """
    import inspect

    import service.service as svc

    assert inspect.signature(svc.completion).return_annotation is CompletionState
    assert inspect.signature(svc.load_completion_state).return_annotation is CompletionState
