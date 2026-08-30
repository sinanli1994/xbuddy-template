"""`/final_output` — reading the finished career plan.

The endpoint that lets a client show the plan `artifact_available` has been promising
since PR 6. Deliberately the same shape as `/history` and `/completion`: same agent
resolution, same `aget_state`, same deny-by-default scoping, same "a new thread is not
an error" answer.

Two properties matter most and get the most tests. It is **read-only** — no graph run,
no model call, no write — and it is **checkpoint-sourced**, so its `artifact_available`
cannot contradict `/completion`'s for the same state.
"""

import inspect
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import SectionState
from schema.schema import FinalOutputResponse
from service import app


def executable_source(fn) -> str:
    """A function's source with its docstring and comments removed.

    Checks about what the code *does* must not be satisfied — or broken — by prose
    explaining what it deliberately avoids. `load_final_output`'s docstring names
    `final-outputs` precisely to record why it is not read, and a raw scan reads that
    as a violation.
    """
    import ast
    import textwrap

    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    body = tree.body[0].body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return NEWLINE.join(ast.unparse(node) for node in body)


NEWLINE = chr(10)

OWNER = 5150
INTRUDER = 6060
THREAD = "final-output-thread"

PLAN = """# Your Job Search Strategy

## Career Direction
Move from backend engineering into AI engineering within six months.

## Positioning
Lead with production Python and distributed systems experience.

- Ship one retrieval project
- Rewrite the CV around inference work
"""


def sections(**statuses) -> dict:
    return {
        section.value: SectionState(
            section_id=section, status=statuses.get(section.value, SectionStatus.DONE)
        )
        for section in SectionID
    }


def agent_with(values: dict | None):
    agent = AsyncMock()

    async def _aget_state(config=None):
        snapshot = Mock()
        snapshot.values = values
        snapshot.tasks = []
        return snapshot

    agent.aget_state = _aget_state
    return agent


def call(values, *, user_id=OWNER, thread_id=THREAD, path="/final_output", agent=None):
    resolved = agent if agent is not None else agent_with(values)
    with patch("service.service.get_agent", Mock(return_value=resolved)):
        return TestClient(app).post(path, json={"thread_id": thread_id, "user_id": user_id})


def completed_state(plan=PLAN) -> dict:
    return {
        "user_id": OWNER,
        "section_states": sections(),
        "should_generate_final_output": True,
        "final_output": plan,
    }


# --------------------------------------------------------------------------
# The artifact exists
# --------------------------------------------------------------------------


def test_a_finished_plan_is_returned():
    response = call(completed_state())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["artifact_available"] is True
    assert body["final_output"] == PLAN
    assert body["thread_id"] == THREAD
    assert body["user_id"] == OWNER


def test_the_plan_is_returned_as_markdown_not_a_structured_object():
    """`implementation_node` renders the structured FinalOutput and stores the
    Markdown, so there is no object to flatten here."""
    body = call(completed_state()).json()
    assert isinstance(body["final_output"], str)
    assert body["final_output"].startswith("#")


def test_the_explicit_agent_route_matches_the_base_route():
    base = call(completed_state()).json()
    prefixed = call(completed_state(), path="/xbuddy/final_output").json()
    assert base == prefixed


def test_an_unknown_agent_is_404():
    with patch("service.service.get_agent", Mock(side_effect=KeyError("nope"))):
        response = TestClient(app).post(
            "/unknown/final_output", json={"thread_id": THREAD, "user_id": OWNER}
        )
    assert response.status_code == 404


# --------------------------------------------------------------------------
# Not ready yet — a read projection, not an error
# --------------------------------------------------------------------------


def test_a_thread_without_a_plan_is_200_and_null():
    response = call(
        {"user_id": OWNER, "section_states": sections(career_goal=SectionStatus.IN_PROGRESS)}
    )
    assert response.status_code == 200
    assert response.json()["artifact_available"] is False
    assert response.json()["final_output"] is None


def test_an_explicit_null_artifact_is_not_available():
    response = call({"user_id": OWNER, "final_output": None})
    assert response.json() == {
        "thread_id": THREAD,
        "user_id": OWNER,
        "artifact_available": False,
        "final_output": None,
    }


@pytest.mark.parametrize("empty", [None, {}])
def test_a_brand_new_thread_is_not_an_error(empty):
    response = call(empty)
    assert response.status_code == 200
    assert response.json()["artifact_available"] is False
    assert response.json()["final_output"] is None


# --------------------------------------------------------------------------
# Scoping
# --------------------------------------------------------------------------


def test_a_different_user_gets_404():
    response = call(completed_state(), user_id=INTRUDER)
    assert response.status_code == 404
    assert response.json()["detail"] == "Thread not found"


def test_the_intruder_response_contains_no_part_of_the_plan():
    response = call(completed_state(), user_id=INTRUDER)
    for leak in ("Career Direction", "backend engineering", "Positioning", "final_output"):
        assert leak not in response.text


def test_a_checkpoint_with_no_stored_user_id_is_denied():
    response = call({"final_output": PLAN})
    assert response.status_code == 404


def test_a_string_user_id_does_not_match_an_int_request():
    response = call({"user_id": str(OWNER), "final_output": PLAN})
    assert response.status_code == 404


def test_user_id_is_required():
    with patch("service.service.get_agent", Mock(return_value=agent_with({}))):
        response = TestClient(app).post("/final_output", json={"thread_id": THREAD})
    assert response.status_code == 422


def test_arbitrary_state_keys_are_not_accepted():
    """The request names a thread and a user, nothing else."""
    from schema.schema import FinalOutputInput

    assert set(FinalOutputInput.model_fields) == {"thread_id", "user_id"}


# --------------------------------------------------------------------------
# Nothing internal crosses the boundary
# --------------------------------------------------------------------------


def test_the_response_has_only_the_four_public_fields():
    body = call(completed_state()).json()
    assert set(body) == {"thread_id", "user_id", "artifact_available", "final_output"}


def test_no_raw_graph_state_leaks():
    response = call(
        {
            **completed_state(),
            "user_data": {"target_roles": ["MLE"]},
            "context_packet": "internal",
            "router_directive": "stay",
            "agent_output": {"decision_reason": "internal"},
            "finished": True,
            "persistence_pending": ["career_goal"],
            "short_memory": "internal",
        }
    )
    body = response.text
    for leak in (
        "user_data", "target_roles", "context_packet", "router_directive",
        "agent_output", "decision_reason", "persistence_pending", "short_memory",
        "section_states", "should_generate_final_output",
    ):
        assert leak not in body, f"leaked {leak}"


def test_finished_cannot_leak():
    """Issue #10 fixed `finished`'s semantics but kept it internal. A new read
    endpoint is exactly where it would slip out."""
    assert "finished" not in call({**completed_state(), "finished": True}).text


def test_the_response_model_annotation_is_what_enforces_the_shape():
    """FastAPI validates and filters against the annotation, so widening it to `dict`
    would silently let raw state through."""
    import service.service as svc

    assert inspect.signature(svc.final_output).return_annotation is FinalOutputResponse
    assert inspect.signature(svc.load_final_output).return_annotation is FinalOutputResponse


# --------------------------------------------------------------------------
# Read-only: no graph run, no model call, no write
# --------------------------------------------------------------------------


def test_the_endpoint_never_invokes_the_graph():
    """The strongest available proof: the agent's execution methods are mocks, and a
    passing request must leave every one of them uncalled."""
    agent = agent_with(completed_state())
    agent.ainvoke = AsyncMock()
    agent.astream = Mock()
    agent.invoke = Mock()
    agent.update_state = AsyncMock()

    response = call(None, agent=agent)

    assert response.status_code == 200
    agent.ainvoke.assert_not_called()
    agent.astream.assert_not_called()
    agent.invoke.assert_not_called()
    agent.update_state.assert_not_called()


def test_the_handler_contains_no_write_or_generation_call():
    import service.service as svc

    source = executable_source(svc.load_final_output)
    for forbidden in (
        "ainvoke", "astream", "update_state", "persist_final_output",
        "synthesize", "render_final_output", "get_model",
    ):
        assert forbidden not in source, f"{forbidden} appears in a read-only handler"


def test_the_artifact_is_returned_unchanged():
    """No re-rendering, no normalisation — the stored document, byte for byte."""
    assert call(completed_state()).json()["final_output"] == PLAN


def test_it_is_not_rate_limited():
    """A cheap state read, like /history and /completion."""
    import service.service as svc

    source = inspect.getsource(svc)
    block = source.split('@router.post("/final_output")')[0].split(
        '@router.post("/{agent_id}/final_output")'
    )[-1]
    assert "limiter.limit" not in block


# --------------------------------------------------------------------------
# One source of truth
# --------------------------------------------------------------------------


def test_availability_agrees_with_the_completion_endpoint():
    """Both derive from `final_output is not None` on the same snapshot, so they
    cannot disagree for a given state."""
    from service.service import public_completion

    for state in (completed_state(), {"user_id": OWNER, "final_output": None}):
        from_final = call(state).json()["artifact_available"]
        from_completion = public_completion(state).artifact_available
        assert from_final == from_completion


def test_the_handler_reads_the_checkpoint_not_supabase():
    """Reading the domain row would introduce a second source that can be missing:
    the Supabase write is deliberately non-fatal."""
    import service.service as svc

    source = executable_source(svc.load_final_output)
    assert "aget_state" in source
    for forbidden in ("supabase", "final-outputs", "_fetch_sync", "stored_markdown"):
        assert forbidden not in source


def test_availability_tracks_the_artifact_not_the_completion_flag():
    """The two genuinely diverge, and the divergence is the documented case:
    `collection_complete=True, artifact_available=False` is a conversation whose five
    sections finished but whose synthesis failed.

    A teeth check proved this test was needed — every other fixture here has the two
    agreeing, so deriving availability from `should_generate_final_output` instead of
    the artifact passed the whole file unchanged.
    """
    response = call(
        {
            "user_id": OWNER,
            "section_states": sections(),
            "should_generate_final_output": True,
            "final_output": None,
        }
    )
    assert response.status_code == 200
    assert response.json()["artifact_available"] is False, "must read the artifact"
    assert response.json()["final_output"] is None


def test_the_inverse_divergence_is_also_handled():
    """An artifact present while the flag says otherwise still reports available —
    the artifact is what the caller can actually be shown."""
    response = call(
        {
            "user_id": OWNER,
            "section_states": sections(),
            "should_generate_final_output": False,
            "final_output": PLAN,
        }
    )
    assert response.json()["artifact_available"] is True
    assert response.json()["final_output"] == PLAN
