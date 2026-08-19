"""Offline tests for the PR 5 Stage 3 implementation node.

Synthesis is faked at `synthesis._synthesis_chain`, the same seam the Stage 2 tests
use, so these exercise the real `synthesize_final_output` and
`assemble_final_output` paths — only the model is replaced. An autouse guard makes
any real chain construction fail loudly.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy import synthesis
from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import (
    ActionAnnotation,
    FinalOutputDraft,
    SectionState,
    XBuddyData,
)
from agents.xbuddy.nodes.implementation import (
    FINAL_OUTPUT_READY_MESSAGE,
    implementation_node,
)

CONFIRMED = [
    "Rewrite the CV summary to lead with the platform migration",
    "Message three former colleagues at target companies this week",
    "Ship a small Kubernetes project and write it up",
]


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("a test tried to build a real LLM chain")

    monkeypatch.setattr("core.llm.get_model", explode)


class CountingChain:
    """Counts model invocations so 'did not pay again' is directly assertable."""

    def __init__(self, result=None, raises=None):
        self.result = result
        self.raises = raises
        self.calls = 0

    async def ainvoke(self, messages, *args, **kwargs):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.result


def draft(steps=CONFIRMED) -> FinalOutputDraft:
    return FinalOutputDraft(
        headline="QA Analyst to Senior SRE within 3 months",
        positioning_summary="Four years of QA moving into automation.",
        strengths_to_leverage=["systems debugging"],
        skill_priorities=["Kubernetes"],
        search_targets=["fintech"],
        action_annotations=[
            ActionAnnotation(step_number=index, rationale=f"reason {index}", timeframe=None)
            for index in range(1, len(steps) + 1)
        ],
        risks_or_constraints=["Three-month timeline is tight"],
    )


def all_done() -> dict:
    return {
        section.value: SectionState(section_id=section, status=SectionStatus.DONE)
        for section in SectionID
    }


def user_data(**overrides) -> XBuddyData:
    values = {
        "target_roles": ["Senior SRE"],
        "target_timeline": "within 3 months",
        "current_role": "QA Analyst",
        "action_items": list(CONFIRMED),
    }
    values.update(overrides)
    return XBuddyData(**values)


def eligible_state(**overrides) -> dict:
    state = {
        "messages": [HumanMessage(content="yes, ship it")],
        "section_states": all_done(),
        "user_data": user_data(),
        "error_count": 0,
    }
    state.update(overrides)
    return state


@pytest.fixture
def chain(monkeypatch):
    counting = CountingChain(result={"parsed": draft(), "parsing_error": None})
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: counting)
    return counting


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_sections_done_produces_a_final_output(chain):
    update = await implementation_node(eligible_state(), {})

    assert chain.calls == 1
    assert update["final_output"]
    assert "last_error" not in update
    assert "error_count" not in update


@pytest.mark.asyncio
async def test_final_output_is_rendered_markdown_not_structured_json(chain):
    update = await implementation_node(eligible_state(), {})
    markdown = update["final_output"]

    assert markdown.startswith("# QA Analyst to Senior SRE within 3 months")
    assert "## Your Action Plan" in markdown
    assert "## What I Still Don't Know" in markdown
    # Not a serialized model: no JSON braces, no schema field names.
    assert not markdown.lstrip().startswith("{")
    for field_name in ("action_annotations", "step_number", "positioning_summary"):
        assert field_name not in markdown


@pytest.mark.asyncio
async def test_confirmed_steps_survive_into_the_document(chain):
    markdown = (await implementation_node(eligible_state(), {}))["final_output"]
    for step in CONFIRMED:
        assert f"**{step}**" in markdown


@pytest.mark.asyncio
async def test_exactly_one_readiness_message_is_appended(chain):
    update = await implementation_node(eligible_state(), {})

    assert len(update["messages"]) == 1
    message = update["messages"][0]
    assert isinstance(message, AIMessage)
    assert message.content == FINAL_OUTPUT_READY_MESSAGE


@pytest.mark.asyncio
async def test_the_readiness_message_does_not_contain_the_artifact(chain):
    """The document lives in the editor; chat only points at it."""
    update = await implementation_node(eligible_state(), {})
    content = update["messages"][0].content

    assert len(content) < 200
    assert "##" not in content
    assert CONFIRMED[0] not in content


# --------------------------------------------------------------------------
# Once-only guard
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_final_output_short_circuits_without_a_model_call(chain):
    update = await implementation_node(
        eligible_state(final_output="# Already generated\n"), {}
    )

    assert update == {}
    assert chain.calls == 0, "an existing artifact must never be paid for twice"


@pytest.mark.asyncio
async def test_existing_final_output_appends_no_second_readiness_message(chain):
    update = await implementation_node(
        eligible_state(final_output="# Already generated\n"), {}
    )
    assert "messages" not in update


@pytest.mark.asyncio
async def test_the_flag_staying_true_does_not_cause_regeneration(chain):
    """`should_generate_final_output` is never cleared, so the node is re-entered
    on every later turn. The artifact itself is the guard."""
    state = eligible_state(
        final_output="# Already generated\n", should_generate_final_output=True
    )

    update = await implementation_node(state, {})

    assert update == {}
    assert chain.calls == 0


@pytest.mark.asyncio
async def test_two_consecutive_runs_synthesize_once(chain):
    """Simulates the real sequence: turn N generates, turn N+1 must not."""
    first = await implementation_node(eligible_state(), {})
    assert chain.calls == 1

    second = await implementation_node(
        eligible_state(final_output=first["final_output"]), {}
    )

    assert second == {}
    assert chain.calls == 1, "the second turn must not pay for synthesis again"


@pytest.mark.asyncio
async def test_an_empty_string_final_output_is_not_treated_as_generated(chain):
    """`""` is falsy and means "nothing generated", not "already done"."""
    update = await implementation_node(eligible_state(final_output=""), {})
    assert chain.calls == 1
    assert update["final_output"]


# --------------------------------------------------------------------------
# Eligibility and the empty-plan invariant
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_action_items_produces_an_error_and_no_artifact(chain):
    """Upstream invariant violation: Section 5 requires at least three steps."""
    update = await implementation_node(
        eligible_state(user_data=user_data(action_items=[])), {}
    )

    assert chain.calls == 0, "an absent plan must not be synthesized around"
    assert "final_output" not in update
    assert "messages" not in update
    assert update["error_count"] == 1
    assert "confirmed action items" in update["last_error"]


@pytest.mark.asyncio
async def test_missing_user_data_produces_an_error_and_no_artifact(chain):
    state = eligible_state()
    del state["user_data"]

    update = await implementation_node(state, {})

    assert chain.calls == 0
    assert "final_output" not in update
    assert "user_data" in update["last_error"]


@pytest.mark.asyncio
async def test_an_incomplete_section_blocks_synthesis(chain):
    sections = all_done()
    sections["action_plan"] = SectionState(
        section_id=SectionID.ACTION_PLAN, status=SectionStatus.IN_PROGRESS
    )

    update = await implementation_node(eligible_state(section_states=sections), {})

    assert chain.calls == 0
    assert "final_output" not in update
    assert "action_plan" in update["last_error"]


@pytest.mark.asyncio
async def test_raw_dict_section_states_are_coerced(chain):
    """Checkpoints can hand back plain dicts; eligibility must still be readable."""
    raw = {
        section.value: {
            "section_id": section.value,
            "status": "done",
            "satisfaction_status": None,
            "content": None,
        }
        for section in SectionID
    }

    update = await implementation_node(eligible_state(section_states=raw), {})

    assert chain.calls == 1
    assert update["final_output"]


@pytest.mark.asyncio
async def test_multiple_ineligibility_reasons_are_joined_and_counted_once(chain):
    """Joined, not replaced — one failure must not mask another."""
    state = eligible_state(section_states={}, user_data=user_data(action_items=[]))

    update = await implementation_node(state, {})

    assert update["error_count"] == 1, "one increment per failing turn"
    assert "still open" in update["last_error"]
    assert "confirmed action items" in update["last_error"]
    assert "; " in update["last_error"]


# --------------------------------------------------------------------------
# Synthesis failure: degrade, never partial, never raise
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parsing_failure_yields_no_partial_artifact(monkeypatch):
    counting = CountingChain(result={"parsed": None, "parsing_error": ValueError("bad json")})
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: counting)

    update = await implementation_node(eligible_state(), {})

    assert "final_output" not in update
    assert "messages" not in update
    assert update["error_count"] == 1
    assert "could not be parsed" in update["last_error"]


@pytest.mark.asyncio
async def test_model_exception_degrades_without_raising(monkeypatch):
    counting = CountingChain(raises=RuntimeError("rate limited"))
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: counting)

    update = await implementation_node(eligible_state(), {})

    assert "final_output" not in update
    assert "rate limited" in update["last_error"]


@pytest.mark.asyncio
async def test_annotation_count_mismatch_degrades(monkeypatch):
    """Assembly disagreement: the model annotated fewer steps than were confirmed."""
    bad = draft(steps=CONFIRMED[:1])
    counting = CountingChain(result={"parsed": bad, "parsing_error": None})
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: counting)

    update = await implementation_node(eligible_state(), {})

    assert "final_output" not in update
    assert "1 steps but 3 were confirmed" in update["last_error"]


@pytest.mark.asyncio
async def test_a_synthesis_function_that_raises_is_still_contained(monkeypatch):
    """Belt and braces: synthesize_final_output should not raise, but if it did the
    turn's conversational work must still survive."""

    async def boom(_user_data):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(
        "agents.xbuddy.nodes.implementation.synthesize_final_output", boom
    )

    update = await implementation_node(eligible_state(), {})

    assert "final_output" not in update
    assert "synthesis raised: unexpected" in update["last_error"]


@pytest.mark.asyncio
async def test_error_count_builds_on_the_existing_value(monkeypatch):
    counting = CountingChain(raises=RuntimeError("nope"))
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: counting)

    update = await implementation_node(eligible_state(error_count=4), {})

    assert update["error_count"] == 5


@pytest.mark.asyncio
async def test_failure_does_not_resurrect_a_stale_last_error(monkeypatch):
    """`last_error` is replaced, not appended to: nothing clears it on success, so
    joining with the previous value would resurface an error from an earlier turn."""
    counting = CountingChain(raises=RuntimeError("fresh failure"))
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: counting)

    update = await implementation_node(
        eligible_state(last_error="something from three turns ago"), {}
    )

    assert "three turns ago" not in update["last_error"]
    assert "fresh failure" in update["last_error"]


@pytest.mark.asyncio
async def test_a_failed_turn_leaves_the_artifact_retryable(monkeypatch):
    """No artifact written means the next turn tries again rather than being locked
    out by its own failure."""
    counting = CountingChain(raises=RuntimeError("transient"))
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: counting)
    failed = await implementation_node(eligible_state(), {})
    assert "final_output" not in failed

    recovered = CountingChain(result={"parsed": draft(), "parsing_error": None})
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: recovered)

    update = await implementation_node(eligible_state(), {})

    assert update["final_output"]
    assert recovered.calls == 1
