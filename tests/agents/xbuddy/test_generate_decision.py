"""Tests for generate_decision_node — the machine-facing half of a turn.

Two themes dominate: the node must never navigate, and its two guards must
short-circuit *before* the model is called. Several tests therefore assert
`decision_chain.call_count == 0` rather than only checking the returned state.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy.enums import DecisionAction, RouterDirective, SectionID
from agents.xbuddy.models import ChatAgentOutput, SectionDecision, XBuddyData
from agents.xbuddy.nodes.generate_decision import (
    INTERNAL_DECISION_TAG,
    MAX_REPLIES_PER_TURN,
    generate_decision_node,
)

NAVIGATION_KEYS = ("current_section", "section_states", "finished")


# --------------------------------------------------------------------------
# Action -> directive mapping
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "target", "expected"),
    [
        (DecisionAction.STAY, None, "stay"),
        (DecisionAction.NEXT, None, "next"),
        (DecisionAction.MODIFY, SectionID.BACKGROUND, "modify:background"),
        (DecisionAction.MODIFY, SectionID.ACTION_PLAN, "modify:action_plan"),
    ],
)
async def test_action_maps_to_directive(
    decision_chain, make_state, make_decision, action, target, expected
):
    decision_chain.decision = make_decision(action=action, modify_target=target)
    update = await generate_decision_node(make_state(), {})
    assert update["router_directive"] == expected


@pytest.mark.asyncio
async def test_modify_without_target_falls_back_to_stay(
    decision_chain, make_state, make_decision
):
    decision_chain.decision = make_decision(action=DecisionAction.MODIFY, modify_target=None)
    update = await generate_decision_node(make_state(), {})

    assert update["router_directive"] == "stay"
    for key in NAVIGATION_KEYS:
        assert key not in update


# --------------------------------------------------------------------------
# Fallbacks
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parsing_error_falls_back(decision_chain, make_state):
    decision_chain.decision = None
    decision_chain.parsing_error = ValueError("schema refusal")

    update = await generate_decision_node(make_state(error_count=0), {})
    assert update["router_directive"] == "stay"
    assert "unparseable" in update["last_error"]
    assert update["error_count"] == 1


@pytest.mark.asyncio
async def test_model_exception_falls_back_and_counts(decision_chain, make_state):
    decision_chain.raises = RuntimeError("rate limited")

    update = await generate_decision_node(make_state(error_count=2), {})
    assert update["router_directive"] == "stay"
    assert update["error_count"] == 3
    assert "rate limited" in update["last_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    ["missing_packet", "model_raises", "unparseable_output"],
)
async def test_every_reachable_error_path_records_and_counts_together(
    decision_chain, make_state, make_decision, scenario
):
    """last_error and error_count must move as a pair on every failure path.

    They used to sit behind separate flags, so three of the four `error=True`
    call sites recorded an error without ever counting it.

    These are the three failure paths a real turn can actually take. The fourth
    `error=True` call site — a directive rejected by the ChatAgentDecision
    validator — is unreachable by construction and is covered by
    test_composed_directive_is_always_valid instead of being faked here.
    """
    state_kwargs: dict = {}
    decision_chain.raises = None
    decision_chain.parsing_error = None
    decision_chain.decision = make_decision()

    if scenario == "missing_packet":
        state_kwargs["with_packet"] = False
    elif scenario == "model_raises":
        decision_chain.raises = RuntimeError("boom")
    elif scenario == "unparseable_output":
        decision_chain.decision = None
        decision_chain.parsing_error = ValueError("refusal")

    update = await generate_decision_node(make_state(error_count=5, **state_kwargs), {})

    assert update["router_directive"] == "stay"
    assert update.get("last_error"), "last_error not set"
    assert update.get("error_count") == 6, "error_count did not increment by exactly 1"


def test_composed_directive_is_always_valid():
    """Why there is no fourth error scenario above.

    Every directive `_compose_directive` can emit is accepted by the
    ChatAgentDecision validator, so the node's final validation gate cannot
    fire. Exhaustive over all (action, modify_target) pairs a validated
    SectionDecision can hold.

    This is the test that would fail if someone changed _compose_directive to
    emit a shape the validator rejects — at which point the defensive branch in
    the node stops being dead code and the scenario list above needs a fourth
    entry.
    """
    from itertools import product

    from agents.xbuddy.models import ChatAgentDecision
    from agents.xbuddy.nodes.generate_decision import _compose_directive

    def build(action, target):
        return SectionDecision(
            action=action,
            modify_target=target,
            is_satisfied=None,
            user_satisfaction_feedback=None,
            should_save_content=False,
            presented_summary=False,
            decision_reason="x",
        )

    combinations = list(product(DecisionAction, [None, *SectionID]))
    assert len(combinations) == 18

    for action, target in combinations:
        directive = _compose_directive(build(action, target))
        # Raises ValidationError if the node's final gate would have rejected it.
        ChatAgentDecision(
            router_directive=directive,
            user_satisfaction_feedback=None,
            is_satisfied=None,
            should_save_content=False,
        )
        assert directive in ("stay", "next") or directive.startswith("modify:")


def test_pydantic_blocks_out_of_range_actions_and_targets():
    """The upstream half of the same argument.

    An invalid action or target never reaches _compose_directive: Pydantic
    rejects it while parsing, which routes to the unparseable-output path
    instead.
    """
    from pydantic import ValidationError

    base = {
        "action": "stay",
        "modify_target": None,
        "is_satisfied": None,
        "user_satisfaction_feedback": None,
        "should_save_content": False,
        "presented_summary": False,
        "decision_reason": "x",
    }
    for bad in ({"action": "skip"}, {"action": "STAY"}, {"modify_target": "nonsense"}):
        with pytest.raises(ValidationError):
            SectionDecision(**{**base, **bad})


@pytest.mark.asyncio
async def test_reply_cap_is_not_an_error(decision_chain, make_state):
    """The cap is normal control flow — it must neither count nor record."""
    state = make_state(
        error_count=5,
        messages=[
            HumanMessage(content="I need a job"),
            AIMessage(content="one"),
            AIMessage(content="two"),
        ],
    )
    update = await generate_decision_node(state, {})

    assert update["router_directive"] == "stay"
    assert "error_count" not in update, "the cap is not a failure"
    assert "last_error" not in update


# --------------------------------------------------------------------------
# Contract: no messages, no navigation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_never_returns_messages_or_navigation(decision_chain, make_state, make_decision):
    decision_chain.decision = make_decision(action=DecisionAction.NEXT)
    update = await generate_decision_node(make_state(), {})

    assert "messages" not in update
    for key in NAVIGATION_KEYS:
        assert key not in update, f"{key} is the router's to write, not the decision node's"


@pytest.mark.asyncio
async def test_agent_output_carries_reply_and_directive(
    decision_chain, make_state, make_decision
):
    decision_chain.decision = make_decision(
        action=DecisionAction.STAY,
        is_satisfied=False,
        user_satisfaction_feedback="wants the timeline changed",
        should_save_content=True,
    )
    state = make_state(
        messages=[HumanMessage(content="hi"), AIMessage(content="What role next?")]
    )
    update = await generate_decision_node(state, {})

    output = update["agent_output"]
    assert isinstance(output, ChatAgentOutput)
    assert output.reply == "What role next?"
    assert output.router_directive == "stay"
    assert output.is_satisfied is False
    assert output.user_satisfaction_feedback == "wants the timeline changed"
    assert output.should_save_content is True


# --------------------------------------------------------------------------
# Guards — both must short-circuit before the model call
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply_cap_short_circuits_before_the_model(decision_chain, make_state):
    """The guard exists to stop a runaway loop; calling the model first would
    still cost a request and a trace span on every capped iteration."""
    state = make_state(
        messages=[
            HumanMessage(content="I need a job"),
            AIMessage(content="reply one"),
            AIMessage(content="reply two"),
        ]
    )
    update = await generate_decision_node(state, {})

    assert decision_chain.call_count == 0, "cap must be checked before the LLM call"
    assert update["router_directive"] == "stay"


@pytest.mark.asyncio
async def test_under_the_cap_the_model_is_called(decision_chain, make_state, make_decision):
    decision_chain.decision = make_decision(action=DecisionAction.NEXT)
    state = make_state(
        messages=[HumanMessage(content="I need a job"), AIMessage(content="reply one")]
    )
    update = await generate_decision_node(state, {})

    assert decision_chain.call_count == 1
    assert update["router_directive"] == "next"
    assert MAX_REPLIES_PER_TURN == 2


@pytest.mark.asyncio
async def test_missing_context_packet_short_circuits_before_the_model(
    decision_chain, make_state
):
    """A missing packet is a real system error, and is accounted for as one.

    It previously set last_error without touching error_count, so a repeatedly
    failing router looked healthy to anything watching the counter.
    """
    update = await generate_decision_node(make_state(with_packet=False, error_count=3), {})

    # Never reaches the model.
    assert decision_chain.call_count == 0, "must not decide without knowing the section"
    # Safe STAY.
    assert update["router_directive"] == "stay"
    # Recorded and counted, exactly once.
    assert "context_packet" in update["last_error"]
    assert update["error_count"] == 4
    # No messages, no navigation.
    assert "messages" not in update
    for key in NAVIGATION_KEYS:
        assert key not in update


# --------------------------------------------------------------------------
# awaiting_satisfaction_feedback (plan §3.1)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "presented", "satisfied", "expected"),
    [
        (DecisionAction.STAY, True, None, True),    # handshake open
        (DecisionAction.STAY, True, True, False),   # confirmed
        (DecisionAction.STAY, True, False, True),   # rejected -> still open
        (DecisionAction.STAY, False, None, False),  # no summary
        (DecisionAction.NEXT, True, True, False),   # moving on
        (DecisionAction.MODIFY, True, None, False), # reopening resets
    ],
)
async def test_awaiting_flag_is_always_written(
    decision_chain, make_state, make_decision, action, presented, satisfied, expected
):
    decision_chain.decision = make_decision(
        action=action,
        modify_target=SectionID.BACKGROUND if action is DecisionAction.MODIFY else None,
        presented_summary=presented,
        is_satisfied=satisfied,
    )
    update = await generate_decision_node(make_state(), {})

    assert "awaiting_satisfaction_feedback" in update, "must be written on every run"
    assert update["awaiting_satisfaction_feedback"] is expected


@pytest.mark.asyncio
async def test_guard_paths_preserve_an_open_handshake(decision_chain, make_state):
    """A guard has no evidence about the handshake, so it must not close one."""
    state = make_state(with_packet=False, awaiting_satisfaction_feedback=True)
    update = await generate_decision_node(state, {})
    assert update["awaiting_satisfaction_feedback"] is True


# --------------------------------------------------------------------------
# Prompt content
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_required_fields_are_computed_and_injected(
    decision_chain, make_state, make_decision
):
    """The node computes completeness; the model only judges the conversation."""
    decision_chain.decision = make_decision()
    state = make_state(
        user_data=XBuddyData(target_roles=["SRE"])  # summary + timeline still empty
    )
    await generate_decision_node(state, {})

    prompt = decision_chain.last_system_prompt
    assert "career_goal_summary" in prompt
    assert "target_timeline" in prompt
    assert "target_roles" not in prompt.split("REQUIRED FIELDS STILL EMPTY:")[1].split("\n")[0]


# --------------------------------------------------------------------------
# Schema must stay strict-compatible
# --------------------------------------------------------------------------


def test_section_decision_schema_is_strict_compatible():
    """OpenAI strict json_schema requires every property in `required`.

    A Pydantic default silently drops a field from `required` and invalidates the
    schema, so this fails the moment anyone adds one.
    """
    schema = SectionDecision.model_json_schema()
    properties = set(schema["properties"])
    required = set(schema["required"])

    assert properties == required, f"not required: {sorted(properties - required)}"
    assert len(properties) == 7


def test_internal_decision_tag_matches_the_service_filter():
    """service.py drops streamed chunks carrying this tag; if the constant drifts
    from that list, decision JSON starts appearing in the user's SSE stream."""
    from pathlib import Path

    service_source = Path("src/service/service.py").read_text(encoding="utf-8")
    assert f'"{INTERNAL_DECISION_TAG}"' in service_source


def test_directive_values_match_router_expectations():
    assert RouterDirective.STAY.value == "stay"
    assert RouterDirective.NEXT.value == "next"
    assert [a.value for a in DecisionAction] == ["stay", "next", "modify"]
