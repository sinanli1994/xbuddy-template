"""Tests for the JobBuddy router node.

The router is deterministic, so these assert exact navigation outcomes. Several
tests deliberately go one step further and run `route_decision` on the merged
state, because the router's contract is only meaningful in terms of the route
the graph actually takes afterwards.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy.enums import RouterDirective, SectionID, SectionStatus
from agents.xbuddy.graph.routes import route_decision
from agents.xbuddy.models import ContextPacket, SectionContent, SectionState, XBuddyData
from agents.xbuddy.nodes.router import router_node
from agents.xbuddy.state_factory import build_initial_state, build_section_states


def cold_state(**overrides):
    state = build_initial_state(user_id=1, thread_id="t-router")
    state["messages"] = []
    state.update(overrides)
    return state


def sections_with(**statuses) -> dict[str, SectionState]:
    """Section states with the named sections set to the given status."""
    sections = build_section_states()
    for section_value, status in statuses.items():
        sections[section_value] = sections[section_value].model_copy(update={"status": status})
    return sections


def all_done() -> dict[str, SectionState]:
    return sections_with(**{s.value: SectionStatus.DONE for s in SectionID})


# --------------------------------------------------------------------------
# Directive application
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stay_leaves_current_section_untouched():
    state = cold_state(
        current_section=SectionID.JOB_PREFERENCES,
        router_directive=RouterDirective.STAY,
        section_states=sections_with(job_preferences=SectionStatus.IN_PROGRESS),
    )
    update = await router_node(state, {})
    assert update["current_section"] is SectionID.JOB_PREFERENCES


@pytest.mark.asyncio
async def test_next_on_cold_start_does_not_skip_career_goal():
    """The regression guard: initialize emits NEXT *and* CAREER_GOAL together.

    Advancing by sequence here would land on BACKGROUND and the user would never
    see the first section.
    """
    update = await router_node(cold_state(), {})
    assert update["current_section"] is SectionID.CAREER_GOAL


@pytest.mark.asyncio
async def test_next_advances_past_a_done_section():
    state = cold_state(
        current_section=SectionID.CAREER_GOAL,
        router_directive=RouterDirective.NEXT,
        section_states=sections_with(career_goal=SectionStatus.DONE),
    )
    update = await router_node(state, {})
    assert update["current_section"] is SectionID.BACKGROUND


@pytest.mark.asyncio
async def test_next_with_everything_done_finishes():
    state = cold_state(
        current_section=SectionID.ACTION_PLAN,
        router_directive=RouterDirective.NEXT,
        section_states=all_done(),
    )
    update = await router_node(state, {})
    assert update["finished"] is True
    assert update["current_section"] is SectionID.ACTION_PLAN


@pytest.mark.asyncio
async def test_valid_modify_jumps_to_target():
    state = cold_state(
        current_section=SectionID.CAREER_GOAL,
        router_directive="modify:skill_assessment",
    )
    update = await router_node(state, {})
    assert update["current_section"] is SectionID.SKILL_ASSESSMENT


# --------------------------------------------------------------------------
# Status transitions
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_promoted_done_and_in_progress_preserved():
    original = sections_with(
        career_goal=SectionStatus.DONE,
        background=SectionStatus.IN_PROGRESS,
    )
    original["career_goal"] = original["career_goal"].model_copy(
        update={"content": SectionContent(content={"type": "doc"}, plain_text="kept")}
    )
    snapshot = {key: value.model_copy(deep=True) for key, value in original.items()}

    state = cold_state(
        current_section=SectionID.JOB_PREFERENCES,
        router_directive=RouterDirective.STAY,
        section_states=original,
    )
    update = await router_node(state, {})
    sections = update["section_states"]

    # Active PENDING section promoted.
    assert sections["job_preferences"].status is SectionStatus.IN_PROGRESS
    # Others untouched.
    assert sections["career_goal"].status is SectionStatus.DONE
    assert sections["career_goal"].content.plain_text == "kept"
    assert sections["background"].status is SectionStatus.IN_PROGRESS
    # The input dict was not mutated in place.
    assert original == snapshot


@pytest.mark.asyncio
async def test_done_section_is_not_downgraded_on_revisit():
    state = cold_state(
        current_section=SectionID.BACKGROUND,
        router_directive="modify:career_goal",
        section_states=sections_with(career_goal=SectionStatus.DONE),
    )
    update = await router_node(state, {})
    assert update["current_section"] is SectionID.CAREER_GOAL
    sections = update.get("section_states", state["section_states"])
    assert sections["career_goal"].status is SectionStatus.DONE


@pytest.mark.asyncio
async def test_raw_dict_section_states_are_coerced():
    state = cold_state(
        current_section=SectionID.BACKGROUND,
        router_directive=RouterDirective.STAY,
        section_states={
            "background": {
                "section_id": "background",
                "status": "in_progress",
                "satisfaction_status": None,
                "content": None,
            }
        },
    )
    update = await router_node(state, {})
    sections = update["section_states"]
    assert isinstance(sections["background"], SectionState)
    assert sections["background"].status.value == "in_progress"
    # The four absent sections are added by the router's own lookup path.
    assert update["current_section"] is SectionID.BACKGROUND


# --------------------------------------------------------------------------
# route_decision preservation contract
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "directive",
    [RouterDirective.STAY, RouterDirective.NEXT, "modify:background"],
)
async def test_valid_directives_are_preserved_and_messages_never_returned(directive):
    state = cold_state(router_directive=directive)
    update = await router_node(state, {})
    assert "router_directive" not in update, "a valid directive must not be rewritten"
    assert "messages" not in update


@pytest.mark.asyncio
async def test_context_packet_matches_resolved_section():
    state = cold_state(
        current_section=SectionID.CAREER_GOAL,
        router_directive="modify:action_plan",
        user_data=XBuddyData(target_roles=["SRE"]),
    )
    update = await router_node(state, {})
    packet = update["context_packet"]
    assert isinstance(packet, ContextPacket)
    assert packet.section_id is SectionID.ACTION_PLAN
    assert packet.section_id == update["current_section"]
    assert "SRE" in packet.system_prompt


# --------------------------------------------------------------------------
# Malformed directive normalization, asserted at the graph-route level
# --------------------------------------------------------------------------

# Directives that route_decision cannot interpret at all: none of them match its
# stay branch, its next branch, or startswith("modify:"), so each falls through
# to `return None` and silently ends the turn unless the router normalizes it.
DEAD_ENDING = ["modify", "garbage", None, "MODIFY"]

# These *do* satisfy route_decision's startswith("modify:") check, so they reach
# generate_reply either way. Normalizing them still matters: without it the
# directive claims a jump the router refused to make, and the next loop through
# the router would re-attempt the same invalid target.
ROUTABLE_BUT_INVALID = ["modify:nonsense", "modify:"]

MALFORMED = DEAD_ENDING + ROUTABLE_BUT_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("directive", MALFORMED)
async def test_malformed_directive_normalizes_to_stay(directive):
    state = cold_state(
        current_section=SectionID.BACKGROUND,
        router_directive=directive,
        section_states=sections_with(background=SectionStatus.IN_PROGRESS),
    )
    update = await router_node(state, {})
    assert update["router_directive"] == RouterDirective.STAY
    assert update["current_section"] is SectionID.BACKGROUND


@pytest.mark.asyncio
@pytest.mark.parametrize("directive", MALFORMED)
async def test_normalized_state_actually_routes_like_stay(directive):
    """The route must follow STAY, not just the section."""
    state = cold_state(
        current_section=SectionID.BACKGROUND,
        router_directive=directive,
        section_states=sections_with(background=SectionStatus.IN_PROGRESS),
        messages=[HumanMessage(content="actually, go back a bit")],
    )
    state.update(await router_node(state, {}))
    assert route_decision(state) == "generate_reply"


@pytest.mark.asyncio
@pytest.mark.parametrize("directive", DEAD_ENDING)
async def test_normalization_is_load_bearing_for_dead_ending_directives(directive):
    """Without normalization these silently end the turn with no reply.

    This is the test that fails if `router_directive = STAY` is ever dropped
    from the router's fallback path.
    """
    state = cold_state(
        current_section=SectionID.BACKGROUND,
        router_directive=directive,
        section_states=sections_with(background=SectionStatus.IN_PROGRESS),
        messages=[HumanMessage(content="actually, go back a bit")],
    )
    assert route_decision(state) is None, "control: un-normalized directive dead-ends"

    state.update(await router_node(state, {}))
    assert route_decision(state) == "generate_reply"


@pytest.mark.asyncio
@pytest.mark.parametrize("directive", MALFORMED)
async def test_normalized_state_ends_turn_without_pending_input(directive):
    """Real `stay` semantics: nothing to answer means end the turn."""
    state = cold_state(
        current_section=SectionID.BACKGROUND,
        router_directive=directive,
        section_states=sections_with(background=SectionStatus.IN_PROGRESS),
        messages=[AIMessage(content="what did you do most recently?")],
    )
    state.update(await router_node(state, {}))
    assert route_decision(state) is None


# --------------------------------------------------------------------------
# Completion routing
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completion_transition_ends_the_turn():
    """Fifth section finishing: memory_updater -> router -> END."""
    state = cold_state(
        current_section=SectionID.ACTION_PLAN,
        router_directive=RouterDirective.NEXT,
        section_states=all_done(),
        messages=[AIMessage(content="here is your plan")],
    )
    state.update(await router_node(state, {}))
    assert state["finished"] is True
    assert route_decision(state) is None


@pytest.mark.asyncio
async def test_new_user_message_on_finished_thread_still_replies():
    """Deliberate exception, pinned so routes.py can't silently go mute.

    Ending here would make the modify-reopen flow unreachable, since only
    generate_decision (downstream of generate_reply) can emit `modify:`.
    """
    state = cold_state(
        current_section=SectionID.ACTION_PLAN,
        router_directive=RouterDirective.NEXT,
        section_states=all_done(),
        messages=[HumanMessage(content="can we revisit my background?")],
    )
    state.update(await router_node(state, {}))
    assert state["finished"] is True
    assert route_decision(state) == "generate_reply"


# --------------------------------------------------------------------------
# Reopening after completion
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_modify_reopens_finished_workflow():
    sections = all_done()
    sections["background"] = sections["background"].model_copy(
        update={
            "content": SectionContent(content={"type": "doc"}, plain_text="8 years at Acme"),
            "satisfaction_status": "satisfied",
        }
    )
    state = cold_state(
        current_section=SectionID.ACTION_PLAN,
        router_directive="modify:background",
        section_states=sections,
        finished=True,
        messages=[HumanMessage(content="I want to redo my background")],
    )
    update = await router_node(state, {})

    assert update["current_section"] is SectionID.BACKGROUND
    assert update["finished"] is False

    merged = update.get("section_states", sections)
    assert merged["background"].status is SectionStatus.DONE, "status preserved for PR 4"
    assert merged["background"].content.plain_text == "8 years at Acme"
    assert merged["background"].satisfaction_status == "satisfied"

    state.update(update)
    assert route_decision(state) == "generate_reply"


@pytest.mark.asyncio
async def test_invalid_modify_does_not_reopen():
    state = cold_state(
        current_section=SectionID.ACTION_PLAN,
        router_directive="modify:nonsense",
        section_states=all_done(),
        finished=True,
    )
    update = await router_node(state, {})
    assert update["router_directive"] == RouterDirective.STAY
    assert update["current_section"] is SectionID.ACTION_PLAN
    assert "finished" not in update, "an invalid modify must not reopen the workflow"


# --------------------------------------------------------------------------
# Integration: initialize -> router through the compiled graph
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_then_router_through_compiled_graph():
    """generate_reply is PR 3, so the run is expected to stop there."""
    import uuid

    from agents.xbuddy.agent import graph

    config = {"configurable": {"thread_id": str(uuid.uuid4()), "user_id": 7}}
    with pytest.raises(NotImplementedError, match="PR 3"):
        await graph.ainvoke({"messages": [HumanMessage(content="I need a new job")]}, config)

    values = (await graph.aget_state(config)).values

    # PR 1's initialization survived, and the router ran on top of it.
    assert values["user_id"] == 7
    assert values["current_section"] is SectionID.CAREER_GOAL
    assert values["section_states"]["career_goal"].status is SectionStatus.IN_PROGRESS
    assert all(
        values["section_states"][s.value].status is SectionStatus.PENDING
        for s in SectionID
        if s is not SectionID.CAREER_GOAL
    )

    packet = values["context_packet"]
    assert packet.section_id is SectionID.CAREER_GOAL
    assert packet.validation_rules["required_fields"] == [
        "target_roles",
        "career_goal_summary",
        "target_timeline",
    ]
    # The user's message was not duplicated by either node.
    assert len(values["messages"]) == 1
