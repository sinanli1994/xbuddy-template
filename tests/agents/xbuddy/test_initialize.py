"""Tests for the JobBuddy initialize node.

The node runs on every invocation, so the properties that matter are:
  * cold start fills everything,
  * warm start preserves everything,
  * it is idempotent, and
  * it never returns `messages` (which carries the add_messages reducer).
"""

import uuid

import pytest

from agents.xbuddy.enums import RouterDirective, SectionID, SectionStatus
from agents.xbuddy.models import SectionContent, SectionState, XBuddyData, XBuddyState
from agents.xbuddy.nodes.initialize import initialize_node
from agents.xbuddy.state_factory import build_section_states

# Every state key initialize is responsible for — derived from the schema so the
# test cannot drift out of sync with the model.
MANAGED_KEYS = set(XBuddyState.__optional_keys__)


def make_config(**configurable):
    return {"configurable": configurable}


def fully_populated_state() -> dict:
    """A warm state with real progress: one section done, one in progress."""
    sections = build_section_states()
    sections[SectionID.CAREER_GOAL.value] = SectionState(
        section_id=SectionID.CAREER_GOAL,
        status=SectionStatus.DONE,
        satisfaction_status="satisfied",
        content=SectionContent(content={"type": "doc"}, plain_text="Become a staff engineer"),
    )
    sections[SectionID.BACKGROUND.value] = SectionState(
        section_id=SectionID.BACKGROUND,
        status=SectionStatus.IN_PROGRESS,
    )
    return {
        "messages": [],
        "user_id": 99,
        "thread_id": "warm-thread",
        "current_section": SectionID.BACKGROUND,
        "context_packet": None,
        "section_states": sections,
        "router_directive": "stay",
        "finished": False,
        "user_data": XBuddyData(target_roles=["Staff Engineer"], years_experience=8),
        "short_memory": [],
        "agent_output": None,
        "awaiting_user_input": True,
        "awaiting_satisfaction_feedback": False,
        "error_count": 0,
        "last_error": None,
        "final_output": None,
        "should_generate_final_output": False,
        "persistence_pending": [],
        "final_output_pending": None,
    }


# --------------------------------------------------------------------------
# 1. Cold start populates every key
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cold_start_populates_every_managed_key():
    update = await initialize_node({"messages": []}, make_config(user_id=7, thread_id="t-1"))

    assert set(update) == MANAGED_KEYS
    assert update["current_section"] is SectionID.CAREER_GOAL
    assert update["router_directive"] == RouterDirective.NEXT
    assert update["user_data"] == XBuddyData()
    assert update["short_memory"] == []
    assert update["context_packet"] is None
    assert update["agent_output"] is None
    assert update["finished"] is False
    assert update["awaiting_user_input"] is False
    assert update["awaiting_satisfaction_feedback"] is False
    assert update["should_generate_final_output"] is False
    assert update["error_count"] == 0
    assert update["last_error"] is None
    assert update["final_output"] is None


# --------------------------------------------------------------------------
# 2. Cold start seeds five PENDING sections
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cold_start_seeds_five_pending_sections():
    update = await initialize_node({"messages": []}, make_config())
    sections = update["section_states"]

    assert set(sections) == {section.value for section in SectionID}
    assert len(sections) == 5
    assert all(isinstance(section, SectionState) for section in sections.values())
    assert all(section.status is SectionStatus.PENDING for section in sections.values())
    # Keys are enum *values*, and each entry knows its own id.
    for key, section in sections.items():
        assert section.section_id.value == key


# --------------------------------------------------------------------------
# 3. Identity resolution (config -> state -> default)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "config", "expected_user_id"),
    [
        # config wins
        ({"messages": []}, make_config(user_id=7), 7),
        # falls back to state
        ({"messages": [], "user_id": 42}, make_config(), 42),
        # nothing anywhere -> default
        ({"messages": []}, make_config(), 1),
        # string from agent_config is coerced
        ({"messages": []}, make_config(user_id="42"), 42),
        # garbage falls back to the default rather than raising
        ({"messages": []}, make_config(user_id="not-an-int"), 1),
        # missing config entirely
        ({"messages": []}, None, 1),
        ({"messages": []}, {}, 1),
    ],
)
async def test_user_id_resolution(state, config, expected_user_id):
    update = await initialize_node(state, config)
    assert update["user_id"] == expected_user_id


@pytest.mark.asyncio
async def test_thread_id_resolution():
    from_config = await initialize_node({"messages": []}, make_config(thread_id="cfg-thread"))
    assert from_config["thread_id"] == "cfg-thread"

    from_state = await initialize_node(
        {"messages": [], "thread_id": "state-thread"}, make_config()
    )
    assert from_state["thread_id"] == "state-thread"

    # Config is authoritative: it is the checkpoint key.
    config_wins = await initialize_node(
        {"messages": [], "thread_id": "state-thread"}, make_config(thread_id="cfg-thread")
    )
    assert config_wins["thread_id"] == "cfg-thread"

    # Absent everywhere -> a fresh, valid UUID.
    generated = await initialize_node({"messages": []}, make_config())
    assert uuid.UUID(generated["thread_id"])


# --------------------------------------------------------------------------
# 4. Warm start preserves progress
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warm_start_preserves_all_progress():
    state = fully_populated_state()
    before = {key: state[key] for key in MANAGED_KEYS if key in state}

    update = await initialize_node(state, make_config(user_id=99, thread_id="warm-thread"))

    # A fully-populated state needs nothing but identity re-affirmed.
    assert set(update) == {"user_id", "thread_id"}
    assert update["user_id"] == 99
    assert update["thread_id"] == "warm-thread"

    # Nothing was mutated in place, and nothing was reset.
    for key, value in before.items():
        assert state[key] == value

    # The details that matter most for leave-and-return:
    assert state["router_directive"] == "stay"
    assert state["current_section"] is SectionID.BACKGROUND
    assert state["user_data"].target_roles == ["Staff Engineer"]
    assert state["section_states"][SectionID.CAREER_GOAL.value].status is SectionStatus.DONE
    assert (
        state["section_states"][SectionID.CAREER_GOAL.value].content.plain_text
        == "Become a staff engineer"
    )


# --------------------------------------------------------------------------
# 5. Section backfill is additive (and coerces raw dicts)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_section_backfill_is_additive_and_coerces_dicts():
    done = SectionState(
        section_id=SectionID.CAREER_GOAL,
        status=SectionStatus.DONE,
        satisfaction_status="satisfied",
        content=SectionContent(content={"type": "doc"}, plain_text="kept"),
    )
    # A checkpoint can hand back a plain dict instead of a model.
    raw_dict_entry = {
        "section_id": SectionID.BACKGROUND.value,
        "status": SectionStatus.IN_PROGRESS.value,
        "satisfaction_status": None,
        "content": None,
    }
    state = {
        "messages": [],
        "section_states": {
            SectionID.CAREER_GOAL.value: done,
            SectionID.BACKGROUND.value: raw_dict_entry,
        },
    }

    update = await initialize_node(state, make_config())
    sections = update["section_states"]

    # All five present; the three absent ones added as PENDING.
    assert set(sections) == {section.value for section in SectionID}
    for section_id in (
        SectionID.JOB_PREFERENCES,
        SectionID.SKILL_ASSESSMENT,
        SectionID.ACTION_PLAN,
    ):
        assert sections[section_id.value].status is SectionStatus.PENDING

    # The pre-existing entry survives untouched.
    assert sections[SectionID.CAREER_GOAL.value] == done
    assert sections[SectionID.CAREER_GOAL.value].content.plain_text == "kept"
    assert sections[SectionID.CAREER_GOAL.value].satisfaction_status == "satisfied"

    # The raw dict became a real model, so attribute access works downstream.
    coerced = sections[SectionID.BACKGROUND.value]
    assert isinstance(coerced, SectionState)
    assert coerced.status is SectionStatus.IN_PROGRESS
    assert coerced.status.value == "in_progress"


# --------------------------------------------------------------------------
# 6. Falsy values are not mistaken for missing keys
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_falsy_values_are_not_overwritten():
    state = {
        "messages": [],
        "finished": False,
        "error_count": 0,
        "awaiting_user_input": False,
        "should_generate_final_output": False,
        "router_directive": "stay",
        "short_memory": [],
        "user_data": XBuddyData(),
        "last_error": None,
        "final_output": None,
        "context_packet": None,
        "agent_output": None,
        "current_section": SectionID.SKILL_ASSESSMENT,
        "awaiting_satisfaction_feedback": False,
    }

    update = await initialize_node(state, make_config())

    # None of the falsy-but-present keys were refilled.
    for key in (
        "finished",
        "error_count",
        "awaiting_user_input",
        "should_generate_final_output",
        "router_directive",
        "short_memory",
        "current_section",
    ):
        assert key not in update, f"{key} was overwritten despite being present"

    assert state["router_directive"] == "stay"
    assert state["current_section"] is SectionID.SKILL_ASSESSMENT


# --------------------------------------------------------------------------
# 7. Idempotence
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_is_idempotent():
    config = make_config(user_id=5, thread_id="t-idem")

    state = {"messages": []}
    state.update(await initialize_node(state, config))
    after_first = dict(state)

    second_update = await initialize_node(state, config)
    state.update(second_update)

    # The second pass is a no-op beyond re-affirming identity.
    assert set(second_update) == {"user_id", "thread_id"}
    assert state == after_first


# --------------------------------------------------------------------------
# 8. `messages` is never returned; defaults are never aliased
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_messages_never_returned_and_defaults_not_aliased():
    first = await initialize_node({"messages": []}, make_config())
    second = await initialize_node({"messages": []}, make_config())

    # Returning `messages` would re-append through the add_messages reducer.
    assert "messages" not in first

    # Mutable defaults must come from factories, not shared literals.
    assert first["user_data"] is not second["user_data"]
    assert first["short_memory"] is not second["short_memory"]
    assert first["section_states"] is not second["section_states"]

    first["short_memory"].append("mutated")
    first["user_data"].target_roles.append("mutated")
    assert second["short_memory"] == []
    assert second["user_data"].target_roles == []

    # Section entries are distinct objects too.
    key = SectionID.CAREER_GOAL.value
    assert first["section_states"][key] is not second["section_states"][key]
