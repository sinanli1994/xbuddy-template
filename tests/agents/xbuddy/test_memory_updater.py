"""Stage 2 tests: section-scoped extraction wired into memory_updater_node.

The node is the first thing in the codebase that writes `user_data`, so the
tests concentrate on two things: it extracts with the *right* schema for the
active section, and no failure path is ever allowed to lose data the user
already gave.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import (
    BackgroundExtract,
    CareerGoalExtract,
    ChatAgentOutput,
    JobPreferencesExtract,
    SectionContent,
    SectionState,
    SkillAssessmentExtract,
    XBuddyData,
)
from agents.xbuddy.nodes.memory_updater import (
    EXTRACTION_WINDOW_SIZE,
    INTERNAL_EXTRACTION_TAG,
    memory_updater_node,
)
from agents.xbuddy.state_factory import build_section_states

FORBIDDEN_KEYS = ("messages", "current_section", "section_states", "finished")


def career_goal(**overrides) -> CareerGoalExtract:
    values = {"target_roles": None, "career_goal_summary": None, "target_timeline": None}
    values.update(overrides)
    return CareerGoalExtract(**values)


def turn(user: str = "I'm after a Senior SRE role", agent: str = "What role next?") -> list:
    return [AIMessage(content=agent), HumanMessage(content=user)]


# --------------------------------------------------------------------------
# Section scoping
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("section", "expected_model"),
    [
        (SectionID.CAREER_GOAL, CareerGoalExtract),
        (SectionID.BACKGROUND, BackgroundExtract),
        (SectionID.JOB_PREFERENCES, JobPreferencesExtract),
        (SectionID.SKILL_ASSESSMENT, SkillAssessmentExtract),
    ],
)
async def test_selects_the_active_sections_schema(
    extraction_chain, make_state, section, expected_model
):
    extraction_chain.extracted = None
    extraction_chain.parsing_error = ValueError("stop after schema selection")

    await memory_updater_node(make_state(section=section, messages=turn()), {})

    assert extraction_chain.last_model is expected_model


@pytest.mark.asyncio
async def test_prompt_carries_the_recent_window_and_section_context(
    extraction_chain, make_state
):
    extraction_chain.extracted = career_goal()
    history = [
        HumanMessage(content=f"msg {i}") if i % 2 == 0 else AIMessage(content=f"reply {i}")
        for i in range(14)
    ]

    await memory_updater_node(make_state(messages=history), {})

    sent = extraction_chain.calls[-1]
    window = sent[1:]  # index 0 is the SystemMessage
    assert len(window) == EXTRACTION_WINDOW_SIZE
    assert window == history[-EXTRACTION_WINDOW_SIZE:]

    prompt = extraction_chain.last_system_prompt
    assert "CURRENT SECTION: career_goal" in prompt
    assert "USE NULL, NOT AN EMPTY LIST" in prompt
    # Already-stored values are shown so corrections are distinguishable.
    assert "ALREADY STORED FOR THIS SECTION" in prompt
    assert "target_roles" in prompt


@pytest.mark.asyncio
async def test_prompt_shows_existing_values_for_correction_detection(
    extraction_chain, make_state
):
    extraction_chain.extracted = career_goal()
    state = make_state(
        user_data=XBuddyData(target_roles=["SRE"], target_timeline="3 months"), messages=turn()
    )

    await memory_updater_node(state, {})

    prompt = extraction_chain.last_system_prompt
    assert "SRE" in prompt
    assert "3 months" in prompt
    assert "not yet provided" in prompt  # career_goal_summary is still empty


def test_tag_matches_the_service_filter():
    """service.py drops streamed chunks carrying this tag; if the constant drifts
    from that list, extraction JSON starts appearing in the user's SSE stream."""
    from pathlib import Path

    source = Path("src/service/service.py").read_text(encoding="utf-8")
    assert f'"{INTERNAL_EXTRACTION_TAG}"' in source


@pytest.mark.asyncio
async def test_chain_is_built_with_the_internal_extraction_tag(make_state, monkeypatch):
    """The tag must be applied where the chain is built, not left to callers."""
    from agents.xbuddy.nodes import memory_updater as module

    captured: dict = {}

    class FakeModel:
        def with_structured_output(self, schema, **kwargs):
            captured["schema"] = schema
            captured["kwargs"] = kwargs
            return self

        def with_config(self, **kwargs):
            captured["tags"] = kwargs.get("tags")
            return self

    monkeypatch.setattr("core.llm.get_model", lambda *a, **k: FakeModel())
    module._extraction_chain(CareerGoalExtract)

    assert captured["tags"] == [INTERNAL_EXTRACTION_TAG]
    assert captured["schema"] is CareerGoalExtract
    assert captured["kwargs"] == {
        "method": "json_schema",
        "strict": True,
        "include_raw": True,
    }


# --------------------------------------------------------------------------
# Successful extraction
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_extraction_populates_only_active_section_fields(
    extraction_chain, make_state
):
    extraction_chain.extracted = career_goal(
        target_roles=["Senior SRE"], target_timeline="3 months"
    )
    state = make_state(
        user_data=XBuddyData(current_role="QA Analyst", strengths=["testing"]), messages=turn()
    )

    update = await memory_updater_node(state, {})

    data = update["user_data"]
    assert data.target_roles == ["Senior SRE"]
    assert data.target_timeline == "3 months"
    # Other sections untouched.
    assert data.current_role == "QA Analyst"
    assert data.strengths == ["testing"]
    # Unmentioned field in the *active* section also untouched.
    assert data.career_goal_summary is None


@pytest.mark.asyncio
async def test_extraction_does_not_mutate_the_state_object(extraction_chain, make_state):
    extraction_chain.extracted = career_goal(target_roles=["SRE"])
    state = make_state(user_data=XBuddyData(), messages=turn())
    original = state["user_data"]

    update = await memory_updater_node(state, {})

    assert original.target_roles == [], "state's XBuddyData was mutated in place"
    assert update["user_data"] is not original


@pytest.mark.asyncio
async def test_correction_overwrites_through_the_real_node_path(extraction_chain, make_state):
    extraction_chain.extracted = career_goal(target_timeline="6 months")
    state = make_state(
        user_data=XBuddyData(target_roles=["SRE"], target_timeline="3 months"),
        messages=turn(user="actually make that six months"),
    )

    update = await memory_updater_node(state, {})

    assert update["user_data"].target_timeline == "6 months"
    assert update["user_data"].target_roles == ["SRE"], "unrelated field was disturbed"


# --------------------------------------------------------------------------
# None / [] must not clobber, through the node
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "extracted_kwargs"),
    [
        ("all null", {}),
        ("empty list", {"target_roles": []}),
        ("empty list + null scalars", {"target_roles": [], "target_timeline": None}),
    ],
)
async def test_no_op_extraction_omits_user_data_entirely(
    extraction_chain, make_state, label, extracted_kwargs
):
    """Nothing new means no `user_data` key at all — not a rewrite of the same value."""
    extraction_chain.extracted = career_goal(**extracted_kwargs)
    existing = XBuddyData(target_roles=["SRE"], target_timeline="3 months")
    state = make_state(user_data=existing, messages=turn())

    update = await memory_updater_node(state, {})

    assert "user_data" not in update, f"{label}: emitted an unchanged user_data"
    assert state["user_data"].target_roles == ["SRE"], f"{label}: clobbered stored data"
    assert state["user_data"].target_timeline == "3 months"


@pytest.mark.asyncio
async def test_partial_extraction_preserves_unmentioned_fields(extraction_chain, make_state):
    extraction_chain.extracted = career_goal(career_goal_summary="Wants more autonomy")
    state = make_state(
        user_data=XBuddyData(target_roles=["SRE"], target_timeline="3 months"), messages=turn()
    )

    update = await memory_updater_node(state, {})

    data = update["user_data"]
    assert data.career_goal_summary == "Wants more autonomy"
    assert data.target_roles == ["SRE"]
    assert data.target_timeline == "3 months"


# --------------------------------------------------------------------------
# Guards and failure paths
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_context_packet_short_circuits_before_the_model(
    extraction_chain, make_state
):
    state = make_state(with_packet=False, error_count=2, messages=turn())
    update = await memory_updater_node(state, {})

    assert extraction_chain.call_count == 0, "must not extract without knowing the section"
    assert "context_packet" in update["last_error"]
    assert update["error_count"] == 3
    assert "user_data" not in update


@pytest.mark.asyncio
async def test_empty_message_history_degrades_without_error(extraction_chain, make_state):
    """Nothing said yet is a no-op, not a failure: no model call, no error."""
    update = await memory_updater_node(make_state(messages=[], error_count=1), {})

    assert extraction_chain.call_count == 0
    assert update == {}
    assert "last_error" not in update
    assert "error_count" not in update


@pytest.mark.asyncio
async def test_model_exception_does_not_raise(extraction_chain, make_state):
    extraction_chain.raises = RuntimeError("rate limited")
    existing = XBuddyData(target_roles=["SRE"])
    state = make_state(user_data=existing, error_count=0, messages=turn())

    update = await memory_updater_node(state, {})

    assert "rate limited" in update["last_error"]
    assert update["error_count"] == 1
    assert "user_data" not in update, "a failed extraction must not touch stored data"
    assert state["user_data"].target_roles == ["SRE"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "extracted", "parsing_error"),
    [
        ("parsing_error set", None, ValueError("schema refusal")),
        ("parsed is None", None, None),
        ("parsed is the wrong model", BackgroundExtract(
            current_role=None, years_experience=None, highest_education=None, work_history=None
        ), None),
    ],
)
async def test_parsing_failures_do_not_mutate_user_data(
    extraction_chain, make_state, label, extracted, parsing_error
):
    extraction_chain.extracted = extracted
    extraction_chain.parsing_error = parsing_error
    existing = XBuddyData(target_roles=["SRE"], target_timeline="3 months")
    state = make_state(user_data=existing, error_count=0, messages=turn())

    update = await memory_updater_node(state, {})

    assert "user_data" not in update, f"{label}: user_data was touched"
    assert update["error_count"] == 1, f"{label}: error not counted exactly once"
    assert update["last_error"], f"{label}: last_error not set"
    assert state["user_data"] == existing


# --------------------------------------------------------------------------
# Node contract
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario", ["success", "no_op", "missing_packet", "model_error", "empty_history"]
)
async def test_node_never_returns_messages_or_navigation(
    extraction_chain, make_state, scenario
):
    kwargs: dict = {"messages": turn()}
    extraction_chain.extracted = career_goal(target_roles=["SRE"])

    if scenario == "no_op":
        extraction_chain.extracted = career_goal()
    elif scenario == "missing_packet":
        kwargs["with_packet"] = False
    elif scenario == "model_error":
        extraction_chain.raises = RuntimeError("boom")
    elif scenario == "empty_history":
        kwargs["messages"] = []

    update = await memory_updater_node(make_state(**kwargs), {})

    for key in FORBIDDEN_KEYS:
        assert key not in update, f"{scenario}: leaked {key}"


@pytest.mark.asyncio
async def test_stage_4_and_5_behaviour_is_absent(extraction_chain, make_state):
    """PR 4 stage boundary: no persistence, no retry queue.

    With no `agent_output` in state nothing is satisfied, so Stage 3's keys are
    absent too and the update is extraction-only.
    """
    extraction_chain.extracted = career_goal(target_roles=["SRE"])
    update = await memory_updater_node(make_state(messages=turn()), {})

    assert "persistence_pending" not in update
    assert "should_generate_final_output" not in update
    assert "section_states" not in update
    assert set(update) == {"user_data"}


@pytest.mark.asyncio
async def test_error_count_increments_once_per_failed_turn(extraction_chain, make_state):
    """Two failures in a row advance the counter by one each, never by two."""
    extraction_chain.raises = RuntimeError("boom")

    first = await memory_updater_node(make_state(error_count=0, messages=turn()), {})
    assert first["error_count"] == 1

    second = await memory_updater_node(make_state(error_count=1, messages=turn()), {})
    assert second["error_count"] == 2


@pytest.mark.asyncio
async def test_unknown_section_fails_without_calling_the_model(
    extraction_chain, make_state, monkeypatch
):
    """Registry drift is caught as an error, not a crash."""
    from agents.xbuddy.nodes import memory_updater as module

    def boom(_section_id):
        raise ValueError("Unknown section_id: 'bogus'")

    monkeypatch.setattr(module, "get_extract_model", boom)

    update = await memory_updater_node(make_state(error_count=0, messages=turn()), {})

    assert extraction_chain.call_count == 0
    assert "no extraction schema" in update["last_error"]
    assert update["error_count"] == 1
    assert "user_data" not in update


# --------------------------------------------------------------------------
# Stage 3: DONE transitions and the completion flag
# --------------------------------------------------------------------------


def satisfied(value: bool | None) -> ChatAgentOutput:
    """An agent_output carrying a satisfaction verdict."""
    return ChatAgentOutput(
        reply="So: Senior SRE within 3 months. Right?",
        router_directive="stay",
        is_satisfied=value,
        user_satisfaction_feedback=None,
        should_save_content=False,
    )


def sections_with(**statuses) -> dict:
    result = build_section_states()
    for key, status in statuses.items():
        result[key] = result[key].model_copy(update={"status": status})
    return result


@pytest.mark.asyncio
async def test_is_satisfied_true_marks_the_current_section_done(extraction_chain, make_state):
    extraction_chain.extracted = career_goal()
    state = make_state(
        section=SectionID.CAREER_GOAL,
        messages=turn(),
        agent_output=satisfied(True),
        section_states=sections_with(career_goal=SectionStatus.IN_PROGRESS),
    )

    update = await memory_updater_node(state, {})

    assert update["section_states"]["career_goal"].status is SectionStatus.DONE


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", [False, None])
async def test_not_satisfied_leaves_status_unchanged(extraction_chain, make_state, verdict):
    extraction_chain.extracted = career_goal()
    state = make_state(
        messages=turn(),
        agent_output=satisfied(verdict),
        section_states=sections_with(career_goal=SectionStatus.IN_PROGRESS),
    )

    update = await memory_updater_node(state, {})

    assert "section_states" not in update, f"is_satisfied={verdict} must not complete a section"
    assert "should_generate_final_output" not in update


@pytest.mark.asyncio
async def test_missing_agent_output_leaves_status_unchanged(extraction_chain, make_state):
    extraction_chain.extracted = career_goal()
    update = await memory_updater_node(
        make_state(messages=turn(), section_states=sections_with()), {}
    )
    assert "section_states" not in update


@pytest.mark.asyncio
async def test_only_the_current_section_transitions(extraction_chain, make_state):
    extraction_chain.extracted = None
    extraction_chain.parsing_error = ValueError("irrelevant here")
    state = make_state(
        section=SectionID.BACKGROUND,
        messages=turn(),
        agent_output=satisfied(True),
        section_states=sections_with(background=SectionStatus.IN_PROGRESS),
    )

    merged = (await memory_updater_node(state, {}))["section_states"]

    assert merged["background"].status is SectionStatus.DONE
    for other in SectionID:
        if other is not SectionID.BACKGROUND:
            assert merged[other.value].status is SectionStatus.PENDING, other.value


@pytest.mark.asyncio
async def test_already_done_is_never_downgraded(extraction_chain, make_state):
    """A modify: revisit that ends satisfied must keep DONE, not rewrite it."""
    extraction_chain.extracted = career_goal()
    sections = build_section_states()
    sections["career_goal"] = sections["career_goal"].model_copy(
        update={
            "status": SectionStatus.DONE,
            "content": SectionContent(content={"type": "doc"}, plain_text="kept"),
            "satisfaction_status": "satisfied",
        }
    )
    state = make_state(messages=turn(), agent_output=satisfied(True), section_states=sections)

    update = await memory_updater_node(state, {})

    # Nothing changed, so no section_states is emitted at all.
    assert "section_states" not in update
    stored = state["section_states"]["career_goal"]
    assert stored.status is SectionStatus.DONE
    assert stored.content.plain_text == "kept"
    assert stored.satisfaction_status == "satisfied"


@pytest.mark.asyncio
async def test_raw_dict_section_states_are_coerced(extraction_chain, make_state):
    """Checkpoints can hand back plain dicts; downstream reads attributes."""
    extraction_chain.extracted = career_goal()
    state = make_state(
        messages=turn(),
        agent_output=satisfied(True),
        section_states={
            "career_goal": {
                "section_id": "career_goal",
                "status": "in_progress",
                "satisfaction_status": None,
                "content": None,
            }
        },
    )

    entry = (await memory_updater_node(state, {}))["section_states"]["career_goal"]

    assert isinstance(entry, SectionState)
    assert entry.status is SectionStatus.DONE
    assert entry.status.value == "done"


@pytest.mark.asyncio
async def test_input_section_states_is_not_mutated(extraction_chain, make_state):
    extraction_chain.extracted = career_goal()
    original = sections_with(career_goal=SectionStatus.IN_PROGRESS)
    snapshot = {key: value.model_copy(deep=True) for key, value in original.items()}
    state = make_state(messages=turn(), agent_output=satisfied(True), section_states=original)

    update = await memory_updater_node(state, {})

    assert original == snapshot, "the checkpointed mapping was mutated in place"
    assert update["section_states"] is not original


@pytest.mark.asyncio
async def test_completion_flag_false_at_four_of_five(extraction_chain, make_state):
    extraction_chain.extracted = career_goal()
    state = make_state(
        section=SectionID.SKILL_ASSESSMENT,
        messages=turn(),
        agent_output=satisfied(True),
        section_states=sections_with(
            career_goal=SectionStatus.DONE,
            background=SectionStatus.DONE,
            job_preferences=SectionStatus.DONE,
            skill_assessment=SectionStatus.IN_PROGRESS,
        ),
    )

    update = await memory_updater_node(state, {})

    assert update["section_states"]["skill_assessment"].status is SectionStatus.DONE
    assert update["section_states"]["action_plan"].status is SectionStatus.PENDING
    assert "should_generate_final_output" not in update


@pytest.mark.asyncio
async def test_completion_flag_true_at_five_of_five(extraction_chain, make_state):
    extraction_chain.extracted = career_goal()
    state = make_state(
        section=SectionID.ACTION_PLAN,
        messages=turn(),
        agent_output=satisfied(True),
        section_states=sections_with(
            career_goal=SectionStatus.DONE,
            background=SectionStatus.DONE,
            job_preferences=SectionStatus.DONE,
            skill_assessment=SectionStatus.DONE,
            action_plan=SectionStatus.IN_PROGRESS,
        ),
    )

    update = await memory_updater_node(state, {})

    assert update["section_states"]["action_plan"].status is SectionStatus.DONE
    assert update["should_generate_final_output"] is True


@pytest.mark.asyncio
async def test_completion_flag_not_set_from_a_partial_mapping(extraction_chain, make_state):
    """all() over a two-entry mapping would be vacuously true — it must not be."""
    extraction_chain.extracted = career_goal()
    state = make_state(
        messages=turn(),
        agent_output=satisfied(True),
        section_states={
            "career_goal": SectionState(
                section_id=SectionID.CAREER_GOAL, status=SectionStatus.IN_PROGRESS
            )
        },
    )

    update = await memory_updater_node(state, {})

    assert update["section_states"]["career_goal"].status is SectionStatus.DONE
    assert "should_generate_final_output" not in update, "only 1 of 5 sections exists"


@pytest.mark.asyncio
async def test_done_transition_survives_an_extraction_failure(extraction_chain, make_state):
    """Satisfaction is a separate signal: a model failure must not withhold it."""
    extraction_chain.raises = RuntimeError("rate limited")
    state = make_state(
        messages=turn(),
        error_count=0,
        agent_output=satisfied(True),
        section_states=sections_with(career_goal=SectionStatus.IN_PROGRESS),
    )

    update = await memory_updater_node(state, {})

    assert update["section_states"]["career_goal"].status is SectionStatus.DONE
    assert update["error_count"] == 1
    assert "rate limited" in update["last_error"]
    assert "user_data" not in update


@pytest.mark.asyncio
async def test_missing_packet_skips_progress_too(extraction_chain, make_state):
    """Without a packet we do not know which section to complete."""
    state = make_state(
        with_packet=False,
        messages=turn(),
        agent_output=satisfied(True),
        section_states=sections_with(career_goal=SectionStatus.IN_PROGRESS),
    )

    update = await memory_updater_node(state, {})

    assert "section_states" not in update
    assert "should_generate_final_output" not in update
    assert update["last_error"]


# --------------------------------------------------------------------------
# should_save_content does not gate persistence — the disagreement is intended
#
# `SectionDecision.should_save_content` is a required field of the strict
# decision schema, but DECISION_RULES never defines it, so the model emits it
# with no rule to apply. Persistence is gated on `current_is_dirty` instead —
# whether the durable row would actually differ. The two disagree in both
# directions, and both directions are deliberate. See the comment beside
# `current_is_dirty` in memory_updater.py.
# --------------------------------------------------------------------------


def decided(*, should_save_content: bool, is_satisfied: bool | None = None) -> ChatAgentOutput:
    """An agent_output with the advisory flag set explicitly.

    Written out rather than reusing `satisfied()`, whose hardcoded
    `should_save_content=False` is exactly the detail these tests are about.
    """
    return ChatAgentOutput(
        reply="So: Senior SRE within 3 months. Right?",
        router_directive="stay",
        is_satisfied=is_satisfied,
        user_satisfaction_feedback=None,
        should_save_content=should_save_content,
    )


@pytest.mark.asyncio
async def test_persists_despite_should_save_content_false(
    extraction_chain, make_state, persistence
):
    """A real state change is written even when the model advises against saving.

    The dangerous direction: honouring the flag here would drop freshly captured
    user data without queueing it in `persistence_pending`, so the durable record
    would fall silently behind state — the exact failure the retry queue exists to
    make visible.
    """
    extraction_chain.extracted = career_goal(target_roles=["Senior SRE"])
    state = make_state(
        messages=turn(),
        agent_output=decided(should_save_content=False),
        section_states=sections_with(career_goal=SectionStatus.IN_PROGRESS),
    )

    update = await memory_updater_node(state, {})

    assert state["agent_output"].should_save_content is False
    assert update["user_data"].target_roles == ["Senior SRE"], "extraction must have changed state"
    assert persistence.sections == ["career_goal"], "a dirty section is persisted regardless"
    assert "persistence_pending" not in update


@pytest.mark.asyncio
async def test_does_not_persist_despite_should_save_content_true(
    extraction_chain, make_state, persistence
):
    """An unchanged section is not written even when the model advises saving.

    All-null extraction means "nothing new was said", the common case. Honouring
    the flag here would re-upsert a byte-identical row every turn, spending a
    network round-trip on the path most likely to fail.
    """
    extraction_chain.extracted = career_goal()  # every field None -> no merge
    state = make_state(
        messages=turn(),
        user_data=XBuddyData(target_roles=["Senior SRE"]),
        agent_output=decided(should_save_content=True),
        section_states=sections_with(career_goal=SectionStatus.IN_PROGRESS),
    )

    update = await memory_updater_node(state, {})

    assert state["agent_output"].should_save_content is True
    assert "user_data" not in update, "nothing extracted, so nothing changed"
    assert "section_states" not in update, "no status change either"
    assert persistence.calls == [], "a clean section is not re-written"
