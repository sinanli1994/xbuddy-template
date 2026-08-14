"""Stage 5 tests: persistence wired into memory_updater, with the retry queue.

The `persistence` fixture is autouse (see conftest) and replaces
`memory_updater.persist_section`, so nothing here touches the live project. Tests
that steer it request it by name.

The property under most scrutiny is that state is allowed to run ahead of the
durable record: a Supabase outage must never roll back a `DONE` the user
confirmed, and the queue is what keeps that divergence visible.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import ChatAgentOutput, XBuddyData
from agents.xbuddy.nodes.memory_updater import memory_updater_node
from agents.xbuddy.state_factory import build_section_states


def career_goal_extract(**overrides):
    return extract_for(SectionID.CAREER_GOAL, **overrides)


def extract_for(section: SectionID, **overrides):
    """Build the extract model that matches `section`, all fields null by default.

    Section-aware on purpose: memory_updater checks
    `isinstance(parsed, extract_model)`, so handing a CareerGoalExtract to a
    Background turn is treated as a parsing failure — which silently turned two
    of these tests into no-op-extraction tests until this helper existed.
    """
    from agents.xbuddy.models import EXTRACT_MODELS

    model = EXTRACT_MODELS[section]
    values = dict.fromkeys(model.model_fields, None)
    values.update(overrides)
    return model(**values)


def turn() -> list:
    return [AIMessage(content="What role next?"), HumanMessage(content="Senior SRE")]


def satisfied(value: bool | None = True) -> ChatAgentOutput:
    return ChatAgentOutput(
        reply="So: Senior SRE. Right?",
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


# --------------------------------------------------------------------------
# Current-section write
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_current_section_success_leaves_no_queue(
    extraction_chain, make_state, persistence
):
    extraction_chain.extracted = career_goal_extract(target_roles=["SRE"])
    update = await memory_updater_node(make_state(messages=turn()), {})

    assert persistence.sections == ["career_goal"]
    assert "persistence_pending" not in update
    assert "last_error" not in update
    assert "error_count" not in update


@pytest.mark.asyncio
async def test_persistence_receives_the_merged_data_not_the_stale_copy(
    extraction_chain, make_state, persistence
):
    """The row must reflect this turn's extraction, not the pre-turn snapshot."""
    extraction_chain.extracted = career_goal_extract(target_roles=["SRE"])
    await memory_updater_node(make_state(messages=turn(), user_data=XBuddyData()), {})

    assert persistence.calls[-1]["user_data"].target_roles == ["SRE"]


@pytest.mark.asyncio
async def test_no_write_when_nothing_changed(extraction_chain, make_state, persistence):
    """A turn that reveals nothing and completes nothing needs no network call."""
    extraction_chain.extracted = career_goal_extract()  # all null -> no-op
    update = await memory_updater_node(make_state(messages=turn()), {})

    assert persistence.call_count == 0
    assert update == {}


@pytest.mark.asyncio
async def test_done_transition_alone_triggers_a_write(
    extraction_chain, make_state, persistence
):
    """Status changed even though extraction found nothing — the row differs."""
    extraction_chain.extracted = career_goal_extract()
    state = make_state(
        messages=turn(),
        agent_output=satisfied(),
        section_states=sections_with(career_goal=SectionStatus.IN_PROGRESS),
    )

    await memory_updater_node(state, {})

    assert persistence.sections == ["career_goal"]
    assert persistence.calls[-1]["status"] == "done"


@pytest.mark.asyncio
async def test_current_section_failure_queues_it(extraction_chain, make_state, persistence):
    extraction_chain.extracted = career_goal_extract(target_roles=["SRE"])
    persistence.fail = True

    update = await memory_updater_node(make_state(messages=turn(), error_count=0), {})

    assert update["persistence_pending"] == ["career_goal"]
    assert "persistence failed for section career_goal" in update["last_error"]
    assert update["error_count"] == 1


@pytest.mark.asyncio
async def test_done_survives_a_persistence_failure(
    extraction_chain, make_state, persistence
):
    """The central Stage 5 guarantee: an outage must not roll back a confirmation."""
    extraction_chain.extracted = career_goal_extract()
    persistence.fail = True
    state = make_state(
        messages=turn(),
        agent_output=satisfied(),
        section_states=sections_with(career_goal=SectionStatus.IN_PROGRESS),
    )

    update = await memory_updater_node(state, {})

    assert update["section_states"]["career_goal"].status is SectionStatus.DONE
    assert update["persistence_pending"] == ["career_goal"]
    assert update["error_count"] == 1


@pytest.mark.asyncio
async def test_user_data_survives_a_persistence_failure(
    extraction_chain, make_state, persistence
):
    """Extraction succeeded; only the write failed. The facts must still be kept."""
    extraction_chain.extracted = career_goal_extract(target_roles=["SRE"])
    persistence.fail = True

    update = await memory_updater_node(make_state(messages=turn()), {})

    assert update["user_data"].target_roles == ["SRE"]
    assert update["persistence_pending"] == ["career_goal"]


# --------------------------------------------------------------------------
# Retry behaviour
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_succeeds_and_clears_the_queued_section(
    extraction_chain, make_state, persistence
):
    extraction_chain.extracted = extract_for(SectionID.BACKGROUND)
    state = make_state(
        section=SectionID.BACKGROUND,
        messages=turn(),
        persistence_pending=["career_goal"],
    )

    update = await memory_updater_node(state, {})

    assert "career_goal" in persistence.sections
    assert update["persistence_pending"] == []


@pytest.mark.asyncio
async def test_retry_failure_keeps_it_queued(extraction_chain, make_state, persistence):
    extraction_chain.extracted = extract_for(SectionID.BACKGROUND)
    persistence.fail_sections = {"career_goal"}
    state = make_state(
        section=SectionID.BACKGROUND, messages=turn(), persistence_pending=["career_goal"]
    )

    update = await memory_updater_node(state, {})

    # Unchanged queue is not re-emitted.
    assert "persistence_pending" not in update
    assert "persistence retry failed for section career_goal" in update["last_error"]
    assert update["error_count"] == 1


@pytest.mark.asyncio
async def test_retries_run_before_the_current_section_write(
    extraction_chain, make_state, persistence
):
    """A section failing for several turns must not be starved behind the active one."""
    extraction_chain.extracted = extract_for(
        SectionID.SKILL_ASSESSMENT, current_skills=["Python"]
    )
    state = make_state(
        section=SectionID.SKILL_ASSESSMENT,
        messages=turn(),
        persistence_pending=["career_goal", "background"],
    )

    await memory_updater_node(state, {})

    assert persistence.sections == ["career_goal", "background", "skill_assessment"]


@pytest.mark.asyncio
async def test_current_section_is_never_written_twice(
    extraction_chain, make_state, persistence
):
    """Queued *and* active in the same turn must still be a single write."""
    extraction_chain.extracted = career_goal_extract(target_roles=["SRE"])
    state = make_state(
        section=SectionID.CAREER_GOAL, messages=turn(), persistence_pending=["career_goal"]
    )

    update = await memory_updater_node(state, {})

    assert persistence.sections == ["career_goal"], "written once, not twice"
    assert update["persistence_pending"] == []


@pytest.mark.asyncio
async def test_a_queued_section_is_written_even_when_nothing_changed(
    extraction_chain, make_state, persistence
):
    """The current section is retried on the strength of the queue alone."""
    extraction_chain.extracted = career_goal_extract()  # no-op extraction
    state = make_state(messages=turn(), persistence_pending=["career_goal"])

    update = await memory_updater_node(state, {})

    assert persistence.sections == ["career_goal"]
    assert update["persistence_pending"] == []


@pytest.mark.asyncio
async def test_partial_retry_success_removes_only_the_successes(
    extraction_chain, make_state, persistence
):
    extraction_chain.extracted = extract_for(SectionID.ACTION_PLAN)
    persistence.fail_sections = {"background"}
    state = make_state(
        section=SectionID.ACTION_PLAN,
        messages=turn(),
        persistence_pending=["career_goal", "background"],
    )

    update = await memory_updater_node(state, {})

    assert update["persistence_pending"] == ["background"]
    assert "background" in update["last_error"]
    assert "career_goal" not in update["last_error"]


@pytest.mark.asyncio
async def test_queued_section_absent_from_state_is_dropped(
    extraction_chain, make_state, persistence
):
    """Nothing to write, so stop retrying it forever."""
    extraction_chain.extracted = career_goal_extract()
    state = make_state(
        section=SectionID.CAREER_GOAL,
        messages=turn(),
        section_states={},  # no sections at all
        persistence_pending=["background"],
    )

    update = await memory_updater_node(state, {})

    assert "background" not in persistence.sections
    assert update["persistence_pending"] == []


# --------------------------------------------------------------------------
# Queue invariants
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_duplicate_queue_entries(extraction_chain, make_state, persistence):
    extraction_chain.extracted = career_goal_extract(target_roles=["SRE"])
    persistence.fail = True
    state = make_state(
        messages=turn(), persistence_pending=["career_goal", "career_goal", "career_goal"]
    )

    update = await memory_updater_node(state, {})

    assert update["persistence_pending"] == ["career_goal"]


@pytest.mark.asyncio
async def test_repeated_failures_keep_the_queue_bounded(
    extraction_chain, make_state, persistence
):
    """Five turns of total failure must not grow the queue past the section count."""
    extraction_chain.extracted = career_goal_extract(target_roles=["SRE"])
    persistence.fail = True

    pending: list[str] = []
    for _ in range(5):
        update = await memory_updater_node(
            make_state(messages=turn(), persistence_pending=pending), {}
        )
        pending = update.get("persistence_pending", pending)

    assert pending == ["career_goal"], "the same section must appear at most once"
    assert len(pending) <= len(SectionID)


@pytest.mark.asyncio
async def test_queue_can_hold_at_most_one_entry_per_section(
    extraction_chain, make_state, persistence
):
    extraction_chain.extracted = career_goal_extract()
    persistence.fail = True
    every_section = [section.value for section in SectionID]
    state = make_state(
        section=SectionID.CAREER_GOAL,
        messages=turn(),
        persistence_pending=every_section + every_section,
    )

    update = await memory_updater_node(state, {})

    queue = update["persistence_pending"]
    assert len(queue) == len(set(queue)) == len(SectionID)


# --------------------------------------------------------------------------
# Error accounting
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_count_increments_once_for_many_failed_rows(
    extraction_chain, make_state, persistence
):
    """It counts bad turns, not bad rows — three failures, one increment."""
    extraction_chain.extracted = career_goal_extract(target_roles=["SRE"])
    persistence.fail = True
    state = make_state(
        section=SectionID.CAREER_GOAL,
        messages=turn(),
        error_count=4,
        persistence_pending=["background", "job_preferences"],
    )

    update = await memory_updater_node(state, {})

    assert persistence.call_count == 3
    assert update["error_count"] == 5, "one increment, not three"


@pytest.mark.asyncio
async def test_extraction_and_persistence_failures_share_one_increment(
    extraction_chain, make_state, persistence
):
    extraction_chain.raises = RuntimeError("rate limited")
    persistence.fail = True
    state = make_state(
        messages=turn(),
        error_count=0,
        agent_output=satisfied(),
        section_states=sections_with(career_goal=SectionStatus.IN_PROGRESS),
    )

    update = await memory_updater_node(state, {})

    assert update["error_count"] == 1, "both failures, still one increment"
    # Neither reason is lost.
    assert "rate limited" in update["last_error"]
    assert "persistence failed" in update["last_error"]


@pytest.mark.asyncio
async def test_successful_persistence_does_not_hide_an_extraction_error(
    extraction_chain, make_state, persistence
):
    """The documented precedence rule: reasons are joined, never replaced."""
    extraction_chain.decision = None
    extraction_chain.extracted = None
    extraction_chain.parsing_error = ValueError("schema refusal")
    state = make_state(
        messages=turn(),
        agent_output=satisfied(),
        section_states=sections_with(career_goal=SectionStatus.IN_PROGRESS),
    )

    update = await memory_updater_node(state, {})

    assert persistence.call_count == 1, "the write still happened"
    assert "unparseable extraction output" in update["last_error"]
    assert update["error_count"] == 1


@pytest.mark.asyncio
async def test_clean_turn_records_no_error(extraction_chain, make_state, persistence):
    extraction_chain.extracted = career_goal_extract(target_roles=["SRE"])
    update = await memory_updater_node(make_state(messages=turn(), error_count=3), {})

    assert "last_error" not in update
    assert "error_count" not in update, "a clean turn must not touch the counter"


@pytest.mark.asyncio
async def test_last_error_names_the_section_without_leaking_secrets(
    extraction_chain, make_state, persistence
):
    extraction_chain.extracted = extract_for(SectionID.BACKGROUND, current_role="SRE")
    persistence.fail = True

    update = await memory_updater_node(
        make_state(section=SectionID.BACKGROUND, messages=turn()), {}
    )

    message = update["last_error"]
    assert "background" in message, "the reason should identify what failed"
    for leak in ("sb_secret", "sb_publishable", "eyJ", "supabase.co", "postgres://"):
        assert leak not in message


# --------------------------------------------------------------------------
# Never raises
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persistence_exception_never_escapes(
    extraction_chain, make_state, monkeypatch
):
    """persist_section is contracted not to raise; if it ever does, absorb it."""
    from agents.xbuddy.nodes import memory_updater as module

    async def explode(*args, **kwargs):
        raise RuntimeError("unexpected boom")

    monkeypatch.setattr(module, "persist_section", explode)
    extraction_chain.extracted = career_goal_extract(target_roles=["SRE"])

    update = await memory_updater_node(make_state(messages=turn(), error_count=0), {})

    assert "unexpected boom" in update["last_error"]
    assert update["error_count"] == 1
    # The extraction result is still kept.
    assert update["user_data"].target_roles == ["SRE"]


@pytest.mark.asyncio
async def test_node_still_returns_no_messages_or_navigation(
    extraction_chain, make_state, persistence
):
    extraction_chain.extracted = career_goal_extract(target_roles=["SRE"])
    persistence.fail = True
    update = await memory_updater_node(make_state(messages=turn()), {})

    for key in ("messages", "current_section", "router_directive", "finished"):
        assert key not in update


# --------------------------------------------------------------------------
# Stage 2/3 semantics preserved
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_packet_still_skips_progress_and_persistence(
    make_state, persistence
):
    update = await memory_updater_node(
        make_state(with_packet=False, messages=turn(), agent_output=satisfied()), {}
    )

    assert persistence.call_count == 0
    assert "section_states" not in update
    assert "persistence_pending" not in update
    assert update["error_count"] == 1


@pytest.mark.asyncio
async def test_empty_history_is_still_a_no_op_but_persists_progress(
    make_state, persistence
):
    """No extraction, no error — yet a confirmed section still reaches the record."""
    state = make_state(
        messages=[],
        error_count=0,
        agent_output=satisfied(),
        section_states=sections_with(career_goal=SectionStatus.IN_PROGRESS),
    )

    update = await memory_updater_node(state, {})

    assert update["section_states"]["career_goal"].status is SectionStatus.DONE
    assert persistence.sections == ["career_goal"]
    assert "last_error" not in update
    assert "error_count" not in update


@pytest.mark.asyncio
async def test_state_field_is_declared_and_defaulted():
    from agents.xbuddy.models import XBuddyState
    from agents.xbuddy.state_factory import COLD_DEFAULTS, build_initial_state

    assert "persistence_pending" in XBuddyState.__optional_keys__
    assert "persistence_pending" in COLD_DEFAULTS
    assert build_initial_state(user_id=1, thread_id="t")["persistence_pending"] == []
    # Fresh list per call — no shared mutable default.
    first = build_initial_state(user_id=1, thread_id="t")["persistence_pending"]
    second = build_initial_state(user_id=1, thread_id="t")["persistence_pending"]
    assert first is not second
