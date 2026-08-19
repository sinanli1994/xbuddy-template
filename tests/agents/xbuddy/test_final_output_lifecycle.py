"""PR 5 Stage 4: the invalidate / reopen / regenerate lifecycle.

`final_output is None` is the entire representation of "no valid artifact" — there
is no stale flag or version. So the lifecycle is:

    generated  --real source change-->  retired + section reopened
               <--reconfirm + regenerate--

The invalidation trigger is `merge_extraction` + `extraction_changed`, a value
comparison. Nothing here depends on the model's advisory `should_save_content`, and
expressing intent to modify is not enough on its own.

All offline: extraction and synthesis are faked, and the autouse `persistence`
fixture in conftest keeps Supabase out of reach.
"""


import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy import synthesis as synthesis_module
from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import (
    ActionAnnotation,
    CareerGoalExtract,
    ChatAgentOutput,
    FinalOutputDraft,
    SectionState,
    XBuddyData,
)
from agents.xbuddy.nodes.implementation import (
    FINAL_OUTPUT_READY_MESSAGE,
    implementation_node,
)
from agents.xbuddy.nodes.memory_updater import memory_updater_node

PLAN = ["Rewrite the CV summary", "Message three colleagues", "Ship a K8s project"]

EXISTING_ARTIFACT = "# QA Analyst to Senior SRE\n\n## Your Action Plan\n1. **stale**\n"


def extract_for(section: SectionID, **overrides):
    """Build the extract model matching `section`, all fields null by default.

    Section-aware on purpose: memory_updater checks `isinstance(parsed, extract_model)`,
    so handing a CareerGoalExtract to a Background turn reads as a parsing failure —
    which would silently turn an invalidation test into a no-op-extraction test.
    """
    from agents.xbuddy.models import EXTRACT_MODELS

    model = EXTRACT_MODELS[section]
    values = dict.fromkeys(model.model_fields, None)
    values.update(overrides)
    return model(**values)


def complete_data(**overrides) -> XBuddyData:
    values = {
        "target_roles": ["Senior SRE"],
        "career_goal_summary": "More autonomy",
        "target_timeline": "within 3 months",
        "current_role": "QA Analyst",
        "years_experience": 4,
        "action_items": list(PLAN),
    }
    values.update(overrides)
    return XBuddyData(**values)


def all_done() -> dict:
    return {
        section.value: SectionState(section_id=section, status=SectionStatus.DONE)
        for section in SectionID
    }


def satisfied(value) -> ChatAgentOutput:
    return ChatAgentOutput(
        reply="So: Senior SRE within 3 months. Right?",
        router_directive="stay",
        is_satisfied=value,
        user_satisfaction_feedback=None,
        should_save_content=False,
    )


def completed_state(make_state, *, section=SectionID.CAREER_GOAL, **overrides) -> dict:
    """A finished thread that already holds a valid artifact."""
    state = make_state(
        section=section,
        messages=[HumanMessage(content="actually, make that six months")],
        user_data=complete_data(),
        section_states=all_done(),
        final_output=EXISTING_ARTIFACT,
        should_generate_final_output=True,
        agent_output=satisfied(None),
    )
    state.update(overrides)
    return state


# --------------------------------------------------------------------------
# Invalidation: a real source change retires the artifact
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_real_career_goal_correction_invalidates_the_artifact(
    extraction_chain, make_state
):
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline="within 6 months"
    )

    update = await memory_updater_node(completed_state(make_state), {})

    assert update["final_output"] is None, "a changed source value retires the document"
    assert update["user_data"].target_timeline == "within 6 months"


@pytest.mark.asyncio
async def test_a_real_background_correction_invalidates_the_artifact(
    extraction_chain, make_state
):
    extraction_chain.extracted = extract_for(SectionID.BACKGROUND, years_experience=6)

    update = await memory_updater_node(
        completed_state(make_state, section=SectionID.BACKGROUND), {}
    )

    assert update["final_output"] is None
    assert update["user_data"].years_experience == 6


@pytest.mark.asyncio
async def test_a_changed_confirmed_action_plan_invalidates_the_artifact(
    extraction_chain, make_state
):
    """No special case: `action_items` is an XBuddyData field like any other."""
    revised = ["Rewrite the CV summary", "Take the CKA exam"]
    extraction_chain.extracted = extract_for(SectionID.ACTION_PLAN, action_items=revised)

    update = await memory_updater_node(
        completed_state(make_state, section=SectionID.ACTION_PLAN), {}
    )

    assert update["final_output"] is None
    assert update["user_data"].action_items == revised


@pytest.mark.asyncio
async def test_invalidation_reopens_the_changed_section(extraction_chain, make_state):
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline="within 6 months"
    )

    update = await memory_updater_node(completed_state(make_state), {})

    assert update["section_states"]["career_goal"].status is SectionStatus.IN_PROGRESS
    for section in SectionID:
        if section is not SectionID.CAREER_GOAL:
            assert update["section_states"][section.value].status is SectionStatus.DONE


@pytest.mark.asyncio
async def test_invalidation_clears_the_completion_flag(extraction_chain, make_state):
    """Otherwise the thread keeps routing into implementation and logs an error
    every turn until the section is reconfirmed."""
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline="within 6 months"
    )

    update = await memory_updater_node(completed_state(make_state), {})

    assert update["should_generate_final_output"] is False


@pytest.mark.asyncio
async def test_invalidation_appends_no_readiness_message(extraction_chain, make_state):
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline="within 6 months"
    )

    update = await memory_updater_node(completed_state(make_state), {})

    assert "messages" not in update


@pytest.mark.asyncio
async def test_an_incomplete_thread_does_not_synthesize_after_invalidation(
    extraction_chain, make_state, monkeypatch
):
    """The reopened section must be finished again before a replacement exists."""
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline="within 6 months"
    )
    calls = {"n": 0}

    class Counting:
        async def ainvoke(self, *args, **kwargs):
            calls["n"] += 1
            raise AssertionError("synthesis must not run while a section is open")

    monkeypatch.setattr(synthesis_module, "_synthesis_chain", lambda: Counting())

    update = await memory_updater_node(completed_state(make_state), {})
    reopened = {
        **completed_state(make_state),
        "final_output": update["final_output"],
        "section_states": update["section_states"],
    }

    result = await implementation_node(reopened, {})

    assert calls["n"] == 0
    assert "final_output" not in result
    assert "career_goal" in result["last_error"]


@pytest.mark.asyncio
async def test_the_reopened_status_is_what_gets_persisted(
    extraction_chain, make_state, persistence
):
    """Invalidation runs before persistence, so the durable row reflects the reopening.

    If the pre-demotion mapping were handed to `_persist`, the stored row would still
    read `done` while state read `in_progress` — the frontend would show a completed
    section that the agent is actively reworking.
    """
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline="within 6 months"
    )

    await memory_updater_node(completed_state(make_state), {})

    career_goal_writes = [
        call for call in persistence.calls if call["section_id"] == "career_goal"
    ]
    assert career_goal_writes, "the reopened section must be written"
    assert career_goal_writes[-1]["status"] == "in_progress"


# --------------------------------------------------------------------------
# No-op revisits preserve the artifact
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_revisit_with_no_extracted_change_preserves_the_artifact(
    extraction_chain, make_state
):
    """The document is still accurate, so regenerating it would buy nothing."""
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline=None
    )

    update = await memory_updater_node(completed_state(make_state), {})

    assert "final_output" not in update, "no change means the document still holds"
    assert "section_states" not in update, "nothing reopened"
    # `_section_progress` re-emits the flag as True on any all-done turn, which is
    # pre-existing idempotent behaviour. What matters is that nothing set it False.
    assert update.get("should_generate_final_output") is not False


@pytest.mark.asyncio
async def test_re_stating_an_identical_value_is_not_a_change(extraction_chain, make_state):
    """`merge_extraction` is idempotent, so an echoed value is a no-op."""
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline="within 3 months"
    )

    update = await memory_updater_node(completed_state(make_state), {})

    assert "final_output" not in update


@pytest.mark.asyncio
async def test_a_failed_extraction_does_not_invalidate(extraction_chain, make_state):
    """A model failure is not evidence that the source data moved."""
    extraction_chain.extracted = None  # parsing_error path

    update = await memory_updater_node(completed_state(make_state), {})

    assert "final_output" not in update
    assert update["last_error"]


@pytest.mark.asyncio
async def test_modify_intent_alone_does_not_invalidate(extraction_chain, make_state):
    """Routing said modify, extraction found nothing. Intent is not a change."""
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline=None
    )
    state = completed_state(make_state, router_directive="modify:career_goal")

    update = await memory_updater_node(state, {})

    assert "final_output" not in update


@pytest.mark.asyncio
@pytest.mark.parametrize("advisory", [True, False])
async def test_invalidation_ignores_the_advisory_should_save_content_flag(
    extraction_chain, make_state, advisory
):
    """Detection is a value comparison, never a model opinion.

    Same deterministic outcome whichever way the decision model set the flag.
    """
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline="within 6 months"
    )
    output = ChatAgentOutput(
        reply="ok",
        router_directive="stay",
        is_satisfied=None,
        user_satisfaction_feedback=None,
        should_save_content=advisory,
    )

    update = await memory_updater_node(
        completed_state(make_state, agent_output=output), {}
    )

    assert update["final_output"] is None


@pytest.mark.asyncio
async def test_a_correction_confirmed_in_the_same_turn_keeps_the_section_done(
    extraction_chain, make_state
):
    """Correcting and confirming together: the confirmation is about the corrected
    summary the user just saw, so promotion wins and the artifact regenerates now."""
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline="within 6 months"
    )

    update = await memory_updater_node(
        completed_state(make_state, agent_output=satisfied(True)), {}
    )

    assert update["final_output"] is None, "the old document is still retired"
    assert "section_states" not in update, "nothing demoted; it was reconfirmed"
    assert update.get("should_generate_final_output") is True


# --------------------------------------------------------------------------
# Regeneration
# --------------------------------------------------------------------------


class CountingSynthesis:
    """Fake synthesis whose headline can change between runs."""

    def __init__(self, headline="QA Analyst to Senior SRE"):
        self.headline = headline
        self.calls = 0

    async def ainvoke(self, messages, *args, **kwargs):
        self.calls += 1
        return {
            "parsed": FinalOutputDraft(
                headline=self.headline,
                positioning_summary="Four years of QA moving into automation.",
                strengths_to_leverage=["systems debugging"],
                skill_priorities=["Kubernetes"],
                search_targets=["fintech"],
                action_annotations=[
                    ActionAnnotation(step_number=n, rationale=f"reason {n}", timeframe=None)
                    for n in range(1, len(PLAN) + 1)
                ],
                risks_or_constraints=[],
            ),
            "parsing_error": None,
        }


@pytest.mark.asyncio
async def test_a_reconfirmed_section_regenerates_exactly_once(monkeypatch, make_state):
    chain = CountingSynthesis()
    monkeypatch.setattr(synthesis_module, "_synthesis_chain", lambda: chain)

    reopened = make_state(
        messages=[HumanMessage(content="yes, six months is right")],
        user_data=complete_data(target_timeline="within 6 months"),
        section_states=all_done(),
        final_output=None,
        should_generate_final_output=True,
    )

    first = await implementation_node(reopened, {})
    assert chain.calls == 1
    assert first["final_output"]

    again = await implementation_node(
        {**reopened, "final_output": first["final_output"]}, {}
    )

    assert again == {}
    assert chain.calls == 1, "later turns must not regenerate"


@pytest.mark.asyncio
async def test_the_regenerated_artifact_reflects_the_new_synthesis(monkeypatch, make_state):
    chain = CountingSynthesis(headline="QA Analyst to Platform Engineer")
    monkeypatch.setattr(synthesis_module, "_synthesis_chain", lambda: chain)

    update = await implementation_node(
        make_state(
            user_data=complete_data(),
            section_states=all_done(),
            final_output=None,
            should_generate_final_output=True,
        ),
        {},
    )

    assert update["final_output"].startswith("# QA Analyst to Platform Engineer")
    assert update["final_output"] != EXISTING_ARTIFACT


@pytest.mark.asyncio
async def test_regeneration_appends_a_second_readiness_message(monkeypatch, make_state):
    """Intentional: it announces a *new* valid artifact."""
    chain = CountingSynthesis()
    monkeypatch.setattr(synthesis_module, "_synthesis_chain", lambda: chain)

    update = await implementation_node(
        make_state(
            user_data=complete_data(),
            section_states=all_done(),
            final_output=None,
            should_generate_final_output=True,
        ),
        {},
    )

    assert len(update["messages"]) == 1
    assert update["messages"][0].content == FINAL_OUTPUT_READY_MESSAGE


# --------------------------------------------------------------------------
# The whole lifecycle, end to end through the node pair
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exactly_two_readiness_messages_across_generate_correct_regenerate(
    extraction_chain, make_state, monkeypatch
):
    """generate -> correct -> reopen -> reconfirm -> regenerate.

    Counts readiness announcements across the whole sequence: one per valid artifact,
    so two, with none emitted by the invalidating turn in between.
    """
    chain = CountingSynthesis()
    monkeypatch.setattr(synthesis_module, "_synthesis_chain", lambda: chain)
    announcements: list[AIMessage] = []

    def collect(update):
        announcements.extend(
            message
            for message in update.get("messages", [])
            if isinstance(message, AIMessage)
            and message.content == FINAL_OUTPUT_READY_MESSAGE
        )

    # 1. Initial generation.
    base = make_state(
        user_data=complete_data(),
        section_states=all_done(),
        final_output=None,
        should_generate_final_output=True,
    )
    generated = await implementation_node(base, {})
    collect(generated)
    assert chain.calls == 1

    # 2. A real correction retires it and reopens Career Goal.
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline="within 6 months"
    )
    invalidated = await memory_updater_node(
        completed_state(make_state, final_output=generated["final_output"]), {}
    )
    collect(invalidated)
    assert invalidated["final_output"] is None
    assert invalidated["should_generate_final_output"] is False

    # 3. Still open: no replacement yet.
    blocked = await implementation_node(
        {
            **base,
            "final_output": None,
            "section_states": invalidated["section_states"],
        },
        {},
    )
    collect(blocked)
    assert chain.calls == 1, "no synthesis while the section is open"

    # 4. Reconfirmed -> all five DONE again -> regenerate once.
    regenerated = await implementation_node(
        {
            **base,
            "final_output": None,
            "user_data": invalidated["user_data"],
            "section_states": all_done(),
        },
        {},
    )
    collect(regenerated)

    assert chain.calls == 2, "exactly one regeneration"
    assert regenerated["final_output"]
    assert len(announcements) == 2, "one announcement per valid artifact"


@pytest.mark.asyncio
async def test_the_lifecycle_needs_no_stale_flag(make_state):
    """`final_output is None` carries the whole meaning of "no valid artifact"."""
    from agents.xbuddy.models import XBuddyState

    assert "final_output_stale" not in XBuddyState.__annotations__
    assert "final_output_version" not in XBuddyState.__annotations__
