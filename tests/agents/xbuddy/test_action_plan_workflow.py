"""Real compiled graph + service regressions. Only model/remote writes are fake.

No prompt-sensitive canned prose: the fake returns structured actions. Production
state routing, validation, rendering, checkpointing and SSE decide what is shown.
"""

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy.action_plan import ActionPlanDraft, ProposedAction
from agents.xbuddy.enums import DecisionAction, SectionID, SectionStatus
from agents.xbuddy.models import SectionState, XBuddyData


@pytest.fixture
def journey(monkeypatch, make_decision):
    from agents.xbuddy import action_plan
    from agents.xbuddy.graph import builder
    from agents.xbuddy.nodes import generate_decision, generate_reply, memory_updater

    def no_live(*args, **kwargs):
        raise AssertionError("No live model calls allowed in this test")

    monkeypatch.setattr("core.llm.get_model", no_live)
    data = XBuddyData(
        target_roles=["AI Engineer"], target_timeline="six months",
        career_goal_summary="Move into AI engineering", current_role="Backend Engineer",
        years_experience=2, highest_education="MSc", work_history=["Acme backend, 2023-2025"],
        preferred_locations=["Toronto"], preferred_work_modes=["remote"],
        target_industries=["SaaS"], employment_types=["full-time"], salary_expectation="CAD 100k",
        strengths=["API design"], current_skills=["Python"], skill_gaps=["LangGraph agent design"],
    )
    actions = [
        ProposedAction(heading="Improve Application Materials", action="Tailor the CV to demonstrate Python work for AI Engineer roles.",
                       basis_fields=["target_roles", "current_skills", "current_role"]),
        ProposedAction(heading="Close Key Skill Gaps", action="Build a LangGraph agent project to demonstrate the missing skill.",
                       basis_fields=["skill_gaps", "target_timeline"]),
        ProposedAction(heading="Target Relevant Openings", action="Shortlist remote Toronto openings that match your background.",
                       basis_fields=["preferred_locations", "preferred_work_modes"]),
        ProposedAction(heading="Build Professional Connections", action="Contact SaaS practitioners to review your project and target roles.",
                       basis_fields=["target_industries", "strengths"]),
    ]
    j = SimpleNamespace(data=data, actions=actions, trace=[], reply_states=[], gate=None)
    j.decision = AsyncMock(return_value={"parsed": make_decision(
        action=DecisionAction.NEXT, presented_summary=True, is_satisfied=True,
    ), "parsing_error": None})
    j.generic = AsyncMock(return_value=AIMessage(content="What concrete steps are you planning to take?"))

    async def propose(messages, config=None):
        if j.gate is not None:
            await j.gate.wait()
        return {"parsed": ActionPlanDraft.model_construct(actions=j.actions), "parsing_error": None}

    j.proposal = AsyncMock(side_effect=propose)
    j.extraction_models = []

    def extract(model):
        j.extraction_models.append(model)
        payload = {field: getattr(j.data, field) for field in model.model_fields}
        return SimpleNamespace(ainvoke=AsyncMock(return_value={
            "parsed": model(**payload), "parsing_error": None,
        }))

    monkeypatch.setattr(generate_decision, "_decision_chain", lambda: SimpleNamespace(ainvoke=j.decision))
    monkeypatch.setattr(generate_reply, "_reply_model", lambda: SimpleNamespace(ainvoke=j.generic))
    monkeypatch.setattr(memory_updater, "_extraction_chain", extract)
    monkeypatch.setattr(action_plan, "_proposal_chain", lambda: SimpleNamespace(ainvoke=j.proposal))

    for name in ("process_confirmation_node", "memory_updater_node", "router_node", "generate_reply_node"):
        original = getattr(builder, name)

        async def spy(state, config, original=original, name=name):
            j.trace.append(name)
            if name == "generate_reply_node":
                j.reply_states.append(dict(state))
            return await original(state, config)

        monkeypatch.setattr(builder, name, spy)

    j.graph = builder.build_xbuddy_graph()
    j.config = {"configurable": {"thread_id": str(uuid.uuid4()), "user_id": 7}}
    j.seed = {
        "user_id": 7, "thread_id": j.config["configurable"]["thread_id"],
        "current_section": SectionID.SKILL_ASSESSMENT, "router_directive": "stay",
        "awaiting_satisfaction_feedback": True, "user_data": data,
        "section_states": {section.value: SectionState(
            section_id=section,
            status=SectionStatus.DONE if index < 3 else (
                SectionStatus.IN_PROGRESS if index == 3 else SectionStatus.PENDING
            ),
        ) for index, section in enumerate(SectionID)},
        "messages": [AIMessage(content="Your strengths are API design, your skill is Python, "
                                "and your gap is LangGraph agent design. Is this summary right?")],
    }
    return j


async def confirm(j):
    await j.graph.aupdate_state(j.config, j.seed)
    return await j.graph.ainvoke({"messages": [HumanMessage(content="yes")]}, j.config)


def assert_proposal(j, result):
    state = j.reply_states[-1]
    assert state["current_section"] is SectionID.ACTION_PLAN
    assert state["context_packet"].section_id is SectionID.ACTION_PLAN
    assert state["reply_intent"] == "PROPOSE_FIRST_DRAFT"
    assert state["user_data"] == j.data
    assert state["section_states"]["skill_assessment"].status is SectionStatus.DONE
    assert j.generic.await_count == 0
    assert j.proposal.await_count == 1
    assert j.decision.await_count == 1
    assert [model.__name__ for model in j.extraction_models] == ["SkillAssessmentExtract"]
    # Only the previous summary and this ONE reply are checkpointed as AI messages.
    replies = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(replies) == 2
    reply = replies[-1].content
    assert reply.startswith("### First-Draft Action Plan\n\n")
    for index, action in enumerate(j.actions, 1):
        assert f"**{index}. {action.heading}**\n\n- {action.action}\n" in reply
    assert "Toronto" in reply and "Python" in reply
    assert "Based on your" not in reply
    assert reply.count("?") == 1
    assert "What concrete steps are you planning" not in reply
    assert reply.endswith("Which of these steps feel realistic, and what would you like to adjust?")
    assert result["user_data"].action_items == [], "a proposal is not user consent"
    assert result["section_states"]["action_plan"].status is SectionStatus.IN_PROGRESS


@pytest.mark.asyncio
@pytest.mark.parametrize("has_handshake", [True, False])
async def test_skill_yes_commits_action_plan_then_produces_one_grounded_draft(journey, has_handshake):
    journey.seed["awaiting_satisfaction_feedback"] = has_handshake
    result = await confirm(journey)
    assert_proposal(journey, result)
    assert journey.trace == ["router_node", "process_confirmation_node", "memory_updater_node",
                             "router_node", "generate_reply_node"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["reply_before_transition", "discard_proposal_intent"])
async def test_transition_regression_has_teeth(journey, monkeypatch, mutation):
    from agents.xbuddy.graph import builder, routes

    if mutation == "reply_before_transition":
        monkeypatch.setattr(builder, "route_turn", routes.route_decision)
    else:
        original = builder.router_node

        async def discard(state, config):
            update = dict(await original(state, config))
            update["reply_intent"] = "CONVERSE"
            return update

        monkeypatch.setattr(builder, "router_node", discard)
    journey.graph = builder.build_xbuddy_graph()
    result = await confirm(journey)
    with pytest.raises(AssertionError):
        assert_proposal(journey, result)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["too_few", "question", "unknown_fact", "duplicate", "dense", "debug", "heading"])
async def test_invalid_model_plan_is_not_shown_or_accepted(journey, bad):
    if bad == "too_few":
        journey.actions = journey.actions[:2]
    elif bad == "question":
        journey.actions[0] = ProposedAction.model_construct(
            heading="Create a Plan", action="What concrete steps are you planning to take?", basis_fields=["target_roles"],
        )
    elif bad == "unknown_fact":
        journey.actions[0] = journey.actions[0].model_copy(update={"basis_fields": ["invented_field"]})
    elif bad == "dense":
        journey.actions[0] = journey.actions[0].model_copy(update={"action": "Practice relevant skills " * 20})
    elif bad == "debug":
        journey.actions[0] = journey.actions[0].model_copy(update={"action": "Based on your skills, build a project."})
    elif bad == "heading":
        journey.actions[0] = journey.actions[0].model_copy(update={"heading": "What should you do?"})
    else:
        journey.actions[1] = journey.actions[0]
    result = await confirm(journey)
    assert "couldn't prepare a valid" in result["messages"][-1].content
    assert "What concrete steps" not in result["messages"][-1].content
    assert result["user_data"].action_items == []
    assert not result.get("final_output")
    assert journey.generic.await_count == 0


@pytest.mark.asyncio
async def test_skill_correction_stays_in_skills_and_does_not_propose(journey, make_decision):
    journey.decision.return_value = {"parsed": make_decision(
        action=DecisionAction.STAY, presented_summary=True, is_satisfied=False,
    ), "parsing_error": None}
    result = await confirm(journey)
    assert result["current_section"] is SectionID.SKILL_ASSESSMENT
    assert journey.proposal.await_count == 0
    assert journey.generic.await_count == 1


async def stream_journey(j, monkeypatch, events, reached, predicate=None):
    from service import service

    monkeypatch.setattr(service, "get_agent", lambda _: j.graph)

    async def handle(*args):
        return {"input": {"messages": [HumanMessage(content="yes")]}, "config": j.config}, "test-run"

    monkeypatch.setattr(service, "_handle_input", handle)
    from schema import StreamInput

    async for chunk in service.message_generator(StreamInput(message="yes", user_id=7), "xbuddy"):
        payload = chunk.removeprefix("data: ").strip()
        event = {"type": "DONE"} if payload == "[DONE]" else json.loads(payload)
        events.append(event)
        if event["type"] == "completion" and (
            predicate(event["content"]) if predicate else
            event["content"]["sections"][3]["status"] == "done"
        ):
            reached.set()


@pytest.mark.asyncio
async def test_committed_progress_precedes_blocked_proposal_and_live_equals_history(journey, monkeypatch):
    """Holding the LLM open must not hold back already committed sidebar progress."""
    from service.service import load_chat_history

    j = journey
    j.gate = asyncio.Event()
    await j.graph.aupdate_state(j.config, j.seed)
    events, reached = [], asyncio.Event()
    task = asyncio.create_task(stream_journey(j, monkeypatch, events, reached))
    try:
        await asyncio.wait_for(reached.wait(), timeout=3)
        assert not task.done(), "progress must precede the still-blocked proposal"
        state = await j.graph.aget_state(j.config)
        assert state.values["section_states"]["skill_assessment"].status is SectionStatus.DONE
    finally:
        j.gate.set()
        await asyncio.wait_for(task, timeout=3)
    live = [e["content"]["content"] for e in events if e["type"] == "message"]
    restored = await load_chat_history("xbuddy", j.config["configurable"]["thread_id"], 7)
    assert live == [m.content for m in restored.messages if m.type == "ai"][1:]
    assert len(live) == 1 and "1. " in live[0] and "3. " in live[0]
    progress = [e["content"] for e in events if e["type"] == "completion"]
    assert all(set(p) == {"collection_complete", "artifact_available", "sections"} for p in progress)
    assert all(len(p["sections"]) == 5 for p in progress)
    assert events[-1]["type"] == "DONE"


@pytest.mark.asyncio
async def test_final_collection_progress_precedes_synthesis_and_ready_matches_history(journey, monkeypatch):
    from agents.xbuddy.models import ActionItem, FinalOutput
    from agents.xbuddy.nodes import implementation
    from service.service import load_chat_history

    j = journey
    steps = [action.action for action in j.actions]
    j.data = j.data.model_copy(update={"action_items": steps})
    j.seed.update(user_data=j.data, current_section=SectionID.ACTION_PLAN)
    j.seed["section_states"]["skill_assessment"].status = SectionStatus.DONE
    j.seed["section_states"]["action_plan"].status = SectionStatus.IN_PROGRESS
    j.seed["messages"] = [AIMessage(content="\n".join(steps) + "\nDoes this plan look right?")]
    gate = asyncio.Event()

    async def synthesize(data):
        await gate.wait()
        return FinalOutput(
            headline="Your job search strategy", positioning_summary="Build on your Python experience.",
            strengths_to_leverage=data.strengths, skill_priorities=data.skill_gaps,
            search_targets=data.preferred_locations,
            action_items=[ActionItem(step=step, rationale="Uses the collected skills.", priority=i,
                                     timeframe=None) for i, step in enumerate(steps, 1)],
            risks_or_constraints=[], unknowns=[],
        ), None

    monkeypatch.setattr(implementation, "synthesize_final_output", synthesize)
    await j.graph.aupdate_state(j.config, j.seed)
    events, reached = [], asyncio.Event()
    task = asyncio.create_task(stream_journey(
        j, monkeypatch, events, reached,
        lambda p: p["collection_complete"] and not p["artifact_available"],
    ))
    try:
        await asyncio.wait_for(reached.wait(), 3)
        assert not task.done()
        assert not any(e["type"] == "message" for e in events)
    finally:
        gate.set()
        await asyncio.wait_for(task, 3)
    live = [e["content"]["content"] for e in events if e["type"] == "message"]
    assert live == [implementation.FINAL_OUTPUT_READY_MESSAGE]
    assert "edit" not in live[0].lower()
    restored = await load_chat_history("xbuddy", j.config["configurable"]["thread_id"], 7)
    assert live == [m.content for m in restored.messages if m.type == "ai"][1:]
    assert events[-2]["type"] == "completion"
    assert events[-2]["content"]["artifact_available"] is True
    assert events[-1]["type"] == "DONE"


@pytest.mark.asyncio
async def test_new_confirmation_id_is_processed_on_the_next_user_turn(journey):
    result = await confirm(journey)
    first_id = result["confirmation_processed_id"]
    assert result["awaiting_satisfaction_feedback"] is True
    # A rejection is still processed rather than bypassed by the previous input id.
    journey.decision.return_value["parsed"] = journey.decision.return_value["parsed"].model_copy(
        update={"action": DecisionAction.STAY, "is_satisfied": False},
    )
    result = await journey.graph.ainvoke({"messages": [HumanMessage(content="Please adjust it")]}, journey.config)
    assert journey.decision.await_count == 2
    assert result["confirmation_processed_id"] != first_id


@pytest.mark.asyncio
async def test_legacy_proposal_confirmation_without_extracted_steps_stays_open(journey):
    await confirm(journey)
    # Legacy checkpoints did not save a structured proposal. Their plain prose
    # still needs extraction; missing steps must not be mistaken for completion.
    await journey.graph.aupdate_state(journey.config, {"pending_action_plan": None})
    # The extraction fake returns no agreed steps, as on a failed/no-op extraction.
    result = await journey.graph.ainvoke({"messages": [HumanMessage(content="yes")]}, journey.config)
    assert result["section_states"]["action_plan"].status is SectionStatus.IN_PROGRESS
    assert result["finished"] is False
    assert not result["should_generate_final_output"]
    assert not result.get("final_output")
    assert len([m for m in result["messages"] if isinstance(m, AIMessage)]) == 3


@pytest.mark.asyncio
async def test_proposal_acceptance_failure_and_retry_each_have_one_reply(journey, monkeypatch):
    from agents.xbuddy.models import ActionItem, FinalOutput
    from agents.xbuddy.nodes import implementation

    j = journey
    await confirm(j)
    # The extraction model now recognizes user agreement to the proposed steps.
    j.data = j.data.model_copy(update={"action_items": [a.action for a in j.actions]})
    output = FinalOutput(
        headline="Your job search strategy", positioning_summary="Use Python experience.",
        strengths_to_leverage=j.data.strengths, skill_priorities=j.data.skill_gaps,
        search_targets=j.data.preferred_locations,
        action_items=[ActionItem(step=step, rationale="Uses collected skills.", priority=i,
                                 timeframe=None) for i, step in enumerate(j.data.action_items, 1)],
        risks_or_constraints=[], unknowns=[],
    )
    synthesize = AsyncMock(side_effect=[(None, "temporary failure"), (output, None)])
    monkeypatch.setattr(implementation, "synthesize_final_output", synthesize)
    result = await j.graph.ainvoke({"messages": [HumanMessage(content="yes")]}, j.config)
    assert len([m for m in result["messages"] if isinstance(m, AIMessage)]) == 3
    assert "couldn't finish" in result["messages"][-1].content
    assert not result.get("final_output")
    result = await j.graph.ainvoke({"messages": [HumanMessage(content="Please retry")]}, j.config)
    assert len([m for m in result["messages"] if isinstance(m, AIMessage)]) == 4
    assert result["messages"][-1].content == implementation.FINAL_OUTPUT_READY_MESSAGE
    assert result["final_output"]
    assert j.generic.await_count == 0
    assert synthesize.await_count == 2
    # Finished threads still answer new messages once; the existing document
    # is not synthesized again and the confirmation id cannot suppress replies.
    result = await j.graph.ainvoke({"messages": [HumanMessage(content="Thank you")]}, j.config)
    assert len([m for m in result["messages"] if isinstance(m, AIMessage)]) == 5
    assert j.generic.await_count == 1
    assert synthesize.await_count == 2
    # A later correction cannot keep an incomplete two-step plan marked Done.
    j.data = j.data.model_copy(update={"action_items": j.data.action_items[:2]})
    result = await j.graph.ainvoke({"messages": [HumanMessage(content="Use just those two steps")]}, j.config)
    assert result["section_states"]["action_plan"].status is SectionStatus.IN_PROGRESS
    assert result["finished"] is False
    assert not result.get("final_output")
    assert synthesize.await_count == 2


@pytest.mark.asyncio
async def test_progress_and_proposal_survive_real_disk_checkpoint_restart(journey, monkeypatch, tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = str(tmp_path / "progress.db")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await saver.setup()
        journey.graph.checkpointer = saver
        await test_committed_progress_precedes_blocked_proposal_and_live_equals_history(journey, monkeypatch)
    # Fresh saver, not the in-process snapshot: the emitted projection really was saved.
    async with AsyncSqliteSaver.from_conn_string(path) as restored:
        journey.graph.checkpointer = restored
        snapshot = await journey.graph.aget_state(journey.config)
        assert snapshot.values["section_states"]["skill_assessment"].status is SectionStatus.DONE
        assert snapshot.values["section_states"]["action_plan"].status is SectionStatus.IN_PROGRESS
        assert "3. " in snapshot.values["messages"][-1].content
        assert snapshot.values["reply_intent"] == "PROPOSE_FIRST_DRAFT"


@pytest.mark.asyncio
@pytest.mark.parametrize("synthesis_fails", [False, True])
@pytest.mark.parametrize("confirmation_action", [DecisionAction.NEXT, DecisionAction.STAY])
@pytest.mark.parametrize("same_role", [False, True])
async def test_confirm_proposal_through_real_service_synthesis_and_disk_restore(
    journey, monkeypatch, tmp_path, final_output_persistence, synthesis_fails, confirmation_action, same_role,
):
    """Start with NO agreed steps; do not seed a successful extraction/artifact.

    Actual service input handling, graph, synthesis assembly, Markdown rendering,
    checkpoint writes, public projections and history/final-output reads run.
    Only the model boundary and remote durable writer are replaced.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from agents.xbuddy import synthesis
    from agents.xbuddy.models import ActionAnnotation, FinalOutputDraft
    from agents.xbuddy.nodes.implementation import FINAL_OUTPUT_READY_MESSAGE
    from schema import StreamInput
    from service import service

    j = journey
    path = str(tmp_path / "final-confirmation.db")
    if same_role:
        j.data = j.data.model_copy(update={
            "current_role": "AI Engineer",
            "target_roles": ["AI Engineer focused on LLM applications and RAG systems"],
            "career_goal_summary": "Specialize in LLM applications and RAG systems",
        })
        j.seed["user_data"] = j.data
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await saver.setup()
        j.graph.checkpointer = saver
        proposed = await confirm(j)
        assert proposed["user_data"].action_items == []
        assert proposed["pending_action_plan"].message_id == proposed["messages"][-1].id
        assert not proposed["should_generate_final_output"]

    # Satisfaction owns completion, including the existing satisfied-STAY case.
    j.decision.return_value["parsed"] = j.decision.return_value["parsed"].model_copy(
        update={"action": confirmation_action},
    )

    gate, collected = asyncio.Event(), asyncio.Event()
    draft = FinalOutputDraft(
        positioning_summary="Build on Python and API design.",
        strengths_to_leverage=j.data.strengths, skill_priorities=j.data.skill_gaps,
        search_targets=j.data.preferred_locations,
        action_annotations=[ActionAnnotation(step_number=i, rationale="Based on collected skills.",
                                             timeframe=None) for i in range(1, len(j.actions) + 1)],
        risks_or_constraints=[],
    )

    async def model(messages):
        await gate.wait()
        if synthesis_fails:
            raise RuntimeError("deterministic synthesis outage")
        return {"parsed": draft, "parsing_error": None}

    model_call = AsyncMock(side_effect=model)
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: SimpleNamespace(ainvoke=model_call))
    monkeypatch.setattr(service, "get_agent", lambda _: j.graph)
    monkeypatch.setattr(service.settings, "LANGFUSE_TRACING", False)
    events = []
    thread_id = j.config["configurable"]["thread_id"]

    async def consume():
        async for frame in service.message_generator(
            StreamInput(message="looks good", thread_id=thread_id, user_id=7), "xbuddy",
        ):
            raw = frame.removeprefix("data: ").strip()
            event = {"type": "DONE"} if raw == "[DONE]" else json.loads(raw)
            events.append(event)
            if event["type"] == "completion" and event["content"]["collection_complete"]:
                collected.set()

    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        j.graph.checkpointer = saver
        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(collected.wait(), 5)
            assert not task.done(), "early completion must not terminate the turn"
            assert not any(event["type"] == "DONE" for event in events)
            snapshot = await j.graph.aget_state(j.config)
            assert snapshot.values["user_data"].action_items == [a.action for a in j.actions]
            assert snapshot.values["should_generate_final_output"] is True
            assert snapshot.values["final_output"] is None
        finally:
            gate.set()
            await asyncio.wait_for(task, 5)

    # A NEW connection reads what really survived disk persistence.
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        j.graph.checkpointer = saver
        values = (await j.graph.aget_state(j.config)).values
        assert all(s.status is SectionStatus.DONE for s in values["section_states"].values())
        assert values["should_generate_final_output"] is True
        assert values["pending_action_plan"] is None
        completion = await service.load_completion_state("xbuddy", thread_id, 7)
        artifact = await service.load_final_output("xbuddy", thread_id, 7)
        history = await service.load_chat_history("xbuddy", thread_id, 7)
        assert completion.collection_complete is True
        assert len(completion.sections) == 5
        assert all(s.status == "done" for s in completion.sections)
        assert events[-1]["type"] == "DONE"
        assert events[-2] == {"type": "completion", "content": completion.model_dump()}
        live = [e["content"]["content"] for e in events if e["type"] == "message"]
        assert live == [history.messages[-1].content]
        assert len(live) == 1
        if synthesis_fails:
            assert values["final_output"] is None
            assert not completion.artifact_available and not artifact.artifact_available
            assert "synthesis call failed" in values["last_error"]
            assert "couldn't finish" in live[0]
            assert final_output_persistence.call_count == 0
        else:
            assert values["final_output"] is not None
            assert completion.artifact_available and artifact.artifact_available
            assert artifact.final_output == values["final_output"]
            assert live == [FINAL_OUTPUT_READY_MESSAGE]
            assert final_output_persistence.markdowns == [values["final_output"]]
            assert all(a.action in values["final_output"] for a in j.actions)
            expected_title = ("AI Engineer Career Plan: LLM applications and RAG systems" if same_role
                              else "Transition from Backend Engineer to AI Engineer")
            assert values["final_output"].startswith(f"# {expected_title}\n")
        assert model_call.await_count == 1
        assert j.generic.await_count == 0
        assert [m.__name__ for m in j.extraction_models] == ["SkillAssessmentExtract"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", [True, False])
async def test_pending_proposal_requires_current_message_and_actual_consent(journey, stale, make_decision):
    j = journey
    await confirm(j)
    if stale:
        await j.graph.aupdate_state(j.config, {"messages": [AIMessage(content="A different draft.")]})
    else:
        j.decision.return_value = {"parsed": make_decision(
            action=DecisionAction.STAY, presented_summary=True, is_satisfied=False,
        ), "parsing_error": None}
    result = await j.graph.ainvoke({"messages": [HumanMessage(content="Please change it")]}, j.config)
    assert result["user_data"].action_items == []
    assert result["final_output"] is None
    assert not result["should_generate_final_output"]
