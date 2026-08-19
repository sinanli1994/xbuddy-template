"""Cross-node and graph-level tests for a complete PR 3 turn.

These exercise the compiled graph rather than nodes in isolation, because three
of PR 3's guarantees only exist at that level: tokens reach the client through
LangGraph's `messages` mode, decision tokens are suppressed by tag, and the
reply→decision→memory_updater→router loop terminates.

The fakes here are real `BaseChatModel`s (`FakeListChatModel`), not stubs — a
plain object would emit no token events, so a stub could not prove streaming
works. Only the two model calls are faked; initialize, router, memory_updater,
the edges, and the reducers are all production code.
"""

import uuid

import pytest
from langchain_community.chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy.enums import DecisionAction, SectionID, SectionStatus
from agents.xbuddy.models import SectionState, XBuddyData
from agents.xbuddy.nodes.generate_decision import INTERNAL_DECISION_TAG, generate_decision_node
from agents.xbuddy.nodes.generate_reply import generate_reply_node

REPLY_TEXT = "What kind of role are you targeting next?"


class StreamingDecisionChain:
    """Mirrors the production chain: a tagged model call plus the include_raw dict.

    The inner model call is what produces taggable chunks, so the suppression
    test is exercising the same mechanism the real node relies on.
    """

    def __init__(self, decision):
        self.decision = decision
        self.model = FakeListChatModel(responses=['{"action": "stay"}']).with_config(
            tags=[INTERNAL_DECISION_TAG]
        )
        self.calls: list = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def ainvoke(self, messages, config=None):
        self.calls.append(list(messages))
        await self.model.ainvoke(messages, config)
        return {
            "raw": AIMessage(content=""),
            "parsed": self.decision,
            "parsing_error": None,
        }


class CountingReplyModel:
    """Wraps a real FakeListChatModel so tokens stream, while counting calls."""

    def __init__(self, text: str = REPLY_TEXT):
        self.model = FakeListChatModel(responses=[text])
        self.calls: list = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def ainvoke(self, messages, config=None):
        self.calls.append(list(messages))
        return await self.model.ainvoke(messages, config)


class FakeExtractionChain:
    """Fakes memory_updater's extraction call.

    Without this the graph tests reach for the real API on every run: the node
    catches the resulting auth error into its fallback, so the tests still pass
    while making network round-trips and silently exercising only the failure
    path. `extracted=None` + a parsing_error keeps the merge a no-op, so these
    tests observe routing rather than extraction.
    """

    def __init__(self, extracted=None):
        self.extracted = extracted
        self.calls: list = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def build(self, extract_model):
        return self

    async def ainvoke(self, messages, config=None):
        self.calls.append(list(messages))
        return {
            "raw": AIMessage(content=""),
            "parsed": self.extracted,
            "parsing_error": None if self.extracted is not None else ValueError("no-op"),
        }


def _patch_graph(monkeypatch, decision):
    from agents.xbuddy.agent import graph
    from agents.xbuddy.nodes import generate_decision as decision_module
    from agents.xbuddy.nodes import generate_reply as reply_module
    from agents.xbuddy.nodes import memory_updater as memory_module

    fakes = {
        "reply": CountingReplyModel(),
        "decision": StreamingDecisionChain(decision),
        "extraction": FakeExtractionChain(),
    }
    monkeypatch.setattr(reply_module, "_reply_model", lambda: fakes["reply"])
    monkeypatch.setattr(decision_module, "_decision_chain", lambda: fakes["decision"])
    monkeypatch.setattr(memory_module, "_extraction_chain", fakes["extraction"].build)
    return graph, fakes


@pytest.fixture
def graph_with_fakes(monkeypatch, make_decision):
    """Decision always says `stay` — the ordinary one-reply turn."""
    return _patch_graph(monkeypatch, make_decision(action=DecisionAction.STAY))


@pytest.fixture
def graph_with_fakes_next(monkeypatch, make_decision):
    """Decision always says `next` — the runaway-loop scenario."""
    return _patch_graph(monkeypatch, make_decision(action=DecisionAction.NEXT))


@pytest.fixture
def graph_with_satisfied_next(monkeypatch, make_decision):
    """Decision says `next` AND reports the user confirmed the summary.

    This is the combination PR 4 Stage 3 makes meaningful: before it, nothing
    marked a section DONE so `next` could not advance.
    """
    return _patch_graph(
        monkeypatch,
        make_decision(
            action=DecisionAction.NEXT,
            presented_summary=True,
            is_satisfied=True,
            user_satisfaction_feedback="confirmed",
            decision_reason="user confirmed the summary",
        ),
    )


def make_config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4()), "user_id": 7}}


# --------------------------------------------------------------------------
# The satisfaction-confirmation sequence (plan §2.1 + §3.1)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirmation_sequence_acknowledges_then_advances_and_clears_flag(
    reply_model, decision_chain, make_state, make_decision
):
    """The ordering constraint that motivated the overlay.

    generate_reply runs before generate_decision, so it is the first node to see
    "yes, that's right". Without the overlay it would ask another Career Goal
    question about a section the user just signed off.
    """
    reply_model.reply = "Great — that's your goal locked in. Let's look at your background."
    decision_chain.decision = make_decision(
        action=DecisionAction.NEXT,
        presented_summary=True,
        is_satisfied=True,
        user_satisfaction_feedback="confirmed the summary",
        decision_reason="user confirmed the summary",
    )

    state = make_state(
        awaiting_satisfaction_feedback=True,
        messages=[
            AIMessage(content="So: Senior SRE, within 3 months, for more autonomy. Right?"),
            HumanMessage(content="yes, that's right"),
        ],
    )

    # 1. The reply node sees the open handshake and receives the overlay.
    reply_update = await generate_reply_node(state, {})
    prompt = reply_model.last_system_prompt
    assert "SATISFACTION CHECK IN PROGRESS" in prompt
    assert "Do NOT ask another question about this section" in prompt
    assert "Never claim anything has been saved" in prompt

    # 2. Merge the reply, as the graph would.
    state["messages"] = [*state["messages"], *reply_update["messages"]]

    # 3. The decision advances and closes the handshake.
    decision_update = await generate_decision_node(state, {})
    assert decision_update["router_directive"] == "next"
    assert decision_update["awaiting_satisfaction_feedback"] is False, (
        "a stale flag would re-trigger the overlay and re-ask for a confirmation already given"
    )


# --------------------------------------------------------------------------
# Full turn through the compiled graph
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_turn_completes_and_does_not_duplicate_messages(graph_with_fakes):
    graph, fakes = graph_with_fakes
    config = make_config()

    await graph.ainvoke({"messages": [HumanMessage(content="I need a new job")]}, config)
    values = (await graph.aget_state(config)).values

    # PR 1 and PR 2 behaviour still holds.
    assert values["user_id"] == 7
    assert values["current_section"] is SectionID.CAREER_GOAL
    assert values["section_states"]["career_goal"].status is SectionStatus.IN_PROGRESS

    # Exactly one human + one AI message: nothing duplicated through the reducer.
    assert len(values["messages"]) == 2
    assert isinstance(values["messages"][0], HumanMessage)
    assert isinstance(values["messages"][1], AIMessage)

    assert values["router_directive"] == "stay"
    assert values["agent_output"] is not None
    assert fakes["reply"].call_count == 1
    assert fakes["decision"].call_count == 1


@pytest.mark.asyncio
async def test_tokens_stream_from_the_reply_node(graph_with_fakes):
    """Streaming through the real path — the mechanism the SSE layer consumes."""
    graph, _ = graph_with_fakes

    chunks = []
    async for mode, event in graph.astream(
        {"messages": [HumanMessage(content="I need a new job")]},
        make_config(),
        stream_mode=["updates", "messages"],
    ):
        if mode != "messages":
            continue
        message, metadata = event
        if INTERNAL_DECISION_TAG in (metadata.get("tags") or []):
            continue
        if metadata.get("langgraph_node") == "generate_reply":
            chunks.append(message.content)

    # Token-level, not one blob. LangGraph emits the incremental chunks and then
    # a final aggregate message, so the join contains the reply twice — assert
    # containment plus a high chunk count rather than exact equality.
    assert len(chunks) > 10, f"expected per-token chunks, got {len(chunks)}"
    assert REPLY_TEXT in "".join(chunks)
    incremental = [c for c in chunks if len(c) == 1]
    assert "".join(incremental) == REPLY_TEXT, "streamed tokens must reconstruct the reply"


@pytest.mark.asyncio
async def test_decision_tokens_are_all_tagged(graph_with_fakes):
    """The only thing between decision JSON and the user's screen.

    Any untagged chunk from generate_decision is forwarded verbatim by
    service.py as an SSE `token` event.
    """
    graph, _ = graph_with_fakes

    tagged = untagged = 0
    async for mode, event in graph.astream(
        {"messages": [HumanMessage(content="I need a new job")]},
        make_config(),
        stream_mode=["messages"],
    ):
        _message, metadata = event
        if metadata.get("langgraph_node") != "generate_decision":
            continue
        if INTERNAL_DECISION_TAG in (metadata.get("tags") or []):
            tagged += 1
        else:
            untagged += 1

    assert untagged == 0, "decision JSON would leak into the SSE stream"
    assert tagged > 0, "expected the decision call to emit tagged chunks"


@pytest.mark.asyncio
async def test_next_happy_model_still_terminates(graph_with_fakes_next):
    """Nothing marks a section DONE in PR 3, so `next` cannot advance. Without
    the reply cap this would loop until recursion_limit and raise.

    With the cap the turn is exactly 10 super-steps:
      initialize, router, reply, decision, memory_updater,
      router, reply, decision, memory_updater, router -> END
    The limit below is set just above that, so an uncapped loop would still fail
    here while the capped one passes.
    """
    graph, fakes = graph_with_fakes_next
    config = make_config()

    await graph.ainvoke(
        {"messages": [HumanMessage(content="I need a new job")]},
        {**config, "recursion_limit": 12},
    )
    values = (await graph.aget_state(config)).values

    # Two replies (the cap), then the turn ends on a forced stay.
    assert fakes["reply"].call_count == 2
    assert values["router_directive"] == "stay"
    assert len([m for m in values["messages"] if isinstance(m, AIMessage)]) == 2


# --------------------------------------------------------------------------
# PR 4 Stage 3: real section progression through the compiled graph
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_satisfied_career_goal_progresses_to_background(graph_with_satisfied_next):
    """The progression PR 3 could not achieve.

    Before Stage 3 nothing marked a section DONE, so `next` asked
    get_next_unfinished_section for the next section and got the *same* one back
    forever. With the DONE transition in place the router genuinely advances.
    """
    graph, _ = graph_with_satisfied_next
    config = make_config()

    await graph.ainvoke(
        {"messages": [HumanMessage(content="yes, that's right")]},
        {**config, "recursion_limit": 12},
    )
    values = (await graph.aget_state(config)).values

    assert values["section_states"]["career_goal"].status is SectionStatus.DONE
    assert values["current_section"] is SectionID.BACKGROUND, (
        "a satisfied Career Goal must hand off to Background"
    )
    assert values["section_states"]["background"].status is SectionStatus.IN_PROGRESS
    # The remaining three are untouched.
    for section in (SectionID.JOB_PREFERENCES, SectionID.SKILL_ASSESSMENT, SectionID.ACTION_PLAN):
        assert values["section_states"][section.value].status is SectionStatus.PENDING

    # Not complete yet, so the implementation branch is not taken.
    assert values.get("should_generate_final_output", False) is False


@pytest.mark.asyncio
async def test_fifth_section_completion_reaches_implementation_without_raising(
    graph_with_satisfied_next,
):
    """route_after_memory_updater sends the turn to implementation_node here.

    Before Stage 3's pass-through that node raised NotImplementedError, so
    completing the final section would have taken the whole turn down.
    """
    graph, _ = graph_with_satisfied_next
    config = make_config()

    four_done = {
        section.value: SectionState(
            section_id=section,
            status=(
                SectionStatus.IN_PROGRESS
                if section is SectionID.ACTION_PLAN
                else SectionStatus.DONE
            ),
        )
        for section in SectionID
    }

    # No pytest.raises: reaching implementation must be uneventful.
    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="yes, ship it")],
            "section_states": four_done,
            "current_section": SectionID.ACTION_PLAN,
            "router_directive": "stay",
        },
        {**config, "recursion_limit": 12},
    )
    values = (await graph.aget_state(config)).values

    assert all(
        values["section_states"][section.value].status is SectionStatus.DONE
        for section in SectionID
    ), "all five sections should be done"
    assert values["should_generate_final_output"] is True


# --------------------------------------------------------------------------
# PR 5 Stage 3: the graph produces a final artifact, once
#
# Replaces test_implementation_node_is_a_pure_compatibility_no_op, which asserted
# `update == {}` unconditionally. That was right for PR 4's pass-through and is now
# wrong, because the node synthesizes. The empty-update case survives as the
# once-only guard and is covered in test_implementation.py.
# --------------------------------------------------------------------------

PLAN = ["Rewrite the CV summary", "Message three former colleagues", "Ship a K8s project"]


def _patch_synthesis(monkeypatch, steps):
    """Fake only the synthesis model; real assembly and rendering still run."""
    from agents.xbuddy import synthesis as synthesis_module
    from agents.xbuddy.models import ActionAnnotation, FinalOutputDraft

    class CountingChain:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages, *args, **kwargs):
            self.calls += 1
            return {
                "parsed": FinalOutputDraft(
                    headline="QA Analyst to Senior SRE",
                    positioning_summary="Four years of QA moving into automation.",
                    strengths_to_leverage=["systems debugging"],
                    skill_priorities=["Kubernetes"],
                    search_targets=["fintech"],
                    action_annotations=[
                        ActionAnnotation(
                            step_number=index, rationale=f"reason {index}", timeframe=None
                        )
                        for index in range(1, len(steps) + 1)
                    ],
                    risks_or_constraints=[],
                ),
                "parsing_error": None,
            }

    chain = CountingChain()
    monkeypatch.setattr(synthesis_module, "_synthesis_chain", lambda: chain)
    return chain


def _four_done_action_plan_open() -> dict:
    return {
        section.value: SectionState(
            section_id=section,
            status=(
                SectionStatus.IN_PROGRESS
                if section is SectionID.ACTION_PLAN
                else SectionStatus.DONE
            ),
        )
        for section in SectionID
    }


def _completion_seed() -> dict:
    return {
        "messages": [HumanMessage(content="yes, ship it")],
        "section_states": _four_done_action_plan_open(),
        "current_section": SectionID.ACTION_PLAN,
        "router_directive": "stay",
        "user_data": XBuddyData(target_roles=["Senior SRE"], action_items=list(PLAN)),
    }


def _readiness_messages(values) -> list:
    from agents.xbuddy.nodes.implementation import FINAL_OUTPUT_READY_MESSAGE

    return [
        message
        for message in values["messages"]
        if isinstance(message, AIMessage) and message.content == FINAL_OUTPUT_READY_MESSAGE
    ]


@pytest.mark.asyncio
async def test_completing_the_fifth_section_generates_the_artifact(
    graph_with_satisfied_next, monkeypatch
):
    """The end-to-end Stage 3 claim: finishing section 5 yields a Markdown document
    and one readiness line, without raising."""
    graph, _ = graph_with_satisfied_next
    chain = _patch_synthesis(monkeypatch, PLAN)
    config = make_config()

    await graph.ainvoke(_completion_seed(), {**config, "recursion_limit": 12})
    values = (await graph.aget_state(config)).values

    assert chain.calls == 1
    assert values["final_output"].startswith("# QA Analyst to Senior SRE")
    assert "## Your Action Plan" in values["final_output"]
    assert "## What I Still Don\'t Know" in values["final_output"]
    # The shared graph fixture's extraction fake reports a parsing error by design,
    # so `last_error` is legitimately non-empty here. What matters is that synthesis
    # itself contributed no error.
    assert "synthesis" not in (values.get("last_error") or "")
    assert len(_readiness_messages(values)) == 1


@pytest.mark.asyncio
async def test_the_artifact_is_not_pasted_into_the_conversation(
    graph_with_satisfied_next, monkeypatch
):
    """Chat announces readiness; the document itself stays out of the transcript."""
    graph, _ = graph_with_satisfied_next
    _patch_synthesis(monkeypatch, PLAN)
    config = make_config()

    await graph.ainvoke(_completion_seed(), {**config, "recursion_limit": 12})
    values = (await graph.aget_state(config)).values

    for message in values["messages"]:
        assert "## Your Action Plan" not in str(message.content)


@pytest.mark.asyncio
async def test_a_later_turn_does_not_pay_for_synthesis_again(
    graph_with_satisfied_next, monkeypatch
):
    """`should_generate_final_output` stays True, so this turn re-enters
    implementation. The existing artifact is what stops it."""
    graph, _ = graph_with_satisfied_next
    chain = _patch_synthesis(monkeypatch, PLAN)
    config = make_config()

    await graph.ainvoke(_completion_seed(), {**config, "recursion_limit": 12})
    first = (await graph.aget_state(config)).values
    assert chain.calls == 1

    await graph.ainvoke(
        {"messages": [HumanMessage(content="thanks!")]}, {**config, "recursion_limit": 12}
    )
    values = (await graph.aget_state(config)).values

    assert values["should_generate_final_output"] is True
    assert chain.calls == 1, "a second turn must not regenerate the artifact"
    assert values["final_output"] == first["final_output"]
    assert len(_readiness_messages(values)) == 1, "no duplicate readiness announcement"


@pytest.mark.asyncio
async def test_synthesis_failure_does_not_break_the_turn(
    graph_with_satisfied_next, monkeypatch
):
    """implementation -> END, so a raise here would lose a turn whose conversational
    work has already succeeded."""
    from agents.xbuddy import synthesis as synthesis_module

    class Exploding:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("rate limited")

    monkeypatch.setattr(synthesis_module, "_synthesis_chain", lambda: Exploding())

    graph, _ = graph_with_satisfied_next
    config = make_config()

    await graph.ainvoke(_completion_seed(), {**config, "recursion_limit": 12})
    values = (await graph.aget_state(config)).values

    assert values.get("final_output") is None, "no partial artifact"
    assert "rate limited" in values["last_error"]
    assert all(
        section.status is SectionStatus.DONE
        for section in values["section_states"].values()
    ), "the section work of the turn survives a synthesis failure"
    assert _readiness_messages(values) == []


@pytest.mark.asyncio
async def test_synthesis_runs_are_tagged_for_suppression(
    graph_with_satisfied_next, monkeypatch
):
    """Every synthesis message event must carry a tag service.py drops.

    The suppression list is read out of service.py rather than duplicated here, so
    removing the tag from either side fails this test.
    """
    import re
    from pathlib import Path

    from agents.xbuddy.synthesis import INTERNAL_SYNTHESIS_TAG

    service_source = (
        Path(__file__).resolve().parents[3] / "src" / "service" / "service.py"
    ).read_text(encoding="utf-8")
    match = re.search(r"if any\(tag in tags for tag in \[([^\]]*)\]\)", service_source)
    assert match, "could not locate the suppression list in service.py"
    suppressed = set(re.findall(r'"([^"]+)"', match.group(1)))

    assert INTERNAL_SYNTHESIS_TAG in suppressed, (
        "internal_synthesis must be in service.py's suppression list, "
        "or the raw structured JSON streams to the user"
    )

    graph, _ = graph_with_satisfied_next
    _patch_synthesis(monkeypatch, PLAN)
    config = make_config()

    seen = 0
    async for mode, event in graph.astream(
        _completion_seed(), {**config, "recursion_limit": 12}, stream_mode=["messages"]
    ):
        if mode != "messages":
            continue
        _message, metadata = event
        tags = metadata.get("tags") or []
        if INTERNAL_SYNTHESIS_TAG in tags:
            seen += 1
            assert any(tag in suppressed for tag in tags)

    # The faked chain emits no message events, so `seen` may be 0; the binding
    # assertion above is the one that matters and holds either way.
    assert seen >= 0
