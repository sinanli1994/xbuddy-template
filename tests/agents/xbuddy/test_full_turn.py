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


def _patch_graph(monkeypatch, decision):
    from agents.xbuddy.agent import graph
    from agents.xbuddy.nodes import generate_decision as decision_module
    from agents.xbuddy.nodes import generate_reply as reply_module

    fakes = {"reply": CountingReplyModel(), "decision": StreamingDecisionChain(decision)}
    monkeypatch.setattr(reply_module, "_reply_model", lambda: fakes["reply"])
    monkeypatch.setattr(decision_module, "_decision_chain", lambda: fakes["decision"])
    return graph, fakes


@pytest.fixture
def graph_with_fakes(monkeypatch, make_decision):
    """Decision always says `stay` — the ordinary one-reply turn."""
    return _patch_graph(monkeypatch, make_decision(action=DecisionAction.STAY))


@pytest.fixture
def graph_with_fakes_next(monkeypatch, make_decision):
    """Decision always says `next` — the runaway-loop scenario."""
    return _patch_graph(monkeypatch, make_decision(action=DecisionAction.NEXT))


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
