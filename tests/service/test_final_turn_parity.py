"""The last turn must show live exactly what it will show after a refresh.

The bug this pins: confirming the Action Plan completed the conversation, and
`implementation_node` appended a readiness line that was checkpointed correctly.
`/history` returned it after F5 — but it never appeared during the live turn, so the
transcript silently grew an assistant bubble on reload.

The cause was in `message_generator`, not in the graph. Its `updates` branch kept two
counters and required a node's payload to be *longer* than one of them:

    initial_message_count  # messages in the thread before the run  (cumulative)
    sent_message_count     # messages emitted so far this stream    (running total)
    len(update_messages)   # ONE node's returned delta              (per-node)

Those are different units. `generate_reply` returns one message and sets the running
total to 1; `implementation` then returns one message and fails `1 > 1`. Any second
node that speaks in a turn was dropped, whatever it was.

So these tests drive the **service streaming boundary** with a fake agent shaped like
the real final turn, and compare what the stream emits against what
`load_chat_history` — the actual `/history` implementation — returns for the same
checkpoint. An `implementation_node` unit test cannot see this class of bug at all:
the node's return value was always correct.

Offline: no model, no checkpointer, no network.
"""

import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from schema import StreamInput
from service.service import load_chat_history, message_generator

OWNER = 4242
THREAD = "final-turn-parity"

CONFIRMATION = "yes, that plan looks right"
REPLY = "Locked in. That is your action plan agreed and the last section closed."
READY = 'Your job search strategy is ready — open "View Final Plan" to read it.'

# What the thread already holds when the final turn starts: four completed sections
# worth of back-and-forth. Long enough that `initial_message_count` was large, which
# is the condition under which the old filter fell through to its second branch.
PRIOR_HISTORY = [
    message
    for pair in range(6)
    for message in (
        HumanMessage(content=f"user turn {pair}"),
        AIMessage(content=f"assistant turn {pair}"),
    )
]

FINAL_PLAN = "# Your Job Search Strategy\n\n## Career Direction\nAI engineering.\n"


def final_turn_agent():
    """An agent shaped like the real last turn.

    `astream` emits the two node updates the graph actually produces, each carrying
    its own delta — which is what `add_messages` requires and what every xbuddy node
    returns. `aget_state` returns the checkpoint as it stands *after* the turn, so
    the same object can be read by `load_chat_history`.
    """
    agent = Mock()

    async def astream(**kwargs):
        yield ("updates", {"generate_reply": {"messages": [AIMessage(content=REPLY)]}})
        yield (
            "updates",
            {
                "implementation": {
                    "final_output": FINAL_PLAN,
                    "messages": [AIMessage(content=READY)],
                }
            },
        )

    async def aget_state(config=None):
        snapshot = Mock()
        snapshot.values = {
            "user_id": OWNER,
            "messages": [
                *PRIOR_HISTORY,
                HumanMessage(content=CONFIRMATION),
                AIMessage(content=REPLY),
                AIMessage(content=READY),
            ],
            "final_output": FINAL_PLAN,
        }
        snapshot.tasks = []
        return snapshot

    agent.astream = astream
    agent.aget_state = aget_state
    return agent


async def stream_events(agent) -> list[dict]:
    async def _handle(user_input, resolved_agent, agent_id):
        return (
            {"config": {"configurable": {"thread_id": THREAD, "user_id": OWNER}}},
            "run-parity",
        )

    with (
        patch("service.service.get_agent", Mock(return_value=agent)),
        patch("service.service._handle_input", side_effect=_handle),
    ):
        events: list[dict] = []
        async for chunk in message_generator(
            StreamInput(message=CONFIRMATION, thread_id=THREAD, user_id=OWNER), "xbuddy"
        ):
            for line in chunk.splitlines():
                if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                    events.append(json.loads(line[6:]))
        return events


def assistant_text(events: list[dict]) -> list[str]:
    """Assistant lines the client is handed as discrete messages."""
    return [
        event["content"]["content"]
        for event in events
        if event["type"] == "message" and event["content"].get("type") == "ai"
    ]


# --------------------------------------------------------------------------
# The invariant
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_transcript_matches_history_after_refresh():
    """The whole point. Every assistant message this turn persists is streamed."""
    agent = final_turn_agent()

    live = assistant_text(await stream_events(agent))

    with patch("service.service.get_agent", Mock(return_value=agent)):
        restored = await load_chat_history("xbuddy", THREAD, OWNER)
    persisted = [m.content for m in restored.messages if m.type == "ai"]

    # The turn's own contribution: everything after the prior history.
    new_persisted = persisted[len([m for m in PRIOR_HISTORY if isinstance(m, AIMessage)]) :]

    assert live == new_persisted, (
        "live stream and restored history disagree; refreshing would change the "
        f"transcript.\n  live      = {live}\n  persisted = {new_persisted}"
    )


@pytest.mark.asyncio
async def test_the_readiness_line_is_streamed_live():
    """Named directly, because this is the message the user reported."""
    live = assistant_text(await stream_events(final_turn_agent()))
    assert READY in live, "the final-ready message reached the checkpoint but not the stream"


@pytest.mark.asyncio
async def test_a_second_speaking_node_is_not_dropped():
    """The general defect, stated without reference to which node speaks second.

    The old filter compared a per-node delta length against a running total, so the
    second one-message node in any turn was always discarded.
    """
    live = assistant_text(await stream_events(final_turn_agent()))
    assert len(live) == 2, f"expected both nodes' messages, got {live}"


# --------------------------------------------------------------------------
# The invariant it must not break: one reply, emitted once
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_message_is_emitted_twice():
    """Guards the duplicate-reply fix from the other direction.

    Emitting every delta must not mean emitting anything twice — a repeated
    `message` event is what puts the same reply on screen in two bubbles.
    """
    live = assistant_text(await stream_events(final_turn_agent()))
    assert len(live) == len(set(live)), f"duplicate message events: {live}"


@pytest.mark.asyncio
async def test_the_echoed_user_message_is_not_streamed_back():
    """LangGraph re-sends the input; it must not arrive as an assistant bubble."""
    events = await stream_events(final_turn_agent())
    humans = [
        event["content"]["content"]
        for event in events
        if event["type"] == "message" and event["content"].get("type") == "human"
    ]
    assert CONFIRMATION not in humans


# --------------------------------------------------------------------------
# Ordering and completion state
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completion_arrives_after_every_message():
    """The client updates progress once the transcript is settled, not before."""
    kinds = [event["type"] for event in await stream_events(final_turn_agent())]
    assert kinds.count("completion") == 1
    assert kinds.index("completion") > max(
        index for index, kind in enumerate(kinds) if kind == "message"
    )


@pytest.mark.asyncio
async def test_the_completion_event_reports_the_artifact_from_this_turn():
    """Read post-stream, so it sees what `implementation` wrote during the turn."""
    events = await stream_events(final_turn_agent())
    completion = next(e["content"] for e in events if e["type"] == "completion")
    assert completion["artifact_available"] is True


@pytest.mark.asyncio
async def test_the_plan_itself_never_enters_the_transcript():
    """The document belongs in the panel. A readiness line points at it; pasting
    Markdown into a chat bubble is the failure mode this guards."""
    events = await stream_events(final_turn_agent())
    for line in assistant_text(events):
        assert "## Career Direction" not in line
        assert line != FINAL_PLAN


# --------------------------------------------------------------------------
# Wording
# --------------------------------------------------------------------------


def test_the_ready_message_does_not_promise_editing():
    """FinalPlanPanel is read-only by construction — no editor, no export. The
    message used to say "you can open and edit it now"."""
    from agents.xbuddy.nodes.implementation import FINAL_OUTPUT_READY_MESSAGE

    lowered = FINAL_OUTPUT_READY_MESSAGE.lower()
    for promise in ("edit", "editing", "editor", "export", "download"):
        assert promise not in lowered, f"the plan is read-only but the message says {promise!r}"


def test_the_ready_message_names_the_control_the_user_has():
    from agents.xbuddy.nodes.implementation import FINAL_OUTPUT_READY_MESSAGE

    assert "View Final Plan" in FINAL_OUTPUT_READY_MESSAGE
    assert "sidebar" in FINAL_OUTPUT_READY_MESSAGE
    assert "completed all five sections" in FINAL_OUTPUT_READY_MESSAGE
    assert "personalized final career plan" in FINAL_OUTPUT_READY_MESSAGE
    assert "information we collected" in FINAL_OUTPUT_READY_MESSAGE
    assert "\n" not in FINAL_OUTPUT_READY_MESSAGE
