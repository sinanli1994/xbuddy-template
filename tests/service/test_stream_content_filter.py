"""PR 6 Stage 1: a JobBuddy reply must never be dropped for containing a word.

`message_generator` used to carry a ~60-entry list of FounderBuddy field names
(`client_name`, `icp_nickname`, `pain1_symptom`, `prize_statement`, …) and matched
them as **substrings** against reply content, skipping any message that hit. Four of
those names are ordinary English that JobBuddy uses constantly:

    "industry"    — Section 3 collects target_industries
    "specialty"   — natural in a Background question
    "mistakes"    — natural coaching language
    "principles"  — natural coaching language

So the `message` event for those replies was silently dropped. Tokens still
streamed, so the text reached the user, but any client assembling from `message`
events lost the turn.

These tests drive `message_generator` directly with a fake agent. Offline: no
model, no network, no checkpointer.
"""

import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from schema import StreamInput
from service.service import message_generator

# Words drawn from the removed list that JobBuddy legitimately says.
FORMERLY_DROPPED = [
    "Which industry are you targeting — fintech, healthtech, something else?",
    "What's your specialty within backend engineering?",
    "Common mistakes at this stage are applying too broadly.",
    "What are the guiding principles behind your search?",
]


def fake_agent(reply: str):
    """An agent whose astream emits one node update carrying the reply.

    Shape matters: `message_generator` requests
    `stream_mode=["updates", "messages", "custom"]` and the `updates` branch
    iterates `event.items()` as `{node_name: {"messages": [...]}}`. An earlier
    version of this fake yielded `("values", ...)`, which the generator ignores
    entirely — every assertion passed vacuously against an empty event list.
    """
    agent = Mock()

    async def astream(**kwargs):
        yield (
            "updates",
            {
                "generate_reply": {
                    "messages": [HumanMessage(content="hello"), AIMessage(content=reply)]
                }
            },
        )

    async def aget_state(config):
        state = Mock()
        state.values = {}  # no current_section -> the section block is skipped
        return state

    agent.astream = astream
    agent.aget_state = aget_state
    return agent


async def collect(reply: str) -> list[dict]:
    """Run the generator and return every parsed SSE payload."""
    stream_input = StreamInput(message="hello", thread_id=None, user_id=1, stream_tokens=False)
    agent = fake_agent(reply)

    events: list[dict] = []
    with patch("service.service.get_agent", Mock(return_value=agent)):
        async for chunk in message_generator(stream_input, "xbuddy"):
            assert chunk.startswith("data: "), chunk
            payload = chunk[len("data: ") :].strip()
            if payload == "[DONE]":
                events.append({"type": "__done__"})
                continue
            events.append(json.loads(payload))
    return events


def messages_in(events: list[dict]) -> list[str]:
    return [
        event["content"]["content"]
        for event in events
        if event.get("type") == "message" and event["content"].get("type") == "ai"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", FORMERLY_DROPPED)
async def test_a_reply_containing_a_founderbuddy_field_word_still_emits_a_message(reply):
    """The regression this stage exists to prevent."""
    events = await collect(reply)

    assert reply in messages_in(events), (
        f"the message event for {reply!r} was dropped — the FounderBuddy content "
        f"filter is back. Emitted: {[e.get('type') for e in events]}"
    )


@pytest.mark.asyncio
async def test_an_ordinary_reply_still_emits_a_message():
    """Control: proves the harness would notice a message going missing at all."""
    reply = "What role are you targeting next?"
    assert reply in messages_in(await collect(reply))


@pytest.mark.asyncio
async def test_the_stream_still_terminates_with_done():
    events = await collect(FORMERLY_DROPPED[0])
    assert events[-1] == {"type": "__done__"}


@pytest.mark.asyncio
async def test_the_stream_still_opens_with_metadata():
    """The frontend reads thread_id/run_id off this first event."""
    events = await collect("hello there")
    assert events[0]["type"] == "metadata"
    for key in ("thread_id", "user_id", "run_id"):
        assert key in events[0]["content"]


@pytest.mark.asyncio
async def test_the_echoed_human_input_is_still_dropped():
    """LangGraph re-sends the input message; that suppression is unrelated to the
    removed filter and must survive it."""
    events = await collect("A reply about industry trends.")
    human = [
        event
        for event in events
        if event.get("type") == "message" and event["content"].get("type") == "human"
    ]
    assert human == []
