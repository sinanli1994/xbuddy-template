"""Tests for generate_reply_node — the user-facing, streamed half of a turn."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy.enums import SectionID
from agents.xbuddy.models import XBuddyData
from agents.xbuddy.nodes.generate_reply import (
    FALLBACK_REPLY,
    SHORT_MEMORY_SIZE,
    generate_reply_node,
)
from agents.xbuddy.sections.base_prompt import SATISFACTION_OVERLAY


def history(count: int) -> list:
    """`count` alternating messages, oldest first."""
    return [
        HumanMessage(content=f"user {i}") if i % 2 == 0 else AIMessage(content=f"agent {i}")
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_returns_exactly_one_ai_message(reply_model, make_state):
    state = make_state(messages=history(6))
    update = await generate_reply_node(state, {})

    # The add_messages reducer appends: returning the history would duplicate it.
    assert len(update["messages"]) == 1
    assert isinstance(update["messages"][0], AIMessage)
    assert update["messages"][0].content == reply_model.reply
    # No section_id/agent_name — those break langchain_to_chat_message.
    assert not update["messages"][0].additional_kwargs


@pytest.mark.asyncio
async def test_system_prompt_is_the_context_packet_verbatim(reply_model, make_state):
    """PR 2 owns prompt assembly; this node must not rebuild it."""
    state = make_state(user_data=XBuddyData(target_roles=["SRE"]))
    await generate_reply_node(state, {})

    assert reply_model.last_system_prompt == state["context_packet"].system_prompt


@pytest.mark.asyncio
async def test_history_passed_in_order_and_trimmed(reply_model, make_state):
    state = make_state(section=SectionID.BACKGROUND, messages=history(14))
    update = await generate_reply_node(state, {})

    sent = reply_model.calls[-1]
    history_sent = sent[1:]  # index 0 is the SystemMessage

    assert len(history_sent) == SHORT_MEMORY_SIZE
    assert history_sent == state["messages"][-SHORT_MEMORY_SIZE:]
    assert update["short_memory"] == history_sent
    # Nothing duplicated into the outgoing message list.
    assert len(sent) == SHORT_MEMORY_SIZE + 1


@pytest.mark.asyncio
async def test_sets_awaiting_user_input(reply_model, make_state):
    update = await generate_reply_node(make_state(), {})
    assert update["awaiting_user_input"] is True


@pytest.mark.asyncio
async def test_missing_context_packet_falls_back_without_calling_model(reply_model, make_state):
    update = await generate_reply_node(make_state(with_packet=False), {})

    assert reply_model.call_count == 0, "must not prompt a model with no section context"
    assert update["messages"][0].content == FALLBACK_REPLY
    assert "context_packet" in update["last_error"]


@pytest.mark.asyncio
async def test_model_error_degrades_to_fallback(make_state, monkeypatch):
    from agents.xbuddy.nodes import generate_reply as module

    class Boom:
        async def ainvoke(self, messages, config=None):
            raise RuntimeError("upstream 503")

    monkeypatch.setattr(module, "_reply_model", Boom)

    update = await generate_reply_node(make_state(error_count=1), {})
    assert update["messages"][0].content == FALLBACK_REPLY
    assert update["error_count"] == 2
    assert "503" in update["last_error"]


# --------------------------------------------------------------------------
# Satisfaction overlay (plan §2.1)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("awaiting", [True, False])
async def test_satisfaction_overlay_applied_only_when_awaiting(reply_model, make_state, awaiting):
    state = make_state(
        awaiting_satisfaction_feedback=awaiting,
        messages=[HumanMessage(content="yes, that's right")],
    )
    await generate_reply_node(state, {})

    prompt = reply_model.last_system_prompt
    assert ("SATISFACTION CHECK IN PROGRESS" in prompt) is awaiting

    if awaiting:
        # Must forbid claiming anything was persisted — PR 3 saves nothing.
        assert "Never claim anything has been saved" in prompt
        assert SATISFACTION_OVERLAY.strip() in prompt
        # The section's own prompt survives; the overlay is additive.
        assert state["context_packet"].system_prompt in prompt
