"""Intermediate progress may describe committed checkpoints only."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from service.service import _read_committed_progress


@pytest.mark.asyncio
async def test_waits_for_the_exact_checkpoint_not_a_newer_pending_snapshot():
    config = {"configurable": {"thread_id": "t", "checkpoint_id": "committed-id"}}
    old = SimpleNamespace(config={"configurable": {"checkpoint_id": "old-id"}}, values={"user_id": 7})
    committed = SimpleNamespace(config=config, values={"user_id": 7, "section_states": {
        "career_goal": {"status": "done"},
    }})
    agent = SimpleNamespace(aget_state=AsyncMock(side_effect=[old, committed]))
    projection = await _read_committed_progress(agent, config, 7)
    assert agent.aget_state.await_count == 2
    assert all(call.kwargs["config"] == config for call in agent.aget_state.await_args_list)
    assert projection.sections[0].status == "done"
    assert set(projection.model_dump()) == {"collection_complete", "artifact_available", "sections"}


@pytest.mark.asyncio
async def test_missing_checkpoint_id_never_emits_guessed_progress():
    agent = SimpleNamespace(aget_state=AsyncMock())
    assert await _read_committed_progress(agent, {"configurable": {"thread_id": "t"}}, 7) is None
    agent.aget_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_echoed_checkpoint_id_with_empty_values_is_not_a_commit():
    config = {"configurable": {"thread_id": "t", "checkpoint_id": "id"}}
    empty = SimpleNamespace(config=config, values={})
    saved = SimpleNamespace(config=config, values={"user_id": 7, "section_states": {
        "skill_assessment": {"status": "done"},
    }})
    agent = SimpleNamespace(aget_state=AsyncMock(side_effect=[empty, saved]))
    projection = await _read_committed_progress(agent, config, 7)
    assert agent.aget_state.await_count == 2
    assert projection.sections[3].status == "done"


@pytest.mark.asyncio
async def test_other_owner_checkpoint_is_not_projected():
    config = {"configurable": {"checkpoint_id": "id"}}
    agent = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(
        config=config, values={"user_id": 999},
    )))
    assert await _read_committed_progress(agent, config, 7) is None


@pytest.mark.asyncio
async def test_failed_checkpoint_read_defers_progress_instead_of_fabricating_it():
    agent = SimpleNamespace(aget_state=AsyncMock(side_effect=RuntimeError("write not committed")))
    assert await _read_committed_progress(agent, {"configurable": {"checkpoint_id": "id"}}, 7) is None
