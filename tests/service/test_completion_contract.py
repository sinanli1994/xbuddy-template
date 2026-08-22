"""PR 6 Stage 2: the narrow public completion contract.

`/invoke` and the SSE `completion` event both read `public_completion`, so these
tests pin one projection and both surfaces.

Three states matter and can all occur in real flows:

    incomplete                    collection_complete=False, artifact_available=False
    collected but no artifact     collection_complete=True,  artifact_available=False
    fully complete                collection_complete=True,  artifact_available=True

The middle one is not hypothetical — a completed conversation whose synthesis
failed sits there until a later turn retries, which is exactly why the contract is
two booleans rather than one.

Offline throughout: no model, no network, no checkpointer.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage

from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import SectionState
from schema import CompletionState, InvokeResponse, StreamInput
from service.service import message_generator, public_completion

PUBLIC_SECTION_KEYS = {"id", "name", "status"}
EXPECTED_IDS = [
    "career_goal",
    "background",
    "job_preferences",
    "skill_assessment",
    "action_plan",
]


def sections(**statuses) -> dict:
    """section_states with every section present; overrides by id."""
    result = {}
    for section in SectionID:
        status = statuses.get(section.value, SectionStatus.PENDING)
        result[section.value] = SectionState(section_id=section, status=status)
    return result


def all_done() -> dict:
    return sections(**{section.value: SectionStatus.DONE for section in SectionID})


INCOMPLETE = {
    "section_states": sections(career_goal=SectionStatus.DONE, background=SectionStatus.IN_PROGRESS),
}
COLLECTED_NO_ARTIFACT = {
    "section_states": all_done(),
    "should_generate_final_output": True,
    "final_output": None,
}
FULLY_COMPLETE = {
    "section_states": all_done(),
    "should_generate_final_output": True,
    "final_output": "# Your strategy\n",
}


# --------------------------------------------------------------------------
# The projection itself
# --------------------------------------------------------------------------


def test_an_incomplete_conversation_reports_neither_flag():
    result = public_completion(INCOMPLETE)
    assert result.collection_complete is False
    assert result.artifact_available is False


def test_collection_complete_without_an_artifact():
    """The state a failed synthesis leaves behind. Both flags must be independent."""
    result = public_completion(COLLECTED_NO_ARTIFACT)
    assert result.collection_complete is True
    assert result.artifact_available is False


def test_fully_complete_reports_both():
    result = public_completion(FULLY_COMPLETE)
    assert result.collection_complete is True
    assert result.artifact_available is True


def test_collection_complete_comes_from_should_generate_final_output_only():
    """Not from counting statuses here, and not from `finished`.

    Deriving it twice would let the API disagree with the agent; memory_updater is
    the single writer of that flag.
    """
    # Every section done, but the agent has not set the flag: report False.
    assert public_completion({"section_states": all_done()}).collection_complete is False
    # Flag set with no section_states at all: report True.
    assert public_completion({"should_generate_final_output": True}).collection_complete is True


def test_artifact_available_is_exactly_final_output_is_not_none():
    assert public_completion({"final_output": "# doc"}).artifact_available is True
    assert public_completion({"final_output": None}).artifact_available is False
    assert public_completion({}).artifact_available is False


def test_an_empty_artifact_string_still_counts_as_present():
    """`is not None`, as specified — not truthiness. An empty document is a defect
    for the eval to catch, not something the API should hide."""
    assert public_completion({"final_output": ""}).artifact_available is True


def test_finished_is_never_consulted_and_never_exposed():
    """Issue #10: `finished` is router-owned and directive-gated, so a completed
    thread can sit at False indefinitely. It must not reach the public contract."""
    with_finished = public_completion({**FULLY_COMPLETE, "finished": False})
    assert with_finished.collection_complete is True

    assert "finished" not in CompletionState.model_fields
    assert "finished" not in with_finished.model_dump()


# --------------------------------------------------------------------------
# The five-section projection
# --------------------------------------------------------------------------


def test_all_five_sections_in_canonical_order():
    result = public_completion(INCOMPLETE)
    assert [section.id for section in result.sections] == EXPECTED_IDS


def test_sections_expose_exactly_three_keys():
    """No `database_id`, no draft content, no satisfaction status."""
    for section in public_completion(FULLY_COMPLETE).sections:
        assert set(section.model_dump()) == PUBLIC_SECTION_KEYS


def test_section_names_are_the_human_readable_titles():
    names = [section.name for section in public_completion(INCOMPLETE).sections]
    assert names == [
        "Career Goal",
        "Background",
        "Job Preferences",
        "Skill Assessment",
        "Action Plan",
    ]


def test_statuses_are_projected_per_section():
    result = public_completion(INCOMPLETE)
    by_id = {section.id: section.status for section in result.sections}
    assert by_id["career_goal"] == "done"
    assert by_id["background"] == "in_progress"
    assert by_id["job_preferences"] == "pending"


def test_a_missing_section_is_reported_pending_not_omitted():
    """The array is always five long, so a client never has to handle a short list."""
    partial = {"section_states": {"career_goal": SectionState(
        section_id=SectionID.CAREER_GOAL, status=SectionStatus.DONE
    )}}
    result = public_completion(partial)
    assert len(result.sections) == 5
    assert [s.status for s in result.sections] == ["done", "pending", "pending", "pending", "pending"]


def test_raw_dict_section_states_are_projected_too():
    """A checkpoint can hand back plain dicts rather than SectionState models."""
    raw = {
        section.value: {"section_id": section.value, "status": "done"}
        for section in SectionID
    }
    assert all(s.status == "done" for s in public_completion({"section_states": raw}).sections)


def test_no_internal_state_leaks_through_the_projection():
    noisy = {
        **FULLY_COMPLETE,
        "user_data": {"salary_expectation": "80k"},
        "short_memory": ["secret"],
        "last_error": "boom",
        "persistence_pending": ["career_goal"],
        "final_output_pending": "write",
        "context_packet": {"system_prompt": "internal"},
    }
    dumped = json.dumps(public_completion(noisy).model_dump())
    for leaked in ("salary_expectation", "short_memory", "last_error", "persistence_pending",
                   "final_output_pending", "system_prompt", "context_packet"):
        assert leaked not in dumped


# --------------------------------------------------------------------------
# /invoke response shape
# --------------------------------------------------------------------------


def test_invoke_response_keeps_its_original_fields_and_adds_three():
    fields = set(InvokeResponse.model_fields)
    assert {"output", "thread_id", "user_id"} <= fields, "existing fields must survive"
    assert {"collection_complete", "artifact_available", "sections"} <= fields
    assert "finished" not in fields


@pytest.mark.parametrize(
    ("values", "expected_complete", "expected_artifact"),
    [
        (INCOMPLETE, False, False),
        (COLLECTED_NO_ARTIFACT, True, False),
        (FULLY_COMPLETE, True, True),
    ],
)
def test_invoke_populates_the_completion_fields_from_state(
    values, expected_complete, expected_artifact
):
    """The endpoint must actually project state, not just declare the fields.

    Asserting only that `InvokeResponse` *has* the keys left a gap: hardcoding
    `collection_complete=False` in the handler passed every other test in this file.
    """
    from fastapi.testclient import TestClient

    from service import app

    agent = Mock()
    agent.ainvoke = AsyncMock(
        return_value=[("values", {"messages": [AIMessage(content="A reply.")]})]
    )

    async def aget_state(config):
        snapshot = Mock()
        snapshot.values = values
        return snapshot

    agent.aget_state = aget_state

    with patch("service.service.get_agent", Mock(return_value=agent)):
        response = TestClient(app).post("/invoke", json={"message": "hi", "user_id": 1})

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["collection_complete"] is expected_complete
    assert body["artifact_available"] is expected_artifact
    assert [section["id"] for section in body["sections"]] == EXPECTED_IDS
    for section in body["sections"]:
        assert set(section) == PUBLIC_SECTION_KEYS
    # Existing fields untouched.
    assert body["output"]["content"] == "A reply."
    assert "thread_id" in body and "user_id" in body
    assert "finished" not in body


# --------------------------------------------------------------------------
# The SSE completion event
# --------------------------------------------------------------------------


def fake_agent(final_values: dict, *, raise_midstream: bool = False):
    agent = Mock()

    async def astream(**kwargs):
        yield ("updates", {"generate_reply": {"messages": [AIMessage(content="A reply.")]}})
        if raise_midstream:
            raise RuntimeError("stream blew up")

    async def aget_state(config):
        state = Mock()
        state.values = final_values
        return state

    agent.astream = astream
    agent.aget_state = aget_state
    return agent


async def collect(final_values: dict, *, raise_midstream: bool = False, stream_tokens: bool = False):
    stream_input = StreamInput(
        message="hello", thread_id=None, user_id=1, stream_tokens=stream_tokens
    )
    agent = fake_agent(final_values, raise_midstream=raise_midstream)
    events: list[dict] = []
    with patch("service.service.get_agent", Mock(return_value=agent)):
        async for chunk in message_generator(stream_input, "xbuddy"):
            payload = chunk[len("data: ") :].strip()
            events.append({"type": "__done__"} if payload == "[DONE]" else json.loads(payload))
    return events


def types_of(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


@pytest.mark.asyncio
async def test_completion_is_emitted_exactly_once_immediately_before_done():
    events = await collect(FULLY_COMPLETE)
    kinds = types_of(events)

    assert kinds.count("completion") == 1, kinds
    assert kinds[-2:] == ["completion", "__done__"], kinds


@pytest.mark.asyncio
async def test_the_completion_event_carries_the_projection():
    events = await collect(FULLY_COMPLETE)
    completion = next(event for event in events if event["type"] == "completion")

    assert completion["content"]["collection_complete"] is True
    assert completion["content"]["artifact_available"] is True
    assert [s["id"] for s in completion["content"]["sections"]] == EXPECTED_IDS
    for section in completion["content"]["sections"]:
        assert set(section) == PUBLIC_SECTION_KEYS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("values", "expected_complete", "expected_artifact"),
    [
        (INCOMPLETE, False, False),
        (COLLECTED_NO_ARTIFACT, True, False),
        (FULLY_COMPLETE, True, True),
    ],
)
async def test_the_completion_event_reflects_each_state(values, expected_complete, expected_artifact):
    events = await collect(values)
    content = next(e for e in events if e["type"] == "completion")["content"]
    assert content["collection_complete"] is expected_complete
    assert content["artifact_available"] is expected_artifact


@pytest.mark.asyncio
async def test_a_failed_stream_keeps_error_then_done_and_emits_no_completion():
    """No synthetic completion on the error path — the existing shape is preserved."""
    events = await collect(FULLY_COMPLETE, raise_midstream=True)
    kinds = types_of(events)

    assert "completion" not in kinds, kinds
    assert kinds[-2:] == ["error", "__done__"], kinds


# --------------------------------------------------------------------------
# Existing event contracts must be unchanged
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_is_still_the_first_event():
    events = await collect(FULLY_COMPLETE)
    assert events[0]["type"] == "metadata"
    for key in ("thread_id", "user_id", "run_id"):
        assert key in events[0]["content"]


@pytest.mark.asyncio
async def test_message_events_are_unchanged():
    events = await collect(FULLY_COMPLETE)
    messages = [e for e in events if e["type"] == "message"]
    assert messages, types_of(events)
    assert messages[0]["content"]["content"] == "A reply."
    assert messages[0]["content"]["type"] == "ai"


@pytest.mark.asyncio
async def test_the_section_event_still_carries_database_id():
    """`_SECTION_DISPLAY_POSITION` stays where the existing active-section metadata
    needs it. Only the new `sections` array omits it."""
    values = {
        **FULLY_COMPLETE,
        "current_section": SectionID.BACKGROUND,
    }
    events = await collect(values)
    section_events = [e for e in events if e["type"] == "section"]

    assert len(section_events) == 1
    content = section_events[0]["content"]
    assert set(content) == {"database_id", "name", "status"}
    assert content["name"] == "Background"


@pytest.mark.asyncio
async def test_no_event_type_was_renamed():
    """The frontend branches on these names; a rename breaks the UI silently."""
    events = await collect({**FULLY_COMPLETE, "current_section": SectionID.CAREER_GOAL})
    kinds = set(types_of(events))
    assert {"metadata", "message", "section", "completion", "__done__"} >= kinds, kinds
    assert kinds >= {"metadata", "message", "completion", "__done__"}


# --------------------------------------------------------------------------
# Internal-tag suppression must still hold
# --------------------------------------------------------------------------


INTERNAL_TAGS = [
    "skip_stream",
    "internal_extraction",
    "do_not_stream",
    "internal_decision",
    "internal_synthesis",
]


def test_the_suppression_list_is_intact_in_source():
    """Read from source rather than duplicated, so removing a tag on either side
    fails here. `internal_synthesis` is what keeps the artifact JSON out of chat."""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "src" / "service" / "service.py").read_text(
        encoding="utf-8"
    )
    match = re.search(r"if any\(tag in tags for tag in \[([^\]]*)\]\)", source)
    assert match, "suppression list not found"
    suppressed = set(re.findall(r'"([^"]+)"', match.group(1)))
    for tag in INTERNAL_TAGS:
        assert tag in suppressed, tag


@pytest.mark.asyncio
async def test_internally_tagged_tokens_are_still_dropped():
    """An extraction/decision/synthesis chunk must never surface as a token."""
    from langchain_core.messages import AIMessageChunk

    agent = Mock()

    async def astream(**kwargs):
        for tag in INTERNAL_TAGS:
            yield ("messages", (AIMessageChunk(content=f"leak-{tag}"), {"tags": [tag]}))
        yield ("messages", (AIMessageChunk(content="visible"), {"tags": []}))

    async def aget_state(config):
        state = Mock()
        state.values = FULLY_COMPLETE
        return state

    agent.astream = astream
    agent.aget_state = aget_state

    stream_input = StreamInput(message="hi", thread_id=None, user_id=1, stream_tokens=True)
    tokens: list[str] = []
    with patch("service.service.get_agent", Mock(return_value=agent)):
        async for chunk in message_generator(stream_input, "xbuddy"):
            payload = chunk[len("data: ") :].strip()
            if payload == "[DONE]":
                continue
            event = json.loads(payload)
            if event["type"] == "token":
                tokens.append(event["content"])

    assert tokens == ["visible"], tokens
    for tag in INTERNAL_TAGS:
        assert f"leak-{tag}" not in tokens
