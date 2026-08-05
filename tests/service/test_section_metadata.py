"""Service-boundary tests for section metadata (PR 3 compatibility fixes).

Two regressions are pinned here:

1. `langchain_to_chat_message` used to raise `NameError: SECTION_ID_MAPPING` for
   any message carrying section metadata — which service.py attaches to every
   human message itself.
2. Section ids must stay strings. `section_states.section_id` is TEXT and
   `save_section_state()` takes a str, so there is no integer identity to map to.
"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy.enums import SectionID
from service.utils import langchain_to_chat_message

SECTION_KWARGS = {"section_id": "career_goal", "agent_name": "xbuddy"}


@pytest.mark.parametrize(
    "message",
    [
        HumanMessage(content="I need a new job", additional_kwargs=dict(SECTION_KWARGS)),
        AIMessage(content="What role next?", additional_kwargs=dict(SECTION_KWARGS)),
    ],
    ids=["human", "ai"],
)
def test_section_metadata_does_not_raise_and_stays_a_string(message):
    chat_message = langchain_to_chat_message(message)

    assert chat_message.custom_data["section_id"] == "career_goal"
    assert isinstance(chat_message.custom_data["section_id"], str)
    assert chat_message.custom_data["agent_name"] == "xbuddy"
    # The fallback in _get_section_name title-cases the id when no template
    # module is registered for the agent, which happens to be correct here.
    assert chat_message.custom_data["section_name"] == "Career Goal"


def test_messages_without_section_metadata_are_unaffected():
    assert langchain_to_chat_message(HumanMessage(content="hi")).custom_data == {}
    assert langchain_to_chat_message(AIMessage(content="hi")).custom_data == {}


def test_display_position_covers_every_section_and_is_service_local():
    """The 1-5 badge is a display position, not a database id.

    It must never leak into the agent package or the persistence path — the DB
    column is TEXT and stores the SectionID string value.
    """
    from service.service import _SECTION_DISPLAY_POSITION

    assert set(_SECTION_DISPLAY_POSITION) == {section.value for section in SectionID}
    assert sorted(_SECTION_DISPLAY_POSITION.values()) == [1, 2, 3, 4, 5]

    src = Path("src")

    # Not referenced anywhere in the agent package.
    agent_hits = [
        path
        for path in (src / "agents").rglob("*.py")
        if "_SECTION_DISPLAY_POSITION" in path.read_text(encoding="utf-8")
    ]
    assert not agent_hits, f"display badge leaked into the agent layer: {agent_hits}"

    # Not referenced on the persistence path.
    persistence = (src / "integrations" / "supabase" / "supabase_client.py").read_text(
        encoding="utf-8"
    )
    assert "_SECTION_DISPLAY_POSITION" not in persistence
    # And that path still takes a string section id.
    assert "section_id: str" in persistence


def test_no_undefined_section_symbols_remain_on_the_invoke_or_stream_path():
    """`SECTION_ID_MAPPING` was undefined everywhere it was used."""
    service_src = Path("src/service/service.py").read_text(encoding="utf-8")
    utils_src = Path("src/service/utils.py").read_text(encoding="utf-8")

    assert "SECTION_ID_MAPPING" not in service_src
    assert "SECTION_ID_MAPPING" not in utils_src
    # FOUNDER_BUDDY_TEMPLATES survives only in notify_section_update, which is
    # independently broken (get_section_string_id is also undefined) and is
    # deliberately out of PR 3 scope.
    assert service_src.count("FOUNDER_BUDDY_TEMPLATES") == 1
