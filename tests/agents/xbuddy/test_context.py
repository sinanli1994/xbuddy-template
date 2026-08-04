"""Tests for context packet assembly and the get_context tool."""

import json

import pytest

from agents.xbuddy.context import (
    build_context_packet,
    build_system_prompt,
    render_known_data,
)
from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import ContextPacket, SectionContent, XBuddyData
from agents.xbuddy.prompts import get_section_template
from agents.xbuddy.sections.base_prompt import BASE_RULES
from agents.xbuddy.tools import get_context


def test_build_context_packet_composes_prompt_and_rules():
    draft = SectionContent(content={"type": "doc"}, plain_text="Aiming for SRE lead")
    packet = build_context_packet(
        section_id=SectionID.CAREER_GOAL,
        status=SectionStatus.IN_PROGRESS,
        draft=draft,
        user_data=XBuddyData(target_roles=["SRE Lead"]),
    )

    assert packet.section_id is SectionID.CAREER_GOAL
    assert packet.status is SectionStatus.IN_PROGRESS
    assert packet.draft is draft

    # Shared rules and the section's own prompt are both present.
    assert BASE_RULES.strip()[:40] in packet.system_prompt
    assert "CURRENT SECTION: Career Goal" in packet.system_prompt

    # The list->dict bridge between SectionTemplate and ContextPacket.
    template = get_section_template(SectionID.CAREER_GOAL)
    assert packet.validation_rules["required_fields"] == template.required_fields
    assert len(packet.validation_rules["rules"]) == len(template.validation_rules)
    assert packet.validation_rules["rules"][0]["field_name"] == "target_roles"


def test_shipped_career_goal_block_order():
    """Pin the shipped composition order for Career Goal.

    The PR 2 A/B arms appended the questioning strategy *after* the KNOWN SO FAR
    block; the shipped template carries it inside system_prompt_template, so it
    lands before. This asserts the shipped order structurally — the
    `shipped` confirmation arm in evals/ covers it behaviourally.
    """
    from agents.xbuddy.sections.section_1 import (
        CAREER_GOAL_BODY,
        CAREER_GOAL_QUESTIONING_STRATEGY,
    )

    prompt = build_system_prompt(
        get_section_template(SectionID.CAREER_GOAL), XBuddyData(target_roles=["SRE"])
    )

    # Anchor on the block *header*, not the bare phrase: BASE_RULES also
    # cross-references "KNOWN SO FAR", so a plain .index() finds that first.
    known_block = "\n\nKNOWN SO FAR\n"
    assert prompt.count(known_block) == 1

    positions = {
        "base_rules": prompt.index(BASE_RULES.strip()[:40]),
        "body": prompt.index("CURRENT SECTION: Career Goal"),
        "strategy": prompt.index(CAREER_GOAL_QUESTIONING_STRATEGY.strip()),
        "known": prompt.index(known_block),
    }
    assert (
        positions["base_rules"] < positions["body"] < positions["strategy"] < positions["known"]
    ), f"unexpected block order: {sorted(positions, key=positions.get)}"

    # No duplication introduced by the body/strategy split.
    assert prompt.count("QUESTIONING STRATEGY") == 1
    assert prompt.count(CAREER_GOAL_BODY.strip()) == 1


def test_build_context_packet_defaults_are_safe():
    packet = build_context_packet(section_id=SectionID.ACTION_PLAN)
    assert packet.status is SectionStatus.PENDING
    assert packet.draft is None
    assert "Nothing collected yet" in packet.system_prompt


def test_prompt_assembly_survives_literal_braces():
    """Templates contain JSON/Tiptap braces; .format() would raise on them."""
    template = get_section_template(SectionID.CAREER_GOAL).model_copy(
        update={"system_prompt_template": 'Example: {"type": "doc"} and {unclosed'}
    )
    prompt = build_system_prompt(template, XBuddyData())
    assert '{"type": "doc"}' in prompt
    assert "{unclosed" in prompt


def test_render_known_data_includes_filled_and_omits_empty():
    data = XBuddyData(
        target_roles=["SRE", "Platform Engineer"],
        years_experience=8,
        salary_expectation="90-110k EUR",
    )
    rendered = render_known_data(data)

    assert "Target role(s): SRE, Platform Engineer" in rendered
    assert "Years of experience: 8" in rendered
    assert "Salary expectation: 90-110k EUR" in rendered

    # Unfilled fields must be absent entirely — no placeholder stand-ins.
    assert "Skill gaps" not in rendered
    assert "Education" not in rendered
    for placeholder in ("[TBD]", "[Not provided]", "None", "TODO", "[]"):
        assert placeholder not in rendered


def test_render_known_data_empty_for_fresh_data():
    assert render_known_data(XBuddyData()) == ""


def test_known_data_block_appears_in_prompt():
    template = get_section_template(SectionID.BACKGROUND)
    prompt = build_system_prompt(template, XBuddyData(current_role="SRE"))
    assert "KNOWN SO FAR" in prompt
    assert "Current role: SRE" in prompt


@pytest.mark.asyncio
async def test_get_context_tool_round_trips_to_a_context_packet():
    raw = await get_context.ainvoke(
        {
            "user_id": 7,
            "thread_id": "t-ctx",
            "section_id": "job_preferences",
            "user_data": {"preferred_locations": ["Berlin"]},
            "status": "in_progress",
            "draft": {"content": {"type": "doc"}, "plain_text": "remote only"},
        }
    )

    assert isinstance(raw, dict)
    # Must be JSON-serializable for an LLM/API caller.
    json.dumps(raw)

    packet = ContextPacket.model_validate(raw)
    assert packet.section_id is SectionID.JOB_PREFERENCES
    assert packet.status is SectionStatus.IN_PROGRESS
    assert packet.draft.plain_text == "remote only"
    assert "Berlin" in packet.system_prompt
    assert packet.validation_rules["required_fields"] == ["preferred_locations", "preferred_work_modes", "target_industries", "employment_types", "salary_expectation"]


@pytest.mark.asyncio
async def test_get_context_tool_minimal_arguments():
    raw = await get_context.ainvoke(
        {"user_id": 1, "thread_id": "t", "section_id": "career_goal"}
    )
    packet = ContextPacket.model_validate(raw)
    assert packet.status is SectionStatus.PENDING
    assert packet.draft is None


@pytest.mark.asyncio
async def test_get_context_tool_and_builder_agree():
    """The tool must be a thin wrapper, not a second implementation."""
    raw = await get_context.ainvoke(
        {
            "user_id": 1,
            "thread_id": "t",
            "section_id": "skill_assessment",
            "user_data": {"strengths": ["systems design"]},
            "status": "in_progress",
        }
    )
    direct = build_context_packet(
        section_id=SectionID.SKILL_ASSESSMENT,
        status=SectionStatus.IN_PROGRESS,
        user_data=XBuddyData(strengths=["systems design"]),
    )
    assert ContextPacket.model_validate(raw) == direct
