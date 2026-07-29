"""Tests for the JobBuddy data model and section enum ordering."""

from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import XBuddyData
from agents.xbuddy.prompts import get_next_section, get_next_unfinished_section
from agents.xbuddy.state_factory import build_section_states

EXPECTED_SECTION_ORDER = [
    SectionID.CAREER_GOAL,
    SectionID.BACKGROUND,
    SectionID.JOB_PREFERENCES,
    SectionID.SKILL_ASSESSMENT,
    SectionID.ACTION_PLAN,
]


# --------------------------------------------------------------------------
# 9. Round-trip / checkpoint serialization contract
# --------------------------------------------------------------------------


def test_xbuddy_data_round_trip_and_defaults():
    # Constructs with no arguments — required for cold start.
    empty = XBuddyData()
    list_fields = [
        name
        for name, field in XBuddyData.model_fields.items()
        if field.annotation == list[str]
    ]
    assert list_fields, "expected at least one list field"
    for name in list_fields:
        assert getattr(empty, name) == []

    # Mutable defaults are per-instance, not shared.
    other = XBuddyData()
    empty.target_roles.append("PM")
    assert other.target_roles == []

    # Partial construction validates — data arrives section by section.
    partial = XBuddyData(target_roles=["Data Scientist"], years_experience=5)
    assert partial.target_roles == ["Data Scientist"]
    assert partial.years_experience == 5
    assert partial.career_goal_summary is None
    assert partial.skill_gaps == []

    # Round trip through the checkpoint representation is lossless.
    populated = XBuddyData(
        target_roles=["Staff Engineer"],
        career_goal_summary="Move into platform work",
        target_timeline="6 months",
        current_role="Senior Engineer",
        years_experience=8,
        highest_education="BSc Computer Science",
        work_history=["Acme 2019-2024"],
        preferred_locations=["Berlin", "Remote"],
        preferred_work_modes=["remote", "hybrid"],
        target_industries=["fintech"],
        employment_types=["full-time"],
        salary_expectation="90-110k EUR",
        strengths=["systems design"],
        current_skills=["Python", "Go"],
        skill_gaps=["Kubernetes"],
        action_items=["Ship a platform RFC"],
    )
    assert XBuddyData.model_validate(populated.model_dump()) == populated

    # Older checkpoints carrying removed fields must not raise.
    restored = XBuddyData.model_validate({"target_roles": ["PM"], "obsolete_field": 123})
    assert restored.target_roles == ["PM"]
    assert not hasattr(restored, "obsolete_field")


# --------------------------------------------------------------------------
# 10. Section enum ordering (get_next_section depends on declaration order)
# --------------------------------------------------------------------------


def test_section_order_and_navigation():
    assert list(SectionID) == EXPECTED_SECTION_ORDER

    # Walking from the first section visits all five, then stops.
    walked = [SectionID.CAREER_GOAL]
    while (nxt := get_next_section(walked[-1])) is not None:
        walked.append(nxt)
    assert walked == EXPECTED_SECTION_ORDER
    assert get_next_section(SectionID.ACTION_PLAN) is None

    # A freshly seeded progress record starts at the first section.
    sections = build_section_states()
    assert get_next_unfinished_section(sections) is SectionID.CAREER_GOAL

    # Marking the first two done advances to the third.
    sections[SectionID.CAREER_GOAL.value].status = SectionStatus.DONE
    sections[SectionID.BACKGROUND.value].status = SectionStatus.DONE
    assert get_next_unfinished_section(sections) is SectionID.JOB_PREFERENCES
