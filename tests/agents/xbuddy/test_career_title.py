"""Offline title-source regressions: state -> assembly -> persisted Markdown."""

import pytest

from agents.xbuddy.career_title import TITLE_MAX, career_plan_title
from agents.xbuddy.final_output import render_final_output
from agents.xbuddy.models import FinalOutputDraft, XBuddyData
from agents.xbuddy.synthesis import assemble_final_output


@pytest.mark.parametrize("current,target,goal,expected", [
    ("AI Engineer", "AI Engineer focused on LLM applications and RAG systems", None,
     "AI Engineer Career Plan: LLM applications and RAG systems"),
    ("Backend Engineer", "AI Engineer", None, "Transition from Backend Engineer to AI Engineer"),
    (None, "Product Designer", None, "Product Designer Career Plan"),
    ("N/A", "Product Designer", None, "Product Designer Career Plan"),
    ("AI Engineer", "AI Engineer", "Advance my career", "AI Engineer Career Plan"),
    ("Nurse", "Nurse specializing in pediatric care", None, "Nurse Career Plan: pediatric care"),
    ("  data ANALYST ", "Data Analyst role focused on forecasting", None,
     "Data Analyst Career Plan: forecasting"),
    ("Product Designer", "Product Designer", "I want to focus on accessibility within six months",
     "Product Designer Career Plan: accessibility"),
    (None, "Nurse with a focus on community health", None, "Nurse Career Plan: community health"),
    ("Teacher", "Teacher focused on literacy within six months", None,
     "Teacher Career Plan: literacy"),
    ("AI Engineer", "AI Engineer with LLM focus", None, "AI Engineer Career Plan: LLM"),
    ("Designer", "Designer (accessibility)", None, "Designer (accessibility) Career Plan"),
    ("unemployed", "Designer", None, "Designer Career Plan"),
])
def test_role_and_focus_semantics(current, target, goal, expected):
    profile = XBuddyData(current_role=current, target_roles=[target], career_goal_summary=goal)
    assert career_plan_title(profile) == expected


def test_no_specialization_is_inferred_from_skills_or_actions():
    profile = XBuddyData(current_role="Engineer", target_roles=["Engineer"],
                        current_skills=["Python"], skill_gaps=["Cloud architecture"],
                        action_items=["Build a robotics prototype"])
    assert career_plan_title(profile) == "Engineer Career Plan"


def test_negated_goal_focus_is_not_presented_as_a_desired_specialization():
    profile = XBuddyData(current_role="Designer", target_roles=["Designer"],
                        career_goal_summary="I do not want to focus on advertising")
    assert career_plan_title(profile) == "Designer Career Plan"


@pytest.mark.parametrize("profile,expected", [
    (XBuddyData(), "Your Career Plan"),
    (XBuddyData(career_goal_summary="Explore leadership opportunities"),
     "Career Plan: Explore leadership opportunities"),
    (XBuddyData(current_role="Teacher", target_roles=["Trainer", "Instructional Designer"]),
     "Career Plan: Trainer / Instructional Designer"),
])
def test_missing_or_ambiguous_direction_is_neutral(profile, expected):
    assert career_plan_title(profile) == expected


def test_title_is_one_concise_line():
    title = career_plan_title(XBuddyData(target_roles=["Designer focused on " + "accessible interfaces " * 30]))
    assert len(title) <= TITLE_MAX
    assert "\n" not in title
    assert title.endswith("…")


def test_assembly_ignores_adversarial_model_headline_and_renderer_uses_corrected_source():
    # Even a malformed/legacy draft carrying a model headline cannot decide it.
    draft = FinalOutputDraft(
        positioning_summary="Specialize using the collected experience.",
        strengths_to_leverage=[], skill_priorities=[], search_targets=[],
        action_annotations=[], risks_or_constraints=[],
    ).model_copy(update={"headline": "Transition from AI Engineer to AI Engineer in quantum robotics"})
    result, error = assemble_final_output(draft, XBuddyData(
        current_role="AI Engineer", target_roles=["AI Engineer with a focus on LLM applications and RAG systems"],
    ))
    assert error is None and result is not None
    assert result.headline == "AI Engineer Career Plan: LLM applications and RAG systems"
    assert render_final_output(result).splitlines()[0] == "# " + result.headline
    assert "AI Engineer to AI Engineer" not in result.headline
    assert "quantum" not in result.headline
    assert "headline" not in FinalOutputDraft.model_json_schema()["properties"]
