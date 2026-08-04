"""Tests for section template loading."""

from itertools import pairwise

import pytest

from agents.xbuddy.enums import SectionID
from agents.xbuddy.models import XBuddyData
from agents.xbuddy.prompts import SECTION_TEMPLATES, get_next_section, get_section_template
from agents.xbuddy.sections.base_prompt import BASE_RULES, SectionTemplate

EXPECTED_NAMES = {
    SectionID.CAREER_GOAL: "Career Goal",
    SectionID.BACKGROUND: "Background",
    SectionID.JOB_PREFERENCES: "Job Preferences",
    SectionID.SKILL_ASSESSMENT: "Skill Assessment",
    SectionID.ACTION_PLAN: "Action Plan",
}


def test_get_section_template_resolves_every_section():
    for section_id, name in EXPECTED_NAMES.items():
        template = get_section_template(section_id)
        assert isinstance(template, SectionTemplate)
        assert template.section_id is section_id
        assert template.name == name
        # The string value resolves to the same object.
        assert get_section_template(section_id.value) is template


@pytest.mark.parametrize("bad", ["not_a_section", "", "SECTION_1"])
def test_get_section_template_rejects_unknown_ids(bad):
    with pytest.raises(ValueError, match="Unknown section_id"):
        get_section_template(bad)


def test_section_templates_registry_matches_the_service_convention():
    """service/utils.py looks up SECTION_TEMPLATES[section_id_str].name."""
    assert set(SECTION_TEMPLATES) == {section.value for section in SectionID}
    for key, template in SECTION_TEMPLATES.items():
        assert template.section_id.value == key
        assert template.name.strip(), f"{key} needs a display name"
        assert template.description.strip()


def test_next_section_chain_follows_declaration_order():
    for current, expected in pairwise(SectionID):
        assert get_section_template(current).next_section is expected
    assert get_section_template(SectionID.ACTION_PLAN).next_section is None
    assert get_next_section(SectionID.ACTION_PLAN) is None


def test_required_fields_are_real_xbuddydata_fields():
    """Drift guard: a renamed XBuddyData field must fail here, not in PR 4."""
    known = set(XBuddyData.model_fields)
    for template in SECTION_TEMPLATES.values():
        assert template.required_fields, f"{template.name} declares no required fields"
        unknown = set(template.required_fields) - known
        assert not unknown, f"{template.name} references non-existent fields: {sorted(unknown)}"


def test_every_template_has_substantive_prompt_and_rules():
    for template in SECTION_TEMPLATES.values():
        prompt = template.system_prompt_template
        assert len(prompt) > 200, f"{template.name} prompt looks like a stub"
        # "TODO:" is the starter-template marker; the bare word is legitimate
        # prompt content (BASE_RULES forbids the agent emitting it).
        assert "TODO:" not in prompt, f"{template.name} still contains stub text"
        assert template.validation_rules, f"{template.name} declares no validation rules"
        for rule in template.validation_rules:
            assert rule.field_name in XBuddyData.model_fields
            assert rule.error_message.strip()


def test_career_goal_ships_the_selected_questioning_strategy():
    """The PR 2 A/B experiment selected variant `a_strict`.

    Guards against silently shipping the rejected `b_anchored` approach, which
    scored 2.8/5 on non-leading guidance and broke the one-question rule in 40%
    of cases. evals/section1_career_goal/variants.py imports the same constant
    for its winning arm, so this also pins the two together.
    """
    from agents.xbuddy.sections.section_1 import (
        CAREER_GOAL_BODY,
        CAREER_GOAL_PROMPT,
        CAREER_GOAL_QUESTIONING_STRATEGY,
    )

    # Composition holds, and the template actually ships it.
    assert CAREER_GOAL_BODY.strip() in CAREER_GOAL_PROMPT
    assert CAREER_GOAL_QUESTIONING_STRATEGY.strip() in CAREER_GOAL_PROMPT
    template = get_section_template(SectionID.CAREER_GOAL)
    assert template.system_prompt_template == CAREER_GOAL_PROMPT

    # The strategy is the strict one: forbid examples, one question only.
    strategy = CAREER_GOAL_QUESTIONING_STRATEGY.lower()
    assert "exactly one question" in strategy
    assert "do not offer examples" in strategy

    # And it is not the rejected anchored approach.
    assert "concrete example answers" not in CAREER_GOAL_PROMPT.lower()
    assert CAREER_GOAL_PROMPT.count("QUESTIONING STRATEGY") == 1


def test_base_rules_carry_the_shared_constraints():
    assert "TODO:" not in BASE_RULES, "starter-template marker left in BASE_RULES"
    assert "Replace this with your agent" not in BASE_RULES
    assert "JobBuddy" in BASE_RULES
    # The four rules downstream nodes and the evals rely on.
    lowered = BASE_RULES.lower()
    assert "one question at a time" in lowered
    assert "placeholder" in lowered
    assert "invent facts" in lowered
    assert "current section" in lowered
