"""Stage 1 tests: section-scoped extraction schemas and the pure merge.

Two properties carry the most weight here:

* **Schema parity.** Each extract model's fields must equal that section's
  `required_fields`. PR 2 already pins those names to XBuddyData, so this closes
  the loop and makes schema drift a test failure rather than a silent Stage 2 bug.
* **Non-destructive merge.** Extraction may add and correct, never forget. A
  dropped field is invisible until the agent re-asks something the user already
  answered.
"""

import pytest
from pydantic import BaseModel, ValidationError

from agents.xbuddy.enums import SectionID
from agents.xbuddy.extraction import (
    extraction_changed,
    get_extract_model,
    merge_extraction,
)
from agents.xbuddy.models import (
    EXTRACT_MODELS,
    ActionPlanExtract,
    BackgroundExtract,
    CareerGoalExtract,
    JobPreferencesExtract,
    SkillAssessmentExtract,
    XBuddyData,
)
from agents.xbuddy.prompts import get_section_template

ALL_EXTRACTS = [
    CareerGoalExtract,
    BackgroundExtract,
    JobPreferencesExtract,
    SkillAssessmentExtract,
    ActionPlanExtract,
]


def nulls_for(model: type[BaseModel]) -> dict:
    """Every field explicitly null — the schema has no defaults to fall back on."""
    return dict.fromkeys(model.model_fields, None)


# --------------------------------------------------------------------------
# Registry and schema parity
# --------------------------------------------------------------------------


def test_registry_covers_every_section():
    assert set(EXTRACT_MODELS) == set(SectionID)
    assert len(EXTRACT_MODELS) == 5
    # No model is reused across two sections.
    assert len(set(EXTRACT_MODELS.values())) == 5


@pytest.mark.parametrize("section", list(SectionID))
def test_extract_fields_match_section_required_fields_exactly(section):
    """Schema drift guard, in both directions.

    A field added to a section template but not the extract model would never be
    extracted; a field on the model but not in required_fields would be
    extracted into something the section does not own.
    """
    model = EXTRACT_MODELS[section]
    required_fields = set(get_section_template(section).required_fields)
    model_fields = set(model.model_fields)

    assert model_fields == required_fields, (
        f"{section.value}: model has {sorted(model_fields)}, "
        f"template requires {sorted(required_fields)}"
    )


@pytest.mark.parametrize("model", ALL_EXTRACTS)
def test_every_extract_field_exists_on_xbuddydata(model):
    unknown = set(model.model_fields) - set(XBuddyData.model_fields)
    assert not unknown, f"{model.__name__} would extract into non-existent fields: {sorted(unknown)}"


@pytest.mark.parametrize("model", ALL_EXTRACTS)
def test_extract_field_types_are_nullable_versions_of_xbuddydata(model):
    """A list field must not become a scalar, or the merge would corrupt state."""
    for name, field in model.model_fields.items():
        # `==`, not `is`: list[str] is a GenericAlias and is not interned, so
        # `annotation is list[str]` is always False and would skip this check.
        target_is_list = XBuddyData.model_fields[name].annotation == list[str]
        annotation = str(field.annotation)
        if target_is_list:
            assert "list[str]" in annotation, f"{model.__name__}.{name} should be list[str] | None"
        assert "None" in annotation, f"{model.__name__}.{name} must be nullable"


# --------------------------------------------------------------------------
# Strict json_schema compatibility (the PR 3 lesson)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_EXTRACTS)
def test_all_fields_required_for_strict_json_schema(model):
    """OpenAI strict mode requires every property in `required`.

    A Pydantic default silently drops a field from `required` and invalidates the
    schema, so this fails the moment anyone adds one.
    """
    schema = model.model_json_schema()
    properties = set(schema["properties"])
    required = set(schema.get("required", []))

    assert properties == required, f"{model.__name__} not required: {sorted(properties - required)}"


@pytest.mark.parametrize("model", ALL_EXTRACTS)
def test_construction_requires_every_field(model):
    """Nothing is optional, so a caller must pass explicit nulls."""
    with pytest.raises(ValidationError):
        model()
    assert model(**nulls_for(model))  # explicit nulls are valid


# --------------------------------------------------------------------------
# get_extract_model
# --------------------------------------------------------------------------


def test_get_extract_model_accepts_enum_and_string():
    assert get_extract_model(SectionID.CAREER_GOAL) is CareerGoalExtract
    assert get_extract_model("career_goal") is CareerGoalExtract


@pytest.mark.parametrize("bad", ["not_a_section", "", "CAREER_GOAL", "section_1"])
def test_get_extract_model_rejects_unknown_sections_loudly(bad):
    with pytest.raises(ValueError, match="Unknown section_id"):
        get_extract_model(bad)


# --------------------------------------------------------------------------
# merge_extraction — applying values
# --------------------------------------------------------------------------


def test_merge_applies_values_to_empty_data():
    extracted = CareerGoalExtract(
        target_roles=["SRE", "Platform Engineer"],
        career_goal_summary="Move into platform work",
        target_timeline="3 months",
    )
    result = merge_extraction(extracted, XBuddyData())

    assert result.target_roles == ["SRE", "Platform Engineer"]
    assert result.career_goal_summary == "Move into platform work"
    assert result.target_timeline == "3 months"


def test_merge_leaves_other_sections_untouched():
    """A Career Goal extraction must not disturb Background or later fields."""
    existing = XBuddyData(current_role="QA Analyst", years_experience=4, strengths=["testing"])
    result = merge_extraction(
        CareerGoalExtract(target_roles=["SRE"], career_goal_summary=None, target_timeline=None),
        existing,
    )

    assert result.target_roles == ["SRE"]
    assert result.current_role == "QA Analyst"
    assert result.years_experience == 4
    assert result.strengths == ["testing"]


def test_merge_handles_int_and_scalar_fields():
    result = merge_extraction(
        BackgroundExtract(
            current_role="SRE", years_experience=8, highest_education=None, work_history=None
        ),
        XBuddyData(),
    )
    assert result.current_role == "SRE"
    assert result.years_experience == 8
    assert result.highest_education is None
    assert result.work_history == []


# --------------------------------------------------------------------------
# merge_extraction — None and [] must never clobber
# --------------------------------------------------------------------------


def test_none_is_a_no_op():
    existing = XBuddyData(
        target_roles=["SRE"], career_goal_summary="platform work", target_timeline="3 months"
    )
    result = merge_extraction(CareerGoalExtract(**nulls_for(CareerGoalExtract)), existing)

    assert result == existing, "all-null extraction must change nothing"


def test_empty_list_is_a_no_op():
    """A model returning [] where it meant null must not wipe stored data."""
    existing = XBuddyData(target_roles=["SRE", "Platform Engineer"])
    result = merge_extraction(
        CareerGoalExtract(target_roles=[], career_goal_summary=None, target_timeline=None),
        existing,
    )

    assert result.target_roles == ["SRE", "Platform Engineer"]


@pytest.mark.parametrize("model", ALL_EXTRACTS)
def test_all_null_extraction_never_clears_any_field(model):
    """Exhaustive across all five sections: a fully-populated record survives."""
    populated = XBuddyData(
        target_roles=["SRE"],
        career_goal_summary="platform work",
        target_timeline="3 months",
        current_role="Senior Engineer",
        years_experience=8,
        highest_education="BSc",
        work_history=["Acme 2019-2024"],
        preferred_locations=["Berlin"],
        preferred_work_modes=["remote"],
        target_industries=["fintech"],
        employment_types=["full-time"],
        salary_expectation="90-110k EUR",
        strengths=["systems design"],
        current_skills=["Python"],
        skill_gaps=["Kubernetes"],
        action_items=["Ship an RFC"],
    )
    result = merge_extraction(model(**nulls_for(model)), populated)
    assert result == populated

    # And the same with [] in every list position. `==` not `is`: list[str] is a
    # GenericAlias and is not interned, so `is` would leave this dict all-None
    # and the [] path would never be exercised.
    empties = {
        name: ([] if XBuddyData.model_fields[name].annotation == list[str] else None)
        for name in model.model_fields
    }
    assert any(value == [] for value in empties.values()), (
        f"{model.__name__}: no list field found, so the [] path is untested"
    )
    assert merge_extraction(model(**empties), populated) == populated


# --------------------------------------------------------------------------
# merge_extraction — corrections
# --------------------------------------------------------------------------


def test_correction_overwrites_a_scalar():
    existing = XBuddyData(target_timeline="3 months")
    result = merge_extraction(
        CareerGoalExtract(target_roles=None, career_goal_summary=None, target_timeline="6 months"),
        existing,
    )
    assert result.target_timeline == "6 months"


def test_correction_replaces_a_list_wholesale():
    """Narrowing must not append: "actually just SRE" replaces the wider list."""
    existing = XBuddyData(target_roles=["SRE", "Platform Engineer", "SWE"])
    result = merge_extraction(
        CareerGoalExtract(target_roles=["SRE"], career_goal_summary=None, target_timeline=None),
        existing,
    )
    assert result.target_roles == ["SRE"]


# --------------------------------------------------------------------------
# Purity and idempotence
# --------------------------------------------------------------------------


def test_input_user_data_is_never_mutated():
    existing = XBuddyData(target_roles=["SRE"], target_timeline="3 months")
    snapshot = existing.model_copy(deep=True)

    result = merge_extraction(
        CareerGoalExtract(
            target_roles=["Platform Engineer"], career_goal_summary="x", target_timeline="6 months"
        ),
        existing,
    )

    assert existing == snapshot, "merge must not mutate its input"
    assert result is not existing
    assert result != existing


def test_returned_lists_do_not_alias_the_input():
    """Deep copy: mutating the result must not reach back into the original."""
    existing = XBuddyData(target_roles=["SRE"])
    result = merge_extraction(CareerGoalExtract(**nulls_for(CareerGoalExtract)), existing)

    assert result.target_roles == ["SRE"]
    result.target_roles.append("mutated")
    assert existing.target_roles == ["SRE"], "result aliases the input's list"


def test_merge_is_idempotent():
    extracted = CareerGoalExtract(
        target_roles=["SRE"], career_goal_summary="platform work", target_timeline="3 months"
    )
    once = merge_extraction(extracted, XBuddyData())
    twice = merge_extraction(extracted, once)
    thrice = merge_extraction(extracted, twice)

    assert once == twice == thrice


def test_no_op_merge_returns_an_equal_but_independent_copy():
    existing = XBuddyData(target_roles=["SRE"])
    result = merge_extraction(CareerGoalExtract(**nulls_for(CareerGoalExtract)), existing)

    assert result == existing
    assert result is not existing


# --------------------------------------------------------------------------
# extraction_changed
# --------------------------------------------------------------------------


def test_extraction_changed_detects_difference():
    before = XBuddyData()
    after = merge_extraction(
        CareerGoalExtract(target_roles=["SRE"], career_goal_summary=None, target_timeline=None),
        before,
    )
    assert extraction_changed(before, after) is True


def test_extraction_changed_false_for_a_no_op():
    before = XBuddyData(target_roles=["SRE"])
    after = merge_extraction(CareerGoalExtract(**nulls_for(CareerGoalExtract)), before)
    assert extraction_changed(before, after) is False


# --------------------------------------------------------------------------
# Round-trip: the merged record must still survive the checkpoint contract
# --------------------------------------------------------------------------


def test_merged_data_still_round_trips():
    merged = merge_extraction(
        JobPreferencesExtract(
            preferred_locations=["Berlin", "Remote"],
            preferred_work_modes=["remote"],
            target_industries=["fintech"],
            employment_types=["full-time"],
            salary_expectation="90-110k EUR",
        ),
        XBuddyData(target_roles=["SRE"]),
    )
    assert XBuddyData.model_validate(merged.model_dump()) == merged
