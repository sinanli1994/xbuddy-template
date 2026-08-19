"""Offline tests for the PR 5 Stage 6 final-artifact eval.

The deterministic evaluators are pure, so the whole scoring contract is provable
here without a key or a network. The judges are exercised only through their prompt
builder and their failure paths — a judge's *score* is a model output and is not
something to assert offline.

The dataset-sync path is PR 4's `evals/memory/sync.py`, reused rather than
reimplemented; one test proves this eval's reference data survives it.
"""

import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[3] / "evals" / "final_output"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from fo_dataset import (
    CASES,
    SECTION_FIELDS,
    as_langsmith_examples,
    build_profile,
    populated_sections,
)
from fo_evaluators import (
    DETERMINISTIC_EVALUATORS,
    action_items_preserved,
    artifact_complete,
    grounding_no_invention,
    priority_valid,
    render_deterministic,
    unknowns_honest,
)
from fo_judges import RUBRICS, build_judge_prompt

from agents.xbuddy.final_output import render_final_output
from agents.xbuddy.models import ActionItem, FinalOutput, XBuddyData
from agents.xbuddy.synthesis import derive_unknowns


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    """No test in this file may construct a real model."""

    def explode(*args, **kwargs):
        raise AssertionError("a test tried to build a real LLM chain")

    monkeypatch.setattr("core.llm.get_model", explode)


def case_by_id(case_id: str) -> tuple[dict, dict, XBuddyData]:
    """Return (case, reference_outputs, profile) for one case id."""
    index = next(n for n, entry in enumerate(CASES) if entry["id"] == case_id)
    _inputs, outputs = as_langsmith_examples()
    return CASES[index], outputs[index], build_profile(CASES[index])


def artifact(profile: XBuddyData, **overrides) -> dict:
    """A well-formed clean artifact for a profile, in runner-output shape.

    Deliberately built *from* the profile rather than filled with placeholders. A real
    artifact's positioning names the current role and seniority, and its search
    targets name the stated locations and industries — an earlier version of this
    helper left those out and made `artifact_complete` look broken when the helper
    was what was unrealistic.
    """
    steps = list(profile.action_items)
    background = ", ".join(
        str(value)
        for value in (
            profile.current_role,
            f"{profile.years_experience} years" if profile.years_experience else None,
            profile.highest_education,
            *profile.work_history,
        )
        if value
    )
    values = {
        "headline": f"Toward {', '.join(profile.target_roles) or 'a new role'}",
        "positioning_summary": (
            f"{background or 'Background not yet collected'}. "
            f"Goal: {profile.career_goal_summary or 'not yet stated'} "
            f"({profile.target_timeline or 'no timeline'})."
        ),
        "strengths_to_leverage": list(profile.strengths),
        "skill_priorities": list(profile.skill_gaps),
        "search_targets": [
            *profile.target_industries,
            *profile.preferred_locations,
            *profile.preferred_work_modes,
            *profile.employment_types,
            *([profile.salary_expectation] if profile.salary_expectation else []),
        ],
        "action_items": [
            ActionItem(step=step, rationale=f"reason {index + 1}", priority=index + 1, timeframe=None)
            for index, step in enumerate(steps)
        ],
        "risks_or_constraints": [],
        "unknowns": derive_unknowns(profile),
    }
    values.update(overrides)
    final_output = FinalOutput(**values)
    markdown = render_final_output(final_output)
    return {
        "structured": final_output.model_dump(),
        "markdown": markdown,
        "render_repeat": markdown,
    }


# --------------------------------------------------------------------------
# Dataset integrity
# --------------------------------------------------------------------------


def test_dataset_size_and_unique_ids():
    assert 8 <= len(CASES) <= 12, f"expected 8-12 cases, got {len(CASES)}"
    assert len({case["id"] for case in CASES}) == len(CASES)


def test_every_case_profile_is_a_valid_xbuddydata():
    known = set(XBuddyData.model_fields)
    for case in CASES:
        stray = set(case["profile"]) - known
        assert not stray, f"{case['id']}: unknown fields {sorted(stray)}"
        build_profile(case)


def test_every_case_has_a_confirmed_action_plan():
    """The plan is the artifact's spine; a case without one tests nothing useful."""
    for case in CASES:
        assert build_profile(case).action_items, case["id"]


def test_the_dataset_covers_the_required_situations():
    _inputs, outputs = as_langsmith_examples()
    unknown_counts = [len(entry["unknowns"]) for entry in outputs]
    plan_lengths = [len(entry["confirmed_action_items"]) for entry in outputs]

    assert min(unknown_counts) == 0, "need a fully populated case"
    assert max(unknown_counts) >= 10, "need a many-unknowns case"
    assert max(plan_lengths) >= 5, "need a longer confirmed plan"
    assert any("transition" in case["id"] or "teacher" in case["id"] for case in CASES)
    assert any("timeline" in case["id"] for case in CASES)


def test_unknowns_come_from_production_code_not_by_hand():
    """The reference must be derived, or it drifts from the behaviour it scores."""
    _inputs, outputs = as_langsmith_examples()
    for case, reference in zip(CASES, outputs, strict=True):
        assert reference["unknowns"] == derive_unknowns(build_profile(case)), case["id"]


def test_section_fields_cover_every_xbuddydata_field():
    """Otherwise `artifact_complete` would silently ignore a whole field."""
    mapped = {field for fields in SECTION_FIELDS.values() for field in fields}
    assert mapped == set(XBuddyData.model_fields)


def test_expected_sections_are_the_populated_ones():
    _inputs, outputs = as_langsmith_examples()
    for case, reference in zip(CASES, outputs, strict=True):
        assert reference["expected_sections"] == populated_sections(build_profile(case))


@pytest.mark.parametrize(
    "key", ["confirmed_facts", "confirmed_action_items", "unknowns", "expected_sections"]
)
def test_every_evaluator_consumed_key_is_exported(key):
    """PR 4's lesson: a locally-correct reference proves nothing if it never uploads."""
    _inputs, outputs = as_langsmith_examples()
    for entry in outputs:
        assert key in entry


def test_the_reference_survives_the_pr4_sync_planner():
    """Reuses `evals/memory/sync.py`, the module that exists because of the stale-
    dataset failure. If this eval's shape broke it, every run would be stale."""
    import importlib.util

    location = Path(__file__).resolve().parents[3] / "evals" / "memory" / "sync.py"
    spec = importlib.util.spec_from_file_location("pr4_sync_undertest", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    inputs, outputs = as_langsmith_examples()
    plan = module.plan_sync(inputs, outputs, [])
    assert len(plan.creates) == len(CASES)
    assert plan.updates == []

    class Stored:
        def __init__(self, case_inputs, case_outputs, index):
            self.id = f"uuid-{index}"
            self.inputs = dict(case_inputs)
            self.outputs = dict(case_outputs)

    stored = [
        Stored(i, o, n) for n, (i, o) in enumerate(zip(inputs, outputs, strict=True))
    ]
    assert module.plan_sync(inputs, outputs, stored).is_noop


# --------------------------------------------------------------------------
# grounding_no_invention
# --------------------------------------------------------------------------


def test_a_grounded_artifact_passes():
    _case, reference, profile = case_by_id("fully_populated_sre")
    assert grounding_no_invention(artifact(profile), reference)["score"] == 1


def test_recommendations_naming_unowned_technology_are_not_invented_facts():
    """The false-positive that matters: "learn Kubernetes" is the product working.

    `skill_priorities` and `search_targets` are recommendation fields and are never
    inspected for entity support, even when the profile has no skill gaps at all.
    """
    _case, reference, profile = case_by_id("no_salary_no_gaps")
    output = artifact(
        profile,
        skill_priorities=["Kubernetes", "AWS", "Rust"],
        search_targets=["Kubernetes-heavy platform teams"],
    )
    assert grounding_no_invention(output, reference)["score"] == 1


def test_a_duration_inside_a_recommendation_is_not_a_seniority_claim():
    """Recommendations legitimately mention spans of time.

    "aim for the depth most teams expect after two years" is advice about a target,
    not a claim that the user has two years' experience — and the profile says four.
    This is the assertion that fails if the recommendation fields are ever folded
    into the fact-asserting set.
    """
    _case, reference, profile = case_by_id("fully_populated_sre")
    assert profile.years_experience == 4
    output = artifact(
        profile,
        skill_priorities=["Kubernetes, to the depth most teams expect after two years"],
        search_targets=["platform teams budgeting 120k EUR for the role"],
    )

    result = grounding_no_invention(output, reference)
    assert result["score"] == 1, result["comment"]


def test_a_salary_invented_when_none_was_collected_fails():
    _case, reference, profile = case_by_id("no_salary_no_gaps")
    output = artifact(
        profile, positioning_summary="Targeting around 90k EUR in the Lisbon market."
    )

    result = grounding_no_invention(output, reference)
    assert result["score"] == 0
    assert "salary was never collected" in result["comment"]


def test_a_salary_matching_the_collected_value_passes():
    """A collected figure may be restated — that is grounding, not invention."""
    _case, reference, profile = case_by_id("fully_populated_sre")
    output = artifact(profile, positioning_summary="Aiming at the 75-85k EUR band.")
    assert grounding_no_invention(output, reference)["score"] == 1


def test_a_salary_figure_outside_the_collected_value_fails():
    _case, reference, profile = case_by_id("fully_populated_sre")
    output = artifact(profile, positioning_summary="Aiming at 140k EUR.")

    result = grounding_no_invention(output, reference)
    assert result["score"] == 0
    assert "not in the collected value" in result["comment"]


def test_experience_claimed_when_never_collected_fails():
    _case, reference, profile = case_by_id("many_unknowns_minimal_profile")
    output = artifact(profile, positioning_summary="Around six years of experience so far.")

    result = grounding_no_invention(output, reference)
    assert result["score"] == 0
    assert "never collected" in result["comment"]


def test_experience_disagreeing_with_the_collected_value_fails():
    _case, reference, profile = case_by_id("fully_populated_sre")
    output = artifact(profile, positioning_summary="Nine years of hands-on delivery.")

    result = grounding_no_invention(output, reference)
    assert result["score"] == 0
    assert "disagrees with 4" in result["comment"]


def test_a_strength_with_no_support_in_the_profile_fails():
    """The seeded degradation's signature move: claiming capability, not suggesting it."""
    _case, reference, profile = case_by_id("no_salary_no_gaps")
    output = artifact(
        profile, strengths_to_leverage=["AWS and Kubernetes operations at scale"]
    )

    result = grounding_no_invention(output, reference)
    assert result["score"] == 0
    assert "no support in the profile" in result["comment"]


def test_short_real_skills_are_not_reported_as_fabricated():
    """Found by the first seeded run, not in review.

    `_tokens` drops words under four characters, so SQL, dbt, AWS, Go and R could never
    match anything and every one of them read as an invented capability. The seeded arm
    flagged `'SQL'` and `'dbt'` against a profile whose `current_skills` listed both —
    a false positive in the evaluator, which would have been reported as an agent
    finding.
    """
    _case, reference, profile = case_by_id("contract_only_senior")
    assert "SQL" in profile.current_skills and "dbt" in profile.current_skills

    output = artifact(profile, strengths_to_leverage=["SQL", "dbt", "Snowflake"])
    result = grounding_no_invention(output, reference)
    assert result["score"] == 1, result["comment"]


def test_a_short_skill_absent_from_the_profile_is_still_flagged():
    """The fallback must not become a blanket exemption for short tokens."""
    _case, reference, profile = case_by_id("contract_only_senior")
    assert "AWS" not in str(profile.model_dump())

    result = grounding_no_invention(
        artifact(profile, strengths_to_leverage=["AWS"]), reference
    )
    assert result["score"] == 0
    assert "no support in the profile" in result["comment"]


def test_a_rephrased_strength_still_counts_as_supported():
    """Token overlap, not exact matching: the model may write it in its own words."""
    _case, reference, profile = case_by_id("fully_populated_sre")
    output = artifact(
        profile, strengths_to_leverage=["debugging complex systems under pressure"]
    )
    assert grounding_no_invention(output, reference)["score"] == 1


def test_a_rationale_may_not_smuggle_in_an_invented_fact():
    """`rationale` is a fact-asserting field even though `step` is not."""
    _case, reference, profile = case_by_id("no_salary_no_gaps")
    items = [
        ActionItem(
            step=step,
            rationale="Because you already earn 95k EUR, aim higher.",
            priority=index + 1,
            timeframe=None,
        )
        for index, step in enumerate(profile.action_items)
    ]
    assert grounding_no_invention(artifact(profile, action_items=items), reference)["score"] == 0


# --------------------------------------------------------------------------
# action_items_preserved
# --------------------------------------------------------------------------


def test_preserved_plan_passes():
    _case, reference, profile = case_by_id("long_confirmed_plan")
    result = action_items_preserved(artifact(profile), reference)
    assert result["score"] == 1
    assert "6 confirmed steps" in result["comment"]


def test_a_reworded_step_fails():
    _case, reference, profile = case_by_id("fully_populated_sre")
    items = [
        ActionItem(step=step, rationale="r", priority=index + 1, timeframe=None)
        for index, step in enumerate(profile.action_items)
    ]
    items[0] = ActionItem(step="Update the CV", rationale="r", priority=1, timeframe=None)

    result = action_items_preserved(artifact(profile, action_items=items), reference)
    assert result["score"] == 0
    assert "dropped or reworded" in result["comment"]


def test_a_reordered_plan_fails():
    _case, reference, profile = case_by_id("fully_populated_sre")
    steps = list(profile.action_items)
    reordered = [steps[1], steps[0], steps[2]]
    items = [
        ActionItem(step=step, rationale="r", priority=index + 1, timeframe=None)
        for index, step in enumerate(reordered)
    ]

    result = action_items_preserved(artifact(profile, action_items=items), reference)
    assert result["score"] == 0
    assert "reordered" in result["comment"]


def test_an_added_step_fails():
    _case, reference, profile = case_by_id("fully_populated_sre")
    steps = [*list(profile.action_items), "Also learn Rust"]
    items = [
        ActionItem(step=step, rationale="r", priority=index + 1, timeframe=None)
        for index, step in enumerate(steps)
    ]

    result = action_items_preserved(artifact(profile, action_items=items), reference)
    assert result["score"] == 0
    assert "not confirmed by the user" in result["comment"]


# --------------------------------------------------------------------------
# unknowns_honest
# --------------------------------------------------------------------------


def test_declared_unknowns_matching_the_derivation_pass():
    _case, reference, profile = case_by_id("partial_no_preferences")
    assert unknowns_honest(artifact(profile), reference)["score"] == 1


def test_a_dropped_unknown_fails():
    _case, reference, profile = case_by_id("partial_no_preferences")
    partial = derive_unknowns(profile)[:-1]

    result = unknowns_honest(artifact(profile, unknowns=partial), reference)
    assert result["score"] == 0
    assert "not declared" in result["comment"]


def test_claiming_a_collected_field_is_unknown_fails():
    _case, reference, profile = case_by_id("fully_populated_sre")
    result = unknowns_honest(
        artifact(profile, unknowns=["Salary expectation was never discussed"]), reference
    )
    assert result["score"] == 0
    assert "but it was collected" in result["comment"]


def test_contradicting_an_unknown_elsewhere_fails():
    """The honesty failure that reads best: gap declared, value asserted anyway."""
    _case, reference, profile = case_by_id("no_salary_no_gaps")
    output = artifact(
        profile, positioning_summary="Comfortable targeting 88k EUR."
    )

    result = unknowns_honest(output, reference)
    assert result["score"] == 0
    assert "salary is declared unknown yet a figure is asserted" in result["comment"]


def test_a_case_with_nothing_missing_passes_with_an_empty_list():
    _case, reference, profile = case_by_id("fully_populated_sre")
    assert reference["unknowns"] == []
    assert unknowns_honest(artifact(profile, unknowns=[]), reference)["score"] == 1


# --------------------------------------------------------------------------
# artifact_complete
# --------------------------------------------------------------------------


def test_a_complete_artifact_passes():
    _case, reference, profile = case_by_id("fully_populated_sre")
    result = artifact_complete(artifact(profile), reference)
    assert result["score"] == 1
    assert "5 populated sections" in result["comment"]


def test_an_artifact_ignoring_a_populated_section_fails():
    """Preferences were collected and the document never mentions any of them."""
    _case, reference, profile = case_by_id("fully_populated_sre")
    output = artifact(
        profile,
        headline="A move",
        positioning_summary="Debugging and automation.",
        strengths_to_leverage=["systems debugging"],
        skill_priorities=[],
        search_targets=[],
    )
    # Strip every preference token from the rendered document.
    output["markdown"] = output["markdown"].replace("Berlin", "").replace("fintech", "")
    output["markdown"] = output["markdown"].replace("hybrid", "").replace("full-time", "")
    output["markdown"] = output["markdown"].replace("75-85k EUR", "")

    result = artifact_complete(output, reference)
    assert result["score"] == 0
    assert "job_preferences" in result["comment"]


def test_a_minimal_profile_only_requires_its_populated_sections():
    _case, reference, profile = case_by_id("many_unknowns_minimal_profile")
    assert reference["expected_sections"] == ["career_goal", "action_plan"]
    assert artifact_complete(artifact(profile), reference)["score"] == 1


# --------------------------------------------------------------------------
# priority_valid and render_deterministic
# --------------------------------------------------------------------------


def test_contiguous_priorities_rendered_in_order_pass():
    _case, reference, profile = case_by_id("long_confirmed_plan")
    assert priority_valid(artifact(profile), reference)["score"] == 1


def test_non_contiguous_priorities_fail_even_though_the_schema_forbids_them():
    """Belt and braces, and reachable only via `model_construct`.

    `FinalOutput.priorities_form_a_total_order` normally makes this impossible, so a
    valid model cannot express it. The evaluator scores a `model_dump()` dict though,
    and would be the only thing left if that validator were ever dropped.
    """
    _case, reference, profile = case_by_id("fully_populated_sre")
    output = artifact(profile)
    items = output["structured"]["action_items"]
    items[1]["priority"] = 9

    result = priority_valid(output, reference)
    assert result["score"] == 0
    assert "contiguous total order" in result["comment"]


def test_a_gap_in_the_rendered_numbering_fails():
    _case, reference, profile = case_by_id("fully_populated_sre")
    output = artifact(profile)
    output["markdown"] = output["markdown"].replace("2. **", "9. **")

    result = priority_valid(output, reference)
    assert result["score"] == 0
    assert "numbered item" in result["comment"]


def test_rendered_order_disagreeing_with_priority_order_fails():
    _case, reference, profile = case_by_id("fully_populated_sre")
    output = artifact(profile)
    lines = output["markdown"].splitlines()
    first = next(n for n, line in enumerate(lines) if line.startswith("1. **"))
    second = next(n for n, line in enumerate(lines) if line.startswith("2. **"))
    lines[first], lines[second] = lines[second], lines[first]
    output["markdown"] = "\n".join(lines)

    result = priority_valid(output, reference)
    assert result["score"] == 0
    assert "rendered order disagrees" in result["comment"]


def test_identical_renders_pass():
    _case, reference, profile = case_by_id("fully_populated_sre")
    assert render_deterministic(artifact(profile), reference)["score"] == 1


def test_a_differing_repeat_render_fails():
    _case, reference, profile = case_by_id("fully_populated_sre")
    output = artifact(profile)
    output["render_repeat"] = output["markdown"] + "drift"

    result = render_deterministic(output, reference)
    assert result["score"] == 0
    assert "re-render differed" in result["comment"]


def test_a_missing_repeat_render_fails_rather_than_passing_silently():
    _case, reference, profile = case_by_id("fully_populated_sre")
    output = artifact(profile)
    del output["render_repeat"]
    assert render_deterministic(output, reference)["score"] == 0


# --------------------------------------------------------------------------
# Every case scores clean on a well-formed artifact
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case_id", [case["id"] for case in CASES])
def test_a_well_formed_artifact_scores_clean_for_every_case(case_id):
    """If any case cannot be satisfied, the eval measures the dataset, not the agent."""
    _case, reference, profile = case_by_id(case_id)
    output = artifact(profile)
    for evaluator in DETERMINISTIC_EVALUATORS:
        result = evaluator(output, reference)
        assert result["score"] == 1, f"{case_id} / {result['key']}: {result['comment']}"


def test_the_seeded_degradation_shape_is_detected_across_the_dataset():
    """Simulates what the seeded arm produces: unknowns withheld and gaps invented.

    Offline proof that the deterministic checks would catch it, independent of the
    paid run. Only cases with something actually missing can degrade — a fully
    populated profile has no gap to invent, which is why the assertion is on the
    subset rather than on all ten.
    """
    degraded_cases = []
    for case in CASES:
        _case, reference, profile = case_by_id(case["id"])
        if not reference["unknowns"]:
            continue
        output = artifact(
            profile,
            positioning_summary="A seasoned professional targeting around 95k EUR.",
            strengths_to_leverage=["AWS and Kubernetes platform operations"],
            unknowns=[],
        )
        grounding = grounding_no_invention(output, reference)["score"]
        honesty = unknowns_honest(output, reference)["score"]
        if grounding == 0 and honesty == 0:
            degraded_cases.append(case["id"])

    with_gaps = [case["id"] for case in CASES if derive_unknowns(build_profile(case))]
    assert degraded_cases == with_gaps, (
        "every case with a real gap must be caught by both grounding and honesty"
    )


# --------------------------------------------------------------------------
# A failed synthesis must not pass vacuously
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "evaluator",
    DETERMINISTIC_EVALUATORS,
    ids=lambda ev: ev.__name__,
)
def test_no_artifact_scores_zero_on_every_deterministic_evaluator(evaluator):
    """Found in the first live run, not in review.

    A case whose synthesis 429'd produced empty outputs, and three evaluators scored
    1.00 on it: empty prose contains no unsupported claim, zero action items are
    trivially a contiguous 1..0, and "" re-renders to "". A total failure read as
    partial success.
    """
    _case, reference, _profile = case_by_id("fully_populated_sre")
    empty = {"structured": None, "markdown": "", "render_repeat": ""}

    result = evaluator(empty, reference)
    assert result["score"] == 0, f"{result['key']} passed with no artifact"
    assert "no artifact" in result["comment"]


# --------------------------------------------------------------------------
# Judges: prompt construction only. A judge's score is a model output.
# --------------------------------------------------------------------------


def test_every_judged_dimension_has_a_rubric_with_all_five_levels():
    assert set(RUBRICS) == {
        "coherence",
        "relevance_to_user",
        "actionability",
        "prioritization_quality",
    }
    for dimension, rubric in RUBRICS.items():
        for level in ("1 —", "2 —", "3 —", "4 —", "5 —"):
            assert level in rubric, f"{dimension} has no {level} description"


def test_the_judge_prompt_carries_the_profile_and_the_gaps():
    """Without the profile, `relevance_to_user` cannot distinguish tailored from
    generic; without the gaps, a judge would read honest unknowns as incompleteness."""
    _case, reference, _profile = case_by_id("partial_no_preferences")
    prompt = build_judge_prompt("relevance_to_user", "# Doc\n", reference)

    assert "RUBRIC" in prompt
    assert "Data Engineer" in prompt
    assert "never collected" in prompt.lower()
    assert "# Doc" in prompt


def test_the_judge_prompt_names_only_the_dimension_being_scored():
    prompt = build_judge_prompt("coherence", "# Doc\n", {})
    assert "DIMENSION: coherence" in prompt
    for other in ("actionability", "prioritization_quality", "relevance_to_user"):
        assert f"DIMENSION: {other}" not in prompt


@pytest.mark.asyncio
async def test_a_judge_failure_scores_none_rather_than_zero(monkeypatch):
    """Zero would be indistinguishable from a genuinely bad document and would
    quietly corrupt the aggregate."""
    import fo_judges as judges

    class Exploding:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("rate limited")

    monkeypatch.setattr(judges, "_judge_chain", lambda: Exploding())

    result = await judges.coherence({"markdown": "# Doc\n"}, {})
    assert result["score"] is None
    assert "judge failed" in result["comment"]


@pytest.mark.asyncio
async def test_a_missing_artifact_is_not_judged(monkeypatch):
    import fo_judges as judges

    monkeypatch.setattr(
        judges, "_judge_chain", lambda: pytest.fail("must not call a judge with no artifact")
    )
    result = await judges.actionability({"markdown": ""}, {})
    assert result["score"] is None


@pytest.mark.asyncio
async def test_a_judge_verdict_is_normalized_and_keeps_its_reasoning(monkeypatch):
    import fo_judges as judges

    class Chain:
        async def ainvoke(self, *args, **kwargs):
            return {
                "parsed": judges.JudgeVerdict(reasoning="Sections build on each other.", score=4),
                "parsing_error": None,
            }

    monkeypatch.setattr(judges, "_judge_chain", lambda: Chain())

    result = await judges.coherence({"markdown": "# Doc\n"}, {})
    assert result["score"] == pytest.approx(0.75)  # (4-1)/4
    assert result["comment"].startswith("4/5 — ")
    assert "Sections build on each other." in result["comment"]


@pytest.mark.asyncio
async def test_an_out_of_range_judge_score_is_clamped(monkeypatch):
    import fo_judges as judges

    class Chain:
        async def ainvoke(self, *args, **kwargs):
            return {
                "parsed": judges.JudgeVerdict(reasoning="r", score=99),
                "parsing_error": None,
            }

    monkeypatch.setattr(judges, "_judge_chain", lambda: Chain())
    result = await judges.coherence({"markdown": "# Doc\n"}, {})
    assert result["score"] == 1.0


def test_judge_verdicts_are_strict_schema_safe():
    from fo_judges import JudgeVerdict

    schema = JudgeVerdict.model_json_schema()
    assert set(schema["properties"]) == set(schema.get("required", []))
