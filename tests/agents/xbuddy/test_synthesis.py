"""Offline tests for PR 5 Stage 2 grounded synthesis.

Everything here is pure or faked. `derive_unknowns`, `build_synthesis_context`, and
`assemble_final_output` take no I/O at all; the one networked function is exercised
through a fake chain. The autouse `no_live_model` fixture makes an accidental real
model construction fail loudly rather than quietly dialling OpenAI — the same guard
discipline as PR 4's autouse `persistence` fixture, added after unit tests once
reached the live Supabase project.
"""

import pytest
from pydantic import ValidationError

from agents.xbuddy import synthesis
from agents.xbuddy.models import (
    ActionAnnotation,
    ActionItem,
    FinalOutput,
    FinalOutputDraft,
    XBuddyData,
)
from agents.xbuddy.synthesis import (
    INTERNAL_SYNTHESIS_TAG,
    UNKNOWN_LABELS,
    assemble_final_output,
    build_synthesis_context,
    derive_unknowns,
    synthesize_final_output,
)


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    """No test in this file may construct a real model."""

    def explode(*args, **kwargs):
        raise AssertionError("a test tried to build a real LLM chain")

    monkeypatch.setattr("core.llm.get_model", explode)


class FakeChain:
    """Records the messages it was given and returns a configured result."""

    def __init__(self, result=None, raises=None):
        self.result = result
        self.raises = raises
        self.calls: list[list] = []

    async def ainvoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        if self.raises is not None:
            raise self.raises
        return self.result


def annotation(step_number=1, rationale="Closes the biggest gap", timeframe=None):
    return ActionAnnotation(
        step_number=step_number, rationale=rationale, timeframe=timeframe
    )


def draft(**overrides) -> FinalOutputDraft:
    values = {
        "headline": "QA Analyst to Senior SRE within 3 months",
        "positioning_summary": "Four years of QA moving into automation.",
        "strengths_to_leverage": ["systems debugging"],
        "skill_priorities": ["Kubernetes"],
        "search_targets": ["fintech"],
        "action_annotations": [annotation()],
        "risks_or_constraints": ["Three-month timeline is tight"],
    }
    values.update(overrides)
    return FinalOutputDraft(**values)


def data(**overrides) -> XBuddyData:
    values = {
        "target_roles": ["Senior SRE"],
        "target_timeline": "within 3 months",
        "action_items": ["Ship a Kubernetes side project"],
    }
    values.update(overrides)
    return XBuddyData(**values)


# --------------------------------------------------------------------------
# A. Blank core strings are rejected
# --------------------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\t", "\n  \n"])
@pytest.mark.parametrize("field", ["step", "rationale"])
def test_blank_action_item_strings_are_rejected(field, blank):
    values = {"step": "Do the thing", "rationale": "Because", "priority": 1, "timeframe": None}
    values[field] = blank
    with pytest.raises(ValidationError, match="must not be blank"):
        ActionItem(**values)


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
@pytest.mark.parametrize("field", ["headline", "positioning_summary"])
def test_blank_final_output_strings_are_rejected(field, blank):
    values = {
        "headline": "A move",
        "positioning_summary": "A summary",
        "strengths_to_leverage": [],
        "skill_priorities": [],
        "search_targets": [],
        "action_items": [],
        "risks_or_constraints": [],
        "unknowns": [],
    }
    values[field] = blank
    with pytest.raises(ValidationError, match="must not be blank"):
        FinalOutput(**values)


def test_meaningful_whitespace_inside_a_value_is_preserved():
    """Blankness is rejected; content is never stripped or reformatted."""
    assert ActionItem(
        step="  Rewrite the CV  ", rationale="r", priority=1, timeframe=None
    ).step == "  Rewrite the CV  "


def test_blank_rejection_does_not_judge_quality():
    """A weak-but-present string is accepted. Subjective quality is the eval's job."""
    assert ActionItem(step="x", rationale="y", priority=1, timeframe=None).step == "x"


# --------------------------------------------------------------------------
# B. Deterministic unknowns
# --------------------------------------------------------------------------


def test_unknown_labels_cover_every_xbuddydata_field():
    """Parity guard: a new XBuddyData field must get a label or this fails.

    Without it the map silently under-reports, and a field nobody collected would
    vanish from UNKNOWNS instead of being declared missing.
    """
    assert set(UNKNOWN_LABELS) == set(XBuddyData.model_fields)


def test_everything_is_unknown_for_empty_data():
    unknowns = derive_unknowns(XBuddyData())
    assert len(unknowns) == len(XBuddyData.model_fields)
    assert unknowns == list(UNKNOWN_LABELS.values())


def test_nothing_is_unknown_when_every_field_is_populated():
    full = XBuddyData(
        target_roles=["Senior SRE"],
        career_goal_summary="More autonomy",
        target_timeline="3 months",
        current_role="QA Analyst",
        years_experience=4,
        highest_education="BSc",
        work_history=["QA at Acme"],
        preferred_locations=["Berlin"],
        preferred_work_modes=["hybrid"],
        target_industries=["fintech"],
        employment_types=["full-time"],
        salary_expectation="80k",
        strengths=["debugging"],
        current_skills=["Python"],
        skill_gaps=["Kubernetes"],
        action_items=["Ship a project"],
    )
    assert derive_unknowns(full) == []


def test_populated_fields_never_appear_in_unknowns():
    result = derive_unknowns(data(current_role="QA Analyst", years_experience=4))
    for populated in ("target_roles", "target_timeline", "current_role", "years_experience", "action_items"):
        assert UNKNOWN_LABELS[populated] not in result


def test_empty_list_counts_as_unknown_preserving_pr4_semantics():
    """`[]` means unknown, not "explicitly none".

    `merge_extraction` treats an empty list as a no-op, so nothing can distinguish
    "no skill gaps" from "never asked". PR 5 must not invent that distinction.
    """
    assert UNKNOWN_LABELS["skill_gaps"] in derive_unknowns(data(skill_gaps=[]))
    assert UNKNOWN_LABELS["skill_gaps"] not in derive_unknowns(data(skill_gaps=["Kubernetes"]))


@pytest.mark.parametrize("empty", [None, "", []])
def test_all_emptiness_forms_count_as_unknown(empty):
    values = {"target_roles": ["SRE"], "action_items": ["step"]}
    values["work_history" if empty == [] else "career_goal_summary"] = empty
    field = "work_history" if empty == [] else "career_goal_summary"
    assert UNKNOWN_LABELS[field] in derive_unknowns(XBuddyData(**values))


def test_unknowns_order_follows_field_declaration_order():
    """Deterministic order, matching render_known_data and the conversation's flow."""
    assert derive_unknowns(XBuddyData()) == derive_unknowns(XBuddyData())
    unknowns = derive_unknowns(XBuddyData())
    assert unknowns.index(UNKNOWN_LABELS["target_roles"]) < unknowns.index(
        UNKNOWN_LABELS["action_items"]
    )


# --------------------------------------------------------------------------
# C. Grounded synthesis context
# --------------------------------------------------------------------------


def test_context_separates_the_three_blocks():
    context = build_synthesis_context(data())
    assert "FACTS" in context
    assert "CONFIRMED ACTION PLAN" in context
    assert "UNKNOWNS" in context
    assert context.index("FACTS") < context.index("CONFIRMED ACTION PLAN") < context.index("UNKNOWNS")


def test_facts_block_holds_collected_values():
    context = build_synthesis_context(data(current_role="QA Analyst"))
    facts = context.split("CONFIRMED ACTION PLAN")[0]
    assert "Target role(s): Senior SRE" in facts
    assert "Current role: QA Analyst" in facts


def test_facts_block_excludes_the_action_plan():
    """The plan has its own block with different permissions; showing it twice
    under two rule sets is what invites the model to edit it."""
    context = build_synthesis_context(data(action_items=["Ship a Kubernetes side project"]))
    facts = context.split("CONFIRMED ACTION PLAN")[0]
    assert "Ship a Kubernetes side project" not in facts
    assert "Action items" not in facts


def test_plan_block_is_numbered_in_source_order():
    context = build_synthesis_context(data(action_items=["first step", "second step", "third step"]))
    plan = context.split("CONFIRMED ACTION PLAN")[1].split("UNKNOWNS")[0]
    assert "1. first step" in plan
    assert "2. second step" in plan
    assert "3. third step" in plan
    assert plan.index("1. first step") < plan.index("2. second step") < plan.index("3. third step")


def test_unknowns_block_holds_derived_lines_not_model_authored_ones():
    context = build_synthesis_context(data(salary_expectation=None))
    unknowns = context.split("UNKNOWNS")[1]
    assert UNKNOWN_LABELS["salary_expectation"] in unknowns


def test_every_block_is_present_even_when_empty():
    """An absent heading is indistinguishable from a prompt-assembly bug."""
    context = build_synthesis_context(XBuddyData())
    assert "(nothing was collected)" in context
    assert "(no steps were agreed)" in context
    for heading in ("FACTS", "CONFIRMED ACTION PLAN", "UNKNOWNS"):
        assert heading in context


def test_context_is_deterministic():
    assert build_synthesis_context(data()) == build_synthesis_context(data())


# --------------------------------------------------------------------------
# D. The draft schema and the chain
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", [ActionAnnotation, FinalOutputDraft])
def test_draft_models_are_strict_schema_safe(model):
    schema = model.model_json_schema()
    assert set(schema.get("properties", {})) == set(schema.get("required", []))


def test_draft_schema_excludes_the_fields_the_model_must_not_author():
    """`unknowns` is derived and `action_items` is assembled — neither is offerable."""
    fields = set(FinalOutputDraft.model_fields)
    assert "unknowns" not in fields
    assert "action_items" not in fields
    assert "action_annotations" in fields


def test_annotation_step_numbers_must_cover_each_item_once():
    with pytest.raises(ValidationError, match="step_numbers must be exactly"):
        draft(action_annotations=[annotation(step_number=1), annotation(step_number=1)])
    with pytest.raises(ValidationError, match="step_numbers must be exactly"):
        draft(action_annotations=[annotation(step_number=1), annotation(step_number=3)])


@pytest.mark.asyncio
async def test_synthesis_call_is_tagged_and_not_streamed(monkeypatch):
    """The tag is what keeps the raw JSON out of the chat."""
    captured = {}

    class Recorder:
        def with_structured_output(self, *args, **kwargs):
            captured["structured"] = kwargs
            return self

        def with_config(self, **kwargs):
            captured["tags"] = kwargs.get("tags")
            return FakeChain(result={"parsed": draft(), "parsing_error": None})

    monkeypatch.setattr("core.llm.get_model", lambda *a, **k: Recorder())

    result, error = await synthesize_final_output(data())

    assert error is None
    assert isinstance(result, FinalOutput), "the tagged path must still produce an artifact"
    assert captured["tags"] == [INTERNAL_SYNTHESIS_TAG]
    assert captured["structured"]["strict"] is True
    assert captured["structured"]["include_raw"] is True


@pytest.mark.asyncio
async def test_the_model_sees_the_three_block_context(monkeypatch):
    chain = FakeChain(result={"parsed": draft(), "parsing_error": None})
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: chain)

    await synthesize_final_output(data())

    sent = chain.calls[0][0].content
    for heading in ("FACTS", "CONFIRMED ACTION PLAN", "UNKNOWNS"):
        assert heading in sent


# --------------------------------------------------------------------------
# E. Action-item preservation
# --------------------------------------------------------------------------


CONFIRMED = [
    "Rewrite the CV summary to lead with the platform migration",
    "Message three former colleagues at target companies this week",
    "Ship a small Kubernetes project and write it up",
]


def three_annotations(**kwargs):
    return [annotation(step_number=n, rationale=f"reason {n}", **kwargs) for n in (1, 2, 3)]


def test_confirmed_steps_are_preserved_exactly_and_in_order():
    result, error = assemble_final_output(
        draft(action_annotations=three_annotations()), data(action_items=CONFIRMED)
    )

    assert error is None
    assert result is not None
    assert [entry.step for entry in result.action_items] == CONFIRMED
    assert [entry.priority for entry in result.action_items] == [1, 2, 3]


def test_step_text_comes_from_state_not_from_the_model():
    """The model cannot reword a step because it never supplies one."""
    assert "step" not in ActionAnnotation.model_fields


def test_annotations_out_of_order_still_attach_to_the_right_step():
    """`step_number` is an alignment key; priority comes from source order."""
    shuffled = [
        annotation(step_number=3, rationale="third"),
        annotation(step_number=1, rationale="first"),
        annotation(step_number=2, rationale="second"),
    ]
    result, error = assemble_final_output(
        draft(action_annotations=shuffled), data(action_items=CONFIRMED)
    )

    assert error is None
    assert result is not None
    assert [entry.step for entry in result.action_items] == CONFIRMED
    assert [entry.rationale for entry in result.action_items] == ["first", "second", "third"]


def test_model_supplied_rationale_and_timeframe_are_kept():
    result, _ = assemble_final_output(
        draft(action_annotations=[annotation(rationale="Closes the gap", timeframe="week 1")]),
        data(action_items=["Ship a Kubernetes side project"]),
    )
    assert result is not None
    assert result.action_items[0].rationale == "Closes the gap"
    assert result.action_items[0].timeframe == "week 1"


def test_unknowns_on_the_assembled_output_are_derived_not_drafted():
    result, _ = assemble_final_output(draft(), data(salary_expectation=None))
    assert result is not None
    assert UNKNOWN_LABELS["salary_expectation"] in result.unknowns


# --------------------------------------------------------------------------
# F. Failure semantics: degrade, never partial, never raise
# --------------------------------------------------------------------------


def test_annotation_count_mismatch_is_a_hard_failure():
    """Padding or truncating would invent the thing assembly exists to protect."""
    result, error = assemble_final_output(
        draft(action_annotations=[annotation()]), data(action_items=CONFIRMED)
    )
    assert result is None
    assert error is not None
    assert "1 steps but 3 were confirmed" in error


def test_blank_rationale_is_rejected_at_the_model_boundary():
    """Caught on the draft, not during assembly, so the failure is attributable.

    A blank rationale reaching `ActionItem` would be reported as an assembly problem
    when it is really a malformed model response. Rejecting it on `ActionAnnotation`
    surfaces it as `parsing_error` instead, which degrades like any other bad output.
    """
    with pytest.raises(ValidationError, match="must not be blank"):
        annotation(rationale="   ")


def test_assembly_never_raises_even_on_a_draft_that_bypassed_validation():
    """`assemble_final_output` promises never to raise. That must hold structurally.

    `ActionAnnotation` already rejects a blank rationale, so this uses
    `model_construct` to skip validators and simulate the only ways it could still
    happen — a schema change that loses a validator, or a hand-built draft. Without
    the try around the `ActionItem` loop this escapes as a ValidationError, which is
    exactly how the bug was found.
    """
    forged = ActionAnnotation.model_construct(step_number=1, rationale="   ", timeframe=None)
    bad = FinalOutputDraft.model_construct(
        headline="h",
        positioning_summary="p",
        strengths_to_leverage=[],
        skill_priorities=[],
        search_targets=[],
        action_annotations=[forged],
        risks_or_constraints=[],
    )

    result, error = assemble_final_output(bad, data(action_items=["Ship it"]))

    assert result is None, "no partial artifact may be returned"
    assert error is not None
    assert "rejected by validation" in error


def test_assembly_reports_a_missing_annotation_number():
    """Right count, wrong numbering: caught by the per-step lookup, not the count."""
    forged = [
        ActionAnnotation.model_construct(step_number=1, rationale="a", timeframe=None),
        ActionAnnotation.model_construct(step_number=3, rationale="b", timeframe=None),
    ]
    bad = FinalOutputDraft.model_construct(
        headline="h",
        positioning_summary="p",
        strengths_to_leverage=[],
        skill_priorities=[],
        search_targets=[],
        action_annotations=forged,
        risks_or_constraints=[],
    )

    result, error = assemble_final_output(bad, data(action_items=["one", "two"]))

    assert result is None
    assert error == "synthesis produced no annotation for step 2"


@pytest.mark.asyncio
async def test_parsing_error_yields_no_artifact(monkeypatch):
    chain = FakeChain(result={"parsed": None, "parsing_error": ValueError("bad json")})
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: chain)

    result, error = await synthesize_final_output(data())

    assert result is None
    assert error is not None
    assert "could not be parsed" in error


@pytest.mark.asyncio
async def test_model_exception_is_caught_not_raised(monkeypatch):
    chain = FakeChain(raises=RuntimeError("rate limited"))
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: chain)

    result, error = await synthesize_final_output(data())

    assert result is None
    assert error is not None
    assert "rate limited" in error


@pytest.mark.asyncio
async def test_missing_parsed_draft_yields_no_artifact(monkeypatch):
    chain = FakeChain(result={"parsed": None, "parsing_error": None})
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: chain)

    result, error = await synthesize_final_output(data())

    assert result is None
    assert error == "synthesis returned no parsed draft"


@pytest.mark.asyncio
async def test_unexpected_shape_yields_no_artifact(monkeypatch):
    chain = FakeChain(result="not a dict")
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: chain)

    result, error = await synthesize_final_output(data())

    assert result is None
    assert error == "synthesis returned an unexpected shape"


@pytest.mark.asyncio
async def test_a_successful_run_returns_a_complete_artifact(monkeypatch):
    chain = FakeChain(
        result={"parsed": draft(action_annotations=three_annotations()), "parsing_error": None}
    )
    monkeypatch.setattr(synthesis, "_synthesis_chain", lambda: chain)

    result, error = await synthesize_final_output(data(action_items=CONFIRMED))

    assert error is None
    assert isinstance(result, FinalOutput)
    assert [entry.step for entry in result.action_items] == CONFIRMED
    assert result.unknowns, "an incomplete XBuddyData must report unknowns"
