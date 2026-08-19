"""Pure tests for the PR 5 final-output contract and renderer.

No model, no network, no I/O — `render_final_output` is a pure function and the
models are plain Pydantic, so the whole contract is provable offline.

These tests pin the parts of the seven-dimension quality bar that are mechanically
checkable: prioritized (a model invariant), honest about missing information (an
unconditional heading), complete (nothing dropped), and verbatim preservation of
confirmed Action Plan items. Grounded / relevant / coherent are judged by the
Stage 6 eval, not here.
"""

import pytest
from pydantic import ValidationError

from agents.xbuddy.final_output import render_final_output
from agents.xbuddy.models import ActionItem, FinalOutput, XBuddyData


def item(
    step="Rewrite the CV summary",
    rationale="Leads with platform work",
    priority=1,
    timeframe=None,
) -> ActionItem:
    return ActionItem(step=step, rationale=rationale, priority=priority, timeframe=timeframe)


def output(**overrides) -> FinalOutput:
    """A complete, valid FinalOutput. Every field supplied — there are no defaults."""
    values = {
        "headline": "QA Analyst to Senior SRE within 3 months",
        "positioning_summary": "Four years of QA moving into automation.",
        "strengths_to_leverage": ["systems debugging"],
        "skill_priorities": ["Kubernetes"],
        "search_targets": ["fintech"],
        "action_items": [item()],
        "risks_or_constraints": ["Three-month timeline is tight"],
        "unknowns": ["Salary expectation was never discussed"],
    }
    values.update(overrides)
    return FinalOutput(**values)


# --------------------------------------------------------------------------
# Model construction and the strict-schema convention
# --------------------------------------------------------------------------


def test_valid_construction():
    result = output()
    assert result.action_items[0].step == "Rewrite the CV summary"
    assert result.unknowns == ["Salary expectation was never discussed"]


def test_empty_lists_are_valid():
    """Nothing-to-say-here must be representable without dropping the field."""
    result = output(
        strengths_to_leverage=[],
        skill_priorities=[],
        search_targets=[],
        risks_or_constraints=[],
        unknowns=[],
    )
    assert result.unknowns == []


@pytest.mark.parametrize("model", [ActionItem, FinalOutput])
def test_all_fields_required_for_strict_json_schema(model):
    """OpenAI strict mode requires every property in `required`.

    A Pydantic default silently drops a field from `required` and invalidates the
    schema — the constraint PR 3 verified live and PR 4's extract models follow.
    Stage 2 will pass these models to `with_structured_output(strict=True)`.
    """
    schema = model.model_json_schema()
    properties = set(schema.get("properties", {}))
    required = set(schema.get("required", []))
    assert properties == required, f"{model.__name__}: {sorted(properties - required)} not required"


def test_timeframe_is_nullable_not_optional():
    """Null is a real value here — no timeline to size against — not an omission."""
    assert item(timeframe=None).timeframe is None
    with pytest.raises(ValidationError):
        ActionItem(step="s", rationale="r", priority=1)  # type: ignore[call-arg]


# --------------------------------------------------------------------------
# Priority semantics: exactly 1..n, each once
# --------------------------------------------------------------------------


def test_priorities_must_be_a_total_order():
    result = output(
        action_items=[item(priority=1), item(step="b", priority=2), item(step="c", priority=3)]
    )
    assert [entry.priority for entry in result.action_items] == [1, 2, 3]


@pytest.mark.parametrize(
    ("priorities", "why"),
    [
        ([1, 1], "duplicates have no single correct reading"),
        ([1, 3], "a gap means a rank was dropped"),
        ([0, 1], "priorities are 1-based"),
        ([2, 3], "must start at 1"),
        ([-1, 1], "negative rank is meaningless"),
    ],
)
def test_invalid_priorities_are_rejected(priorities, why):
    """Rejected, never silently renumbered.

    Normalizing would invent an ordering the model did not express. Stage 2 sees
    this as a parsing_error and degrades, matching PR 4's extraction path.
    """
    with pytest.raises(ValidationError, match="priorities must be exactly"):
        output(action_items=[item(step=f"s{p}", priority=p) for p in priorities])


def test_an_empty_plan_is_structurally_allowed():
    """The validator must not reject `[]` — 1..0 is vacuously a total order.

    An empty plan is a synthesis defect, but it is the node's and the eval's to
    catch; a validator error here would turn a bad document into a lost turn.
    """
    assert output(action_items=[]).action_items == []


# --------------------------------------------------------------------------
# Rendering: order, stability, purity
# --------------------------------------------------------------------------


EXPECTED_HEADINGS = [
    "## Where You Stand",
    "## Strengths to Leverage",
    "## Skill Priorities",
    "## Where to Look",
    "## Your Action Plan",
    "## Risks and Constraints",
    "## What I Still Don't Know",
]


def headings(markdown: str) -> list[str]:
    return [line for line in markdown.splitlines() if line.startswith("## ")]


def test_section_order_is_fixed():
    assert headings(render_final_output(output())) == EXPECTED_HEADINGS


def test_headline_is_the_only_h1():
    rendered = render_final_output(output())
    assert rendered.startswith("# QA Analyst to Senior SRE within 3 months\n")
    assert [line for line in rendered.splitlines() if line.startswith("# ")] == [
        "# QA Analyst to Senior SRE within 3 months"
    ]


def test_rendering_is_byte_identical_for_equal_input():
    assert render_final_output(output()) == render_final_output(output())


def test_rendering_does_not_mutate_the_input():
    """The renderer sorts by priority; it must copy rather than reorder in place."""
    source = output(
        action_items=[item(step="second", priority=2), item(step="first", priority=1)]
    )
    before = source.model_dump()

    render_final_output(source)

    assert source.model_dump() == before
    assert [entry.step for entry in source.action_items] == ["second", "first"]


def test_action_items_render_in_priority_order_not_list_order():
    rendered = render_final_output(
        output(action_items=[item(step="second", priority=2), item(step="first", priority=1)])
    )
    assert rendered.index("**first**") < rendered.index("**second**")
    assert "1. **first**" in rendered
    assert "2. **second**" in rendered


def test_step_and_rationale_stay_on_separate_labelled_lines():
    """Recommendations must stay distinguishable from the user's own words."""
    rendered = render_final_output(
        output(action_items=[item(step="Rewrite the CV", rationale="Closes the gap")])
    )
    assert "1. **Rewrite the CV**" in rendered
    assert "   - Why: Closes the gap" in rendered


def test_timeframe_is_omitted_when_null():
    with_frame = render_final_output(output(action_items=[item(timeframe="this week")]))
    without = render_final_output(output(action_items=[item(timeframe=None)]))
    assert "   - Timeframe: this week" in with_frame
    assert "Timeframe" not in without


# --------------------------------------------------------------------------
# Empty lists, and the two sections that must never disappear
# --------------------------------------------------------------------------


def test_empty_optional_sections_are_omitted_entirely():
    """No placeholder headings — the same no-placeholder rule render_known_data uses."""
    rendered = render_final_output(
        output(
            strengths_to_leverage=[],
            skill_priorities=[],
            search_targets=[],
            risks_or_constraints=[],
        )
    )
    assert headings(rendered) == [
        "## Where You Stand",
        "## Your Action Plan",
        "## What I Still Don't Know",
    ]
    assert "Strengths to Leverage" not in rendered
    assert rendered.endswith("\n")
    assert "\n\n\n" not in rendered, "omitted sections must not leave blank-line gaps"


def test_unknowns_are_rendered_explicitly():
    rendered = render_final_output(
        output(unknowns=["Salary was never discussed", "No location given"])
    )
    assert "## What I Still Don't Know" in rendered
    assert "- Salary was never discussed" in rendered
    assert "- No location given" in rendered


def test_unknowns_heading_survives_an_empty_list_with_an_affirmative_statement():
    """Honesty must not be omissible.

    An absent heading would be indistinguishable from a renderer bug, so an empty
    `unknowns` becomes an explicit claim that nothing is outstanding.
    """
    rendered = render_final_output(output(unknowns=[]))
    assert "## What I Still Don't Know" in rendered
    assert "Nothing outstanding" in rendered


def test_empty_plan_says_so_rather_than_vanishing():
    rendered = render_final_output(output(action_items=[]))
    assert "## Your Action Plan" in rendered
    assert "No action items were produced" in rendered


# --------------------------------------------------------------------------
# Section 5 preservation — the spine
# --------------------------------------------------------------------------


CONFIRMED_ACTION_ITEMS = [
    "Rewrite the CV summary to lead with the platform migration",
    "Message three former colleagues at target companies this week",
    "Ship a small Kubernetes project and write it up",
]


def map_confirmed(action_items: list[str]) -> list[ActionItem]:
    """The mapping Stage 2 will perform: list order becomes priority order.

    `XBuddyData.action_items` is `list[str]` whose order already means "what to do
    first" (the Action Plan prompt says so), which is why index+1 is the priority
    and not an arbitrary choice.
    """
    return [
        ActionItem(
            step=step,
            rationale=f"Confirmed in the Action Plan section ({index + 1})",
            priority=index + 1,
            timeframe=None,
        )
        for index, step in enumerate(action_items)
    ]


def test_confirmed_action_items_map_without_loss():
    """list[str] -> list[ActionItem] must preserve the string and the order."""
    data = XBuddyData(action_items=CONFIRMED_ACTION_ITEMS)
    mapped = map_confirmed(data.action_items)

    assert [entry.step for entry in mapped] == CONFIRMED_ACTION_ITEMS
    assert [entry.priority for entry in mapped] == [1, 2, 3]
    # The invariant holds by construction, so a valid FinalOutput accepts them.
    assert output(action_items=mapped).action_items == mapped


def test_confirmed_action_items_survive_rendering_verbatim():
    rendered = render_final_output(output(action_items=map_confirmed(CONFIRMED_ACTION_ITEMS)))
    for step in CONFIRMED_ACTION_ITEMS:
        assert f"**{step}**" in rendered, f"confirmed wording altered or dropped: {step!r}"


def test_renderer_does_not_transform_step_text():
    """No capitalization, punctuation, or Markdown normalization of user wording."""
    awkward = "message 3 ex-colleagues (fintech) — *this* week"
    rendered = render_final_output(output(action_items=[item(step=awkward)]))
    assert f"**{awkward}**" in rendered
