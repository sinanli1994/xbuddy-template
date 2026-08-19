"""Grounded synthesis of the final job search strategy.

Turns `XBuddyData` into a `FinalOutput`. One model call, tagged so its tokens never
reach the chat, and everything the model is not trusted with is computed here.

What the model decides, and what it does not
--------------------------------------------
The model returns a `FinalOutputDraft`. Two fields of the real artifact are absent
from that schema on purpose:

* **`action_items`** — assembled by `assemble_final_output` from the confirmed
  Action Plan plus the model's annotations. The step text is never part of the
  model's output, so "preserved exactly" is a property of the type rather than a
  hope about copying. Priority comes from the source order.
* **`unknowns`** — derived by `derive_unknowns` from `XBuddyData`. A model asked
  which facts are missing can both overlook one and invent one; state already knows
  the answer exactly, so asking is strictly worse.

The grounding boundary
----------------------
`build_synthesis_context` renders three labelled blocks — FACTS, CONFIRMED ACTION
PLAN, UNKNOWNS — and `SYNTHESIS_RULES` gives each different permissions: facts may
be restated but never added to, the plan may be annotated but not edited, unknowns
may not be filled in. Recommendations are allowed and expected; asserting them as
things the user said is the failure this separation exists to prevent.

Failure semantics
-----------------
`synthesize_final_output` never raises and never returns a partial artifact. On a
model error, a parsing error, or an assembly disagreement it returns
`(None, reason)` and the caller (Stage 3) folds `reason` into the existing
`last_error` / `error_count` contract. There is no retry: PR 5 measures the failure
rate before deciding whether one is worth paying for.
"""

import logging
from typing import Any

from langchain_core.messages import SystemMessage

from .context import _FIELD_LABELS
from .models import (
    ActionItem,
    FinalOutput,
    FinalOutputDraft,
    XBuddyData,
)
from .sections.base_prompt import SYNTHESIS_RULES

logger = logging.getLogger(__name__)

# Suppressed at the SSE layer alongside the PR 3/PR 4 internal tags, so the raw
# JSON never appears in the chat. The user is told the document is ready; they read
# it in the editor.
INTERNAL_SYNTHESIS_TAG = "internal_synthesis"

# Field -> how a missing value is described in the artifact.
#
# A dedicated map rather than reusing `_FIELD_LABELS` verbatim, because "Salary
# expectation" is a column header while "Salary expectation was never discussed" is
# a sentence the user reads. Drift is prevented by
# `test_unknown_labels_cover_every_xbuddydata_field`, which asserts these keys equal
# `XBuddyData.model_fields` exactly — the same parity technique PR 4 used to tie the
# extract models to `required_fields`.
UNKNOWN_LABELS: dict[str, str] = {
    "target_roles": "Target role was never pinned down",
    "career_goal_summary": "What this move should change was never stated",
    "target_timeline": "No timeline was given, so nothing here is dated",
    "current_role": "Current role was never given",
    "years_experience": "Years of experience were never given",
    "highest_education": "Education was never discussed",
    "work_history": "Work history was never collected",
    "preferred_locations": "Preferred locations were never discussed",
    "preferred_work_modes": "Remote / hybrid / on-site preference was never discussed",
    "target_industries": "Target industries were never discussed",
    "employment_types": "Employment type (full-time, contract) was never discussed",
    "salary_expectation": "Salary expectation was never discussed",
    "strengths": "Strengths were never collected",
    "current_skills": "Current skills were never collected",
    "skill_gaps": "Skill gaps were never identified",
    "action_items": "No action plan was agreed",
}


def _is_missing(value: Any) -> bool:
    """The same emptiness notion the merge and the renderers already use.

    `[]` counts as missing, which carries PR 4's limitation forward deliberately.
    `merge_extraction` treats an empty list as a no-op, so nothing in the system can
    distinguish "the user has no skill gaps" from "we never asked" — inventing an
    explicit-none semantic here would let the artifact claim a certainty the data
    does not support. Widening this is a schema change, not a PR 5 change.
    """
    return value is None or value == [] or value == ""


def derive_unknowns(user_data: XBuddyData) -> list[str]:
    """Which fields were never collected, as reader-facing lines.

    Deterministic in both membership and order: iteration follows
    `XBuddyData.model_fields`, the same declaration order `render_known_data` uses,
    so the artifact's UNKNOWNS block matches the section order of the conversation.

    A populated field never appears. An absent one always does.
    """
    return [
        UNKNOWN_LABELS[field_name]
        for field_name in user_data.__class__.model_fields
        if _is_missing(getattr(user_data, field_name, None))
    ]


def _render_facts(user_data: XBuddyData) -> str:
    """Populated fields as labelled lines, excluding the action plan.

    Close to `context.render_known_data` but deliberately not a call to it: the plan
    gets its own block with different permissions, so including it here would show
    the model the same steps twice under two sets of rules. Labels come from the
    shared `_FIELD_LABELS` so the artifact and the conversation name things alike.
    """
    lines: list[str] = []
    for field_name in user_data.__class__.model_fields:
        if field_name == "action_items":
            continue
        value = getattr(user_data, field_name, None)
        if _is_missing(value):
            continue
        label = _FIELD_LABELS.get(field_name, field_name.replace("_", " ").capitalize())
        if isinstance(value, list):
            lines.append(f"- {label}: {', '.join(str(entry) for entry in value)}")
        else:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def build_synthesis_context(user_data: XBuddyData) -> str:
    """The three-block grounding context. Pure and deterministic.

    Every block is always present, including when empty. An absent FACTS heading
    would be indistinguishable from a prompt-assembly bug, and the model needs to
    see that a block is empty rather than infer it from silence.
    """
    facts = _render_facts(user_data) or "(nothing was collected)"

    plan_lines = [
        f"{index}. {step}" for index, step in enumerate(user_data.action_items, start=1)
    ]
    plan = "\n".join(plan_lines) or "(no steps were agreed)"

    unknown_lines = [f"- {line}" for line in derive_unknowns(user_data)]
    unknowns = "\n".join(unknown_lines) or "(nothing is missing)"

    return (
        f"FACTS\n{facts}\n\n"
        f"CONFIRMED ACTION PLAN\n{plan}\n\n"
        f"UNKNOWNS\n{unknowns}"
    )


def _synthesis_chain():
    """Structured-output chain for `FinalOutputDraft`.

    Indirection exists so tests patch one function instead of the model registry.
    `.with_config(tags=...)` propagates to the inner LLM run, which is what
    suppresses these tokens at the SSE layer.

    The `method="json_schema"` ignore is the same pre-existing constraint documented
    on `memory_updater._extraction_chain`: `get_model()`'s return type is a union
    including providers that support only function_calling or json_mode, while the
    configured default is an OpenAI model. PR 5 does not widen that.
    """
    from core.llm import get_model

    return (
        get_model()
        .with_structured_output(
            FinalOutputDraft,
            method="json_schema",  # type: ignore[arg-type]
            strict=True,
            include_raw=True,
        )
        .with_config(tags=[INTERNAL_SYNTHESIS_TAG])
    )


def assemble_final_output(
    draft: FinalOutputDraft, user_data: XBuddyData
) -> tuple[FinalOutput | None, str | None]:
    """Combine the model's draft with the fields it is not trusted to author.

    Returns `(final_output, None)` or `(None, reason)`. Never raises.

    Step text and priority come from `user_data.action_items`, never from the model:
    `action_items[i]` becomes `ActionItem(step=<that string, unchanged>,
    priority=i+1)`. The list order is already the agreed order — the Action Plan
    prompt asks for it "in order of what to do first" — so this promotes an implicit
    order into an explicit one without reinterpreting it.

    An annotation count that disagrees with the source is a hard failure rather than
    something to pad or truncate: it means the model did not see the plan it was
    asked to annotate, and guessing which step lost its rationale would be inventing
    the very thing this function exists to protect.
    """
    confirmed = list(user_data.action_items)
    by_number = {annotation.step_number: annotation for annotation in draft.action_annotations}

    if len(draft.action_annotations) != len(confirmed):
        return None, (
            f"synthesis annotated {len(draft.action_annotations)} steps but "
            f"{len(confirmed)} were confirmed"
        )

    items: list[ActionItem] = []
    try:
        for index, step in enumerate(confirmed):
            annotation = by_number.get(index + 1)
            if annotation is None:
                return None, f"synthesis produced no annotation for step {index + 1}"
            items.append(
                ActionItem(
                    step=step,  # verbatim, by construction
                    rationale=annotation.rationale,
                    priority=index + 1,  # source order, not the model's choice
                    timeframe=annotation.timeframe,
                )
            )
    except ValueError as exc:
        # `ActionAnnotation` already rejects a blank rationale, so reaching this
        # means the draft bypassed validation (model_construct, or a schema change
        # that loses a validator). It stays because the "never raises" contract must
        # hold structurally, not because the first line of defence is expected to
        # fail. A confirmed step is never the cause: it comes from state, and
        # XBuddyData cannot hold a blank one that the Action Plan section confirmed.
        return None, f"synthesis output rejected by validation: {exc}"

    try:
        final_output = FinalOutput(
            headline=draft.headline,
            positioning_summary=draft.positioning_summary,
            strengths_to_leverage=draft.strengths_to_leverage,
            skill_priorities=draft.skill_priorities,
            search_targets=draft.search_targets,
            action_items=items,
            risks_or_constraints=draft.risks_or_constraints,
            unknowns=derive_unknowns(user_data),
        )
    except ValueError as exc:  # blank core strings, priority invariant
        return None, f"synthesis output rejected by validation: {exc}"

    return final_output, None


async def synthesize_final_output(
    user_data: XBuddyData,
) -> tuple[FinalOutput | None, str | None]:
    """Run synthesis. Returns `(final_output, None)` or `(None, reason)`.

    Never raises and never returns a partial artifact: a caller that gets `None` has
    nothing to persist and nothing to show, which is the intended outcome. No retry
    — see the module docstring.
    """
    context = build_synthesis_context(user_data)
    messages = [
        SystemMessage(content=f"{SYNTHESIS_RULES.strip()}\n\n{context}"),
    ]

    try:
        result = await _synthesis_chain().ainvoke(messages)
    except Exception as exc:  # network, auth, provider errors
        logger.exception("synthesis: model call failed")
        return None, f"synthesis call failed: {exc}"

    if not isinstance(result, dict):
        return None, "synthesis returned an unexpected shape"

    if result.get("parsing_error"):
        logger.warning("synthesis: parsing error %s", result["parsing_error"])
        return None, f"synthesis output could not be parsed: {result['parsing_error']}"

    draft = result.get("parsed")
    if not isinstance(draft, FinalOutputDraft):
        return None, "synthesis returned no parsed draft"

    return assemble_final_output(draft, user_data)
