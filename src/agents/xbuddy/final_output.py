"""Deterministic rendering of the final job search strategy.

Pure: no LLM, no network, no I/O. Given a `FinalOutput`, `render_final_output`
returns Markdown, and the same input always returns byte-identical output.

Why structured-first
--------------------
Synthesis produces a `FinalOutput`; this module turns it into the document. The
alternative — asking the model for Markdown directly — was rejected because the
grounding checks that matter most only work against fields. "Every proper noun in
the document appears in XBuddyData" is an assertion over `strengths_to_leverage`;
over prose it is a regex. This is the third application of the pattern already
used by `context.render_known_data` and `persistence.render_section_content`.

Markdown, not Tiptap: the editor's `convertToTiptapFormat` parses `#`/`##`/`###`
and `- ` lists, and its own saves write Markdown into both `content` and
`markdown_content`, so Markdown is what the frontend already round-trips.

The quality contract
--------------------
Seven dimensions, and what actually holds each one:

1. **Grounded** — user-specific facts must trace to collected `XBuddyData`. Not
   enforceable here: this module cannot see `XBuddyData`. Held by the Stage 2
   prompt and by the eval's `grounding_no_invention` check.
2. **Relevant** — same: prompt and eval.
3. **Coherent** — partly structural. Fixed section order means the document cannot
   arrive shuffled; readability within a section is a judged dimension.
4. **Actionable** — `ActionItem.step` is required and rendered as an imperative
   list; `rationale` must name the fact it follows from.
5. **Prioritized** — a model invariant. `FinalOutput.priorities_form_a_total_order`
   rejects anything that is not exactly 1..n, and this renderer emits items in
   priority order regardless of list order.
6. **Honest about missing information** — `unknowns` is required, and its heading
   is rendered **unconditionally**. An empty list becomes an explicit statement
   that nothing is missing, so silence is never available as an option.
7. **Complete** — every non-empty field reaches the document; nothing is dropped.

Two further rules this module holds:

* **Recommendations stay distinguishable from user facts.** `step` may be a
  confirmed action item in the user's own words; `rationale` is always synthesis.
  They are rendered on separate lines under distinct labels, never merged into one
  sentence.
* **Confirmed Section 5 action items are preserved verbatim.** `XBuddyData.action_items`
  is `list[str]` and its order is already meaningful — the Action Plan prompt says
  to summarize "in order of what to do first". Stage 2 therefore maps
  `action_items[i]` to `ActionItem(step=<that string, unchanged>, priority=i+1)`,
  which is lossless for the string and turns existing order into explicit
  priority. This renderer never rewrites `step`.

What this module deliberately does not do
-----------------------------------------
No sorting of anything except by `priority`; no deduplication; no capitalization
or punctuation fixes; no truncation. Every one of those would be a silent content
transformation, and the point of a deterministic renderer is that the document
contains exactly what synthesis committed to.
"""

from .models import ActionItem, FinalOutput

# Canonical section order, mirroring FinalOutput's field order so the two cannot
# drift. `always` sections are rendered even when empty — see the contract above:
# the plan is the document's spine and `unknowns` is its honesty, so neither may
# vanish quietly. The rest are omitted when empty, matching render_known_data's
# no-placeholder rule.
_HEADING_POSITIONING = "Where You Stand"
_HEADING_STRENGTHS = "Strengths to Leverage"
_HEADING_SKILLS = "Skill Priorities"
_HEADING_TARGETS = "Where to Look"
_HEADING_PLAN = "Your Action Plan"
_HEADING_RISKS = "Risks and Constraints"
_HEADING_UNKNOWNS = "What I Still Don't Know"

_EMPTY_PLAN = "_No action items were produced. This is a defect, not a finding._"
_NO_UNKNOWNS = "_Nothing outstanding — every section was completed._"


def _bullets(values: list[str]) -> list[str]:
    """Render strings as Markdown bullets, in the order given.

    Order is preserved rather than sorted: for `skill_priorities` the order is the
    ranking, and for the others it is the author's emphasis.
    """
    return [f"- {value}" for value in values]


def _render_action_item(item: ActionItem) -> list[str]:
    """One numbered item, with rationale and timeframe on their own lines.

    The number is `item.priority`, not an enumeration counter, so the rank in the
    document is the rank in the data even if a caller renders a subset.
    """
    lines = [f"{item.priority}. **{item.step}**"]
    lines.append(f"   - Why: {item.rationale}")
    if item.timeframe:
        lines.append(f"   - Timeframe: {item.timeframe}")
    return lines


def render_final_output(final_output: FinalOutput) -> str:
    """Render a `FinalOutput` as Markdown.

    Pure and total: never raises for a valid model, never mutates the input, and
    returns byte-identical output for equal input.

    Section order is fixed (see module docstring). Action items are emitted in
    `priority` order — the model validator guarantees those are exactly 1..n, so
    the ordering is total and no tie-breaking is needed.
    """
    parts: list[str] = [f"# {final_output.headline}", ""]

    parts += [f"## {_HEADING_POSITIONING}", "", final_output.positioning_summary, ""]

    for heading, values in (
        (_HEADING_STRENGTHS, final_output.strengths_to_leverage),
        (_HEADING_SKILLS, final_output.skill_priorities),
        (_HEADING_TARGETS, final_output.search_targets),
    ):
        if values:
            parts += [f"## {heading}", "", *_bullets(values), ""]

    # Always rendered: the plan is the artifact's spine.
    parts += [f"## {_HEADING_PLAN}", ""]
    if final_output.action_items:
        for item in sorted(final_output.action_items, key=lambda entry: entry.priority):
            parts += _render_action_item(item)
        parts.append("")
    else:
        parts += [_EMPTY_PLAN, ""]

    if final_output.risks_or_constraints:
        parts += [
            f"## {_HEADING_RISKS}",
            "",
            *_bullets(final_output.risks_or_constraints),
            "",
        ]

    # Always rendered: honesty about gaps must not be omissible.
    parts += [f"## {_HEADING_UNKNOWNS}", ""]
    if final_output.unknowns:
        parts += _bullets(final_output.unknowns)
    else:
        parts.append(_NO_UNKNOWNS)

    return "\n".join(parts).rstrip() + "\n"
