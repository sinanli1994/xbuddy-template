"""LLM-as-a-judge evaluators for the four inherently subjective dimensions.

Used **only** where determinism cannot reach. Grounding, preservation, honesty,
completeness and ordering are all decidable from data, so they live in
`evaluators.py`; a judge asked about those would add cost and non-reproducibility to
questions that already have exact answers.

What is left is genuinely a matter of reading: does the document hang together, is
it about *this* person, could they start on Monday, and is the ordering defensible.

Design
------
* **Bounded 1-5 with a written rubric per level.** "Rate the quality" produces
  drift; naming what a 2 looks like versus a 4 produces something comparable across
  runs.
* **Reasoning is required and recorded.** The score alone is not reviewable; the
  comment is what lets a human check whether the judge was right, which matters
  most when a judge disagrees with a deterministic check.
* **Normalized to 0-1** for LangSmith aggregation, with the raw 1-5 in the comment.
* **Temperature 0** via the project's `get_model()`.

A judge score is never allowed to substitute for a deterministic one. The seeded arm
is expected to keep coherence high while grounding collapses — that gap is the whole
point, and it only reads as a finding because the two families are separate.
"""

import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

JUDGE_TAG = "internal_judge"


class JudgeVerdict(BaseModel):
    """Bounded score plus the reasoning that produced it.

    No defaults: this goes through `with_structured_output(strict=True)`, which
    requires every property in `required`.
    """

    reasoning: str = Field(
        description=(
            "Two or three sentences naming the specific evidence in the document that "
            "drove the score. Quote or paraphrase it. Do not restate the rubric."
        )
    )
    score: int = Field(description="An integer from 1 to 5, using the rubric exactly.")


RUBRICS: dict[str, str] = {
    "coherence": """Score how well the document reads as one coherent strategy rather than a
list of parts.

5 — Sections build on each other. The positioning explains why these skill
    priorities, and the plan follows from both. Nothing contradicts anything else.
4 — Coherent, with one section that feels bolted on or one mild redundancy.
3 — Individually sensible sections that do not obviously relate; the reader has to
    join them up.
2 — At least one internal contradiction, or the positioning describes a different
    person than the plan assumes.
1 — Incoherent: contradictory claims, or a document that could not be acted on as a
    whole.""",
    "relevance_to_user": """Score how specific the document is to THIS person rather than to anyone with
the same job title.

5 — Could not be reused for another candidate. Their actual background, constraints,
    and stated goal shape every recommendation.
4 — Clearly personalized, with one or two paragraphs of generic career advice.
3 — Half tailored, half boilerplate that would fit any applicant for the role.
2 — Mostly generic; the specifics are decoration on standard advice.
1 — Template text. Swapping in another candidate's name would require no other edit.""",
    "actionability": """Score whether the reader could start on Monday morning without asking a
follow-up question.

5 — Every step names a concrete first move, and where it matters a quantity or
    cadence. No step needs interpretation.
4 — Mostly concrete; one or two steps are directional rather than specific.
3 — Mixed: real actions alongside aspirations like "improve networking".
2 — Mostly aspiration. The reader would have to design the actual work themselves.
1 — Nothing here can be started. Goals restated as instructions.""",
    "prioritization_quality": """Score whether the ordering is defensible given the person's timeline and
constraints — not merely whether it is numbered.

5 — The first steps are the highest-leverage or most time-critical, and the ordering
    respects the stated timeline and any constraint.
4 — Sensible ordering with one item arguably misplaced.
3 — Ordering is plausible but arbitrary; a reshuffle would read the same.
2 — Ordering works against the timeline — slow-burn work ahead of something urgent.
1 — Actively misleading: the first step should clearly be last, or the plan ignores
    a hard deadline.""",
}

SYSTEM_PROMPT = """You are evaluating one final artifact produced by a career-coaching agent for a
specific user. You are scoring exactly one dimension, using the rubric given.

Rules that matter:
- Use the rubric's own wording. If the document matches the description of a 3, it is
  a 3, however impressive or weak it feels overall.
- Score only the named dimension. A well-grounded document with an arbitrary order
  still scores low on prioritization, and a fluent invention still scores high on
  coherence — other checks catch that, and you must not compensate for them.
- Recommendations are expected. Suggesting the user learn something they do not yet
  know is correct behaviour, not a fault.
- Sections marked as unknown or missing are honest, not incomplete.
- Give the reasoning before committing to the score, and name concrete evidence."""


def _judge_chain():
    """Structured-output chain for a judge verdict.

    Indirection so tests patch one function. Tagged so its tokens are suppressed the
    same way every other internal call is — a judge is not conversation.
    """
    from core.llm import get_model

    return (
        get_model()
        .with_structured_output(
            JudgeVerdict,
            method="json_schema",  # type: ignore[arg-type]
            strict=True,
            include_raw=True,
        )
        .with_config(tags=[JUDGE_TAG])
    )


def build_judge_prompt(dimension: str, markdown: str, reference: dict[str, Any]) -> str:
    """The user-side content: rubric, the profile that was collected, the document.

    The profile is included so `relevance_to_user` can be judged at all — without it
    the judge cannot tell tailored from generic. It is labelled as context rather
    than as something to score.
    """
    facts = reference.get("confirmed_facts") or {}
    fact_lines = "\n".join(
        f"- {name}: {value}" for name, value in sorted(facts.items()) if value not in (None, [], "")
    )
    unknowns = "\n".join(f"- {entry}" for entry in reference.get("unknowns") or [])

    return (
        f"DIMENSION: {dimension}\n\n"
        f"RUBRIC\n{RUBRICS[dimension]}\n\n"
        f"WHAT THE AGENT ACTUALLY COLLECTED (context, not scored)\n"
        f"{fact_lines or '(nothing)'}\n\n"
        f"WHAT WAS NEVER COLLECTED (declaring these is honest)\n"
        f"{unknowns or '(nothing missing)'}\n\n"
        f"THE ARTIFACT\n{markdown}"
    )


async def _score_dimension(
    dimension: str, outputs: dict, reference_outputs: dict | None
) -> dict:
    """Run one judge. Never raises; a judge failure scores nothing rather than 0.

    A failed judge returning 0 would be indistinguishable from a genuinely bad
    document, which would quietly corrupt the aggregate. `None` keeps it out.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    markdown = outputs.get("markdown") or ""
    if not markdown:
        return {"key": dimension, "score": None, "comment": "no artifact to judge"}

    prompt = build_judge_prompt(dimension, markdown, reference_outputs or {})
    try:
        result = await _judge_chain().ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
    except Exception as exc:  # noqa: BLE001 - a judge outage must not look like a bad artifact
        return {"key": dimension, "score": None, "comment": f"judge failed: {exc}"}

    if not isinstance(result, dict) or result.get("parsing_error"):
        return {
            "key": dimension,
            "score": None,
            "comment": f"judge output unparseable: {(result or {}).get('parsing_error')}",
        }

    verdict = result.get("parsed")
    if not isinstance(verdict, JudgeVerdict):
        return {"key": dimension, "score": None, "comment": "judge returned no verdict"}

    raw = max(1, min(5, int(verdict.score)))
    return {
        "key": dimension,
        # Normalized so LangSmith averages it alongside the 0/1 deterministic keys.
        "score": (raw - 1) / 4,
        "comment": f"{raw}/5 — {verdict.reasoning}",
    }


async def coherence(outputs: dict, reference_outputs: dict | None = None, **_: Any) -> dict:
    return await _score_dimension("coherence", outputs, reference_outputs)


async def relevance_to_user(outputs: dict, reference_outputs: dict | None = None, **_: Any) -> dict:
    return await _score_dimension("relevance_to_user", outputs, reference_outputs)


async def actionability(outputs: dict, reference_outputs: dict | None = None, **_: Any) -> dict:
    return await _score_dimension("actionability", outputs, reference_outputs)


async def prioritization_quality(
    outputs: dict, reference_outputs: dict | None = None, **_: Any
) -> dict:
    return await _score_dimension("prioritization_quality", outputs, reference_outputs)


JUDGE_EVALUATORS = [coherence, relevance_to_user, actionability, prioritization_quality]
