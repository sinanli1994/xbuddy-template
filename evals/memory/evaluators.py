"""Deterministic evaluators for the JobBuddy memory eval.

Pure functions over a dict — no network, no model, no imports from the eval
runner — so `tests/agents/xbuddy/test_memory_evaluators.py` can exercise them
offline while ordinary pytest stays free of API keys.

Each takes the runner's `outputs` and the dataset's `reference_outputs` and
returns the LangSmith feedback shape `{"key", "score", "comment"}`.

The runner's `outputs` carry:
    reply        the model's next turn
    known_block  the deterministic KNOWN SO FAR rendering the model was shown
    before       XBuddyData before this turn's merge, as a plain dict
    after        XBuddyData after this turn's merge, as a plain dict
"""

import re
from typing import Any

# Sections whose designed behaviour is to *propose* content rather than ask the
# user to supply it, which makes `advances_unfilled_field` structurally
# inapplicable. The Action Plan prompt says: "Propose a first draft yourself...
# Do not ask the user to invent the plan from nothing." A reply that delivers a
# five-step plan and asks which steps feel realistic is ideal behaviour, and
# scoring it as a failure to advance would be measuring the wrong thing.
#
# Scoped to this one section on purpose: the four collection sections keep the
# full-strength check.
PROPOSE_SECTIONS = frozenset({"action_plan"})

# Phrases that indicate a reply is *soliciting* a field.
#
# Deliberately phrase-level rather than single nouns. A bare "role" matched
# "When would you like to be in the new role?" — a timeline question — and
# scored it as re-asking the role. Generic nouns that appear incidentally in
# other questions ("role", "where", "plan") are only used inside phrasings that
# actually request the value.
FIELD_TERMS: dict[str, tuple[str, ...]] = {
    "target_roles": (
        "what role",
        "which role",
        "what kind of role",
        "what sort of role",
        "what type of role",
        "what job title",
        "which job title",
        "what title",
        "what position",
        "which position",
    ),
    "career_goal_summary": (
        "why",
        "what's driving",
        "what is driving",
        "motivat",
        "matter to you",
        "what are you hoping",
        # The Career Goal prompt defines this field as "what they want this move
        # to change", so the agent asks for it in those words. Missing this
        # phrasing made a perfectly correct reply score as failing to advance.
        "move to change",
        "this move to change",
    ),
    "target_timeline": ("when", "timeline", "how soon", "by when", "time frame", "timeframe"),
    "current_role": ("current role", "what do you do", "currently do", "most recent role"),
    "years_experience": ("how many years", "years of experience", "how long have you"),
    "highest_education": ("education", "degree", "qualification", "studied"),
    "work_history": ("work history", "previous roles", "past roles", "where have you worked"),
    "preferred_locations": (
        "which city",
        "what city",
        "location",
        "based in",
        "where are you based",
        "where would you like to work",
        "relocat",
    ),
    "preferred_work_modes": ("remote", "hybrid", "on-site", "onsite", "in the office"),
    "target_industries": ("industry", "industries", "sector"),
    "employment_types": ("full-time", "part-time", "contract", "freelance", "employment type"),
    "salary_expectation": ("salary", "pay range", "compensation", "how much", "day rate"),
    "strengths": (
        "strength",
        "good at",
        "best at",
        # The Skill Assessment prompt requires an example alongside each strength,
        # so evidence-seeking is how this field actually gets advanced.
        "share an example",
        "give me an example",
        "example of a project",
    ),
    "current_skills": ("what skills", "which skills", "technologies", "tech stack"),
    "skill_gaps": ("gap", "missing", "lacking", "need to learn", "weakness"),
    "action_items": ("next step", "action item", "what would you like to start", "first move"),
}

# Human-readable labels, mirroring context._FIELD_LABELS so the rendering check
# can look for what the user would actually see.
FIELD_LABELS: dict[str, str] = {
    "target_roles": "Target role(s)",
    "career_goal_summary": "Career goal",
    "target_timeline": "Timeline",
    "current_role": "Current role",
    "years_experience": "Years of experience",
    "highest_education": "Education",
    "work_history": "Work history",
    "preferred_locations": "Preferred locations",
    "preferred_work_modes": "Work modes",
    "target_industries": "Target industries",
    "employment_types": "Employment types",
    "salary_expectation": "Salary expectation",
    "strengths": "Strengths",
    "current_skills": "Current skills",
    "skill_gaps": "Skill gaps",
    "action_items": "Action items",
}


def is_empty(value: Any) -> bool:
    """The same emptiness notion the merge and rendering use."""
    return value is None or value == [] or value == ""


def value_tokens(value: Any) -> list[str]:
    """Lower-cased tokens a reply could echo to acknowledge a value."""
    if isinstance(value, list):
        items = [str(item) for item in value]
    else:
        items = [str(value)]
    return [item.lower() for item in items if item]


# Every phrase in FIELD_TERMS, used only to decide whether a question names any
# field at all. See question_scopes.
ALL_FIELD_TERMS: frozenset[str] = frozenset(
    term for terms in FIELD_TERMS.values() for term in terms
)


def question_scopes(reply: str) -> list[tuple[str, str]]:
    """Each question sentence paired with the sentence immediately before it.

    Returns `(question, preceding)` rather than one merged string, because the two
    sentences are *not* equally good evidence and `asks_about` treats them
    differently. Nothing beyond one sentence back is ever returned — scanning the
    whole reply reintroduces exactly the false positives that phrase-level terms
    were added to remove.
    """
    parts = [part.strip() for part in re.split(r"(?<=[?.!])\s+", reply or "") if part.strip()]
    scopes: list[tuple[str, str]] = []
    for index, part in enumerate(parts):
        if "?" not in part:
            continue
        preceding = parts[index - 1] if index > 0 else ""
        scopes.append((part.lower(), preceding.lower()))
    return scopes


def mentions_a_field(text: str) -> bool:
    """Whether `text` names any tracked field."""
    return any(term in text for term in ALL_FIELD_TERMS)


def asks_about(reply: str, field: str, known_value: Any = None) -> bool:
    """Whether `reply` asks the user to supply `field`.

    Two tiers, in order:

    1. **The question sentence names the field.** The normal case, and the only
       evidence used when the question names *any* field.
    2. **The question names no field at all** — a referential "What is it?" — in
       which case the preceding sentence supplies the topic:

           "Let's capture your highest completed education next. What is it?"

       Question-only scoping scored that as advancing nothing, because "education"
       is in the statement.

    Tier 2 is gated on the question being topically bare on purpose. An unconditional
    two-sentence window flags the acknowledgement that routinely precedes a question:

        "Great, we'll aim for that timeline. What role are you looking to move into next?"

    That is not a timeline re-ask, but "timeline" is in the window. Because the
    question already names `target_roles`, it is not bare, so the lookback never
    happens and only the role question is seen. A live case regressed on exactly this.

    When a known value is given, a scope that *echoes* it is acknowledgement rather
    than a re-ask — "For a Senior SRE role, when do you want to move?" mentions the
    role but asks about timing.
    """
    terms = FIELD_TERMS.get(field, ())
    if not terms:
        return False
    tokens = value_tokens(known_value) if not is_empty(known_value) else []
    for question, preceding in question_scopes(reply):
        if any(term in question for term in terms):
            scope = question
        elif not mentions_a_field(question) and any(term in preceding for term in terms):
            scope = f"{preceding} {question}"
        else:
            continue
        if tokens and any(token in scope for token in tokens):
            continue  # the value is restated -> acknowledgement
        return True
    return False


# --------------------------------------------------------------------------
# 1. known_block_complete
# --------------------------------------------------------------------------


def known_block_complete(outputs: dict, reference_outputs: dict | None = None, **_: Any) -> dict:
    """Every populated field must appear in the KNOWN SO FAR rendering.

    This is the first thing a memory regression breaks: if the merge drops a
    field, the block silently stops mentioning it and the agent has no way to
    know it was ever collected.
    """
    block = (outputs.get("known_block") or "").lower()
    after = outputs.get("after") or {}

    missing_from_block: list[str] = []
    for field, value in after.items():
        if is_empty(value):
            continue
        label = FIELD_LABELS.get(field, field).lower()
        tokens = value_tokens(value)
        if label not in block or not all(token in block for token in tokens):
            missing_from_block.append(field)

    return {
        "key": "known_block_complete",
        "score": int(not missing_from_block),
        "comment": (
            f"absent from KNOWN SO FAR: {missing_from_block}"
            if missing_from_block
            else "all populated fields rendered"
        ),
    }


# --------------------------------------------------------------------------
# 2. extraction_no_clobber
# --------------------------------------------------------------------------


def extraction_no_clobber(outputs: dict, reference_outputs: dict | None = None, **_: Any) -> dict:
    """A populated value must never become empty through a merge.

    Overwrites with a *new* value are fine — that is how corrections work. Only
    populated -> None/[]/"" is a defect.
    """
    before = outputs.get("before") or {}
    after = outputs.get("after") or {}

    clobbered = [
        field
        for field, old in before.items()
        if not is_empty(old) and is_empty(after.get(field))
    ]

    return {
        "key": "extraction_no_clobber",
        "score": int(not clobbered),
        "comment": (
            f"values lost by the merge: {clobbered}" if clobbered else "no stored value lost"
        ),
    }


# --------------------------------------------------------------------------
# 3. no_redundant_question
# --------------------------------------------------------------------------


def no_redundant_question(outputs: dict, reference_outputs: dict | None = None, **_: Any) -> dict:
    """The reply must not ask for anything memory already holds.

    The user-visible symptom of every memory failure in this PR, and the reason
    this eval exists.
    """
    reply = outputs.get("reply") or ""
    # Judged against what was known *before* the turn: that is what the user
    # already told us, whether or not the merge managed to keep it.
    before = outputs.get("before") or {}

    # Fields the section prompt prescribes following up on are exempt: asking
    # "how many days on-site?" after "hybrid" refines a stored value, it does not
    # re-request it. Listed per case, never suppressed globally — a work-mode
    # question in a case without that annotation still counts as a re-ask.
    exempt = set((reference_outputs or {}).get("refinement_pending") or [])

    redundant = [
        field
        for field, value in before.items()
        if field not in exempt and not is_empty(value) and asks_about(reply, field, value)
    ]

    return {
        "key": "no_redundant_question",
        "score": int(not redundant),
        "comment": (
            f"re-asked already-known fields: {redundant}"
            if redundant
            else "asked nothing already known"
        ),
    }


# --------------------------------------------------------------------------
# 4. advances_unfilled_field
# --------------------------------------------------------------------------


def advances_unfilled_field(outputs: dict, reference_outputs: dict | None = None, **_: Any) -> dict:
    """When work remains in the section, the reply should move it forward.

    "Forward" means asking about a genuinely missing field *or* following up on a
    stored value the prompt says to refine. Both are progress; neither is a stall.

    Not applicable to Action Plan (see PROPOSE_SECTIONS) or to a section with
    nothing left, and auto-passes in those cases with an explicit comment.
    """
    reference = reference_outputs or {}
    reply = outputs.get("reply") or ""
    section = reference.get("section")
    missing = list(reference.get("missing_fields") or [])
    refinement = list(reference.get("refinement_pending") or [])

    if section in PROPOSE_SECTIONS:
        return {
            "key": "advances_unfilled_field",
            "score": 1,
            "comment": (
                f"not applicable: {section} proposes its items rather than asking "
                "the user to invent them"
            ),
        }

    candidates = missing + refinement
    if not candidates:
        return {
            "key": "advances_unfilled_field",
            "score": 1,
            "comment": "section complete, nothing to advance",
        }

    targeted = [field for field in candidates if asks_about(reply, field)]

    return {
        "key": "advances_unfilled_field",
        "score": int(bool(targeted)),
        "comment": (
            f"targets {targeted}" if targeted else f"targets none of {candidates}"
        ),
    }


EVALUATORS = [
    known_block_complete,
    extraction_no_clobber,
    no_redundant_question,
    advances_unfilled_field,
]
