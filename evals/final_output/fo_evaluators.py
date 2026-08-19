"""Deterministic evaluators for the JobBuddy final artifact.

Pure functions over dicts — no network, no model, no import from the runner — so
`tests/agents/xbuddy/test_final_output_eval.py` exercises them offline while
ordinary pytest stays free of API keys.

They score the **structured `FinalOutput`** wherever possible rather than regexing
the rendered Markdown. That is the reason PR 5 chose structured-first synthesis:
"every strength traces to a collected fact" is an assertion over a list of strings;
over prose it is a guess.

The runner's `outputs` carry:
    structured   FinalOutput.model_dump()
    markdown     the rendered document
    metrics      lengths and parse status, for the token-budget question

Each takes `(outputs, reference_outputs)` and returns the LangSmith feedback shape
`{"key", "score", "comment"}`.

The grounding boundary
----------------------
The hardest judgement here is which artifact fields *assert facts about the user*
and which *offer recommendations*. Getting it wrong in one direction lets
fabrication through; in the other it flags "learn Kubernetes" as an invented fact.
So the split is explicit and narrow:

* **Fact-asserting** — `headline`, `positioning_summary`, `strengths_to_leverage`,
  `risks_or_constraints`, and each `action_items[].rationale`. These describe the
  person, so every user-specific claim in them must trace to `XBuddyData`.
* **Recommendation** — `skill_priorities`, `search_targets`, and each
  `action_items[].step`/`timeframe`. These propose what to *do*, and naming a
  technology the user does not yet know is the point rather than a defect.

`grounding_no_invention` therefore never inspects the recommendation fields for
entity support. What it does check there is nothing at all — deliberately.
"""

import re
from typing import Any

# Fields whose content describes the user, and so must be supported.
FACT_ASSERTING_FIELDS = (
    "headline",
    "positioning_summary",
    "strengths_to_leverage",
    "risks_or_constraints",
)

# Money-shaped claims: "75k", "£70,000", "800 EUR/day", "$120k".
CURRENCY_PATTERN = re.compile(
    r"(?:[$£€]\s?\d[\d,.]*\s?k?)|(?:\b\d[\d,.]*\s?k\b)|(?:\b\d[\d,.]*\s?(?:EUR|USD|GBP)\b)",
    re.IGNORECASE,
)

# "8 years", "five years of experience".
YEARS_PATTERN = re.compile(
    r"\b(\d{1,2})\s*\+?\s*years?\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|twenty)\s+years?\b",
    re.IGNORECASE,
)

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "twenty": 20,
}

# Tokens too generic to prove that a strength traces to a collected fact.
_STOPWORD_TEXT = (
    "a an and the of to in for with on at by from your you is are be as or "
    "strong strength strengths skill skills experience experienced work working "
    "ability able good great deep solid proven track record using use used "
    "across into over through this that these those their them they it its "
    "more most very well highly successfully years year role roles"
)
STOPWORDS = frozenset(_STOPWORD_TEXT.split(" "))

MIN_TOKEN_LENGTH = 4


def _tokens(text: str) -> set[str]:
    """Content words, lowercased, long enough to carry meaning."""
    words = re.findall(r"[a-z0-9+#.]+", (text or "").lower())
    return {
        word.strip(".")
        for word in words
        if len(word) >= MIN_TOKEN_LENGTH and word not in STOPWORDS
    }


def _fact_text(structured: dict[str, Any]) -> str:
    """Every fact-asserting field, concatenated."""
    parts: list[str] = []
    for field in FACT_ASSERTING_FIELDS:
        value = structured.get(field)
        if isinstance(value, list):
            parts.extend(str(entry) for entry in value)
        elif value:
            parts.append(str(value))
    for item in structured.get("action_items") or []:
        if isinstance(item, dict) and item.get("rationale"):
            parts.append(str(item["rationale"]))
    return "\n".join(parts)


def _profile_tokens(facts: dict[str, Any]) -> set[str]:
    """Every content token anywhere in the confirmed profile."""
    tokens: set[str] = set()
    for value in facts.values():
        if isinstance(value, list):
            for entry in value:
                tokens |= _tokens(str(entry))
        elif value is not None:
            tokens |= _tokens(str(value))
    return tokens


def _profile_text(facts: dict[str, Any]) -> str:
    """The whole profile as one lowercased string, for short-token lookups."""
    parts: list[str] = []
    for value in facts.values():
        if isinstance(value, list):
            parts.extend(str(entry) for entry in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).lower()


def _strength_is_supported(strength: str, profile_tokens: set[str], profile_text: str) -> bool:
    """Whether a claimed strength traces to something in the profile.

    Content-token overlap first, which lets the model rephrase. Then a word-boundary
    fallback for tokens shorter than `MIN_TOKEN_LENGTH`, because several real skills
    are three characters or fewer — SQL, AWS, dbt, Go, R. Without the fallback those
    can never match anything and every one of them reads as fabricated: the first
    seeded run flagged `'SQL'` and `'dbt'` on a profile whose `current_skills` listed
    both. That was a false positive in this evaluator, not a finding about the agent.
    """
    if _tokens(strength) & profile_tokens:
        return True
    short = [
        word
        for word in re.findall(r"[a-z0-9+#.]+", strength.lower())
        if len(word) < MIN_TOKEN_LENGTH and word not in STOPWORDS
    ]
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", profile_text)
        for word in short
    )


def _no_artifact(outputs: dict) -> bool:
    """Whether synthesis produced nothing at all.

    Every evaluator has to check this explicitly. Without it three of them pass
    vacuously on a failed run: empty prose contains no unsupported claim, zero action
    items are trivially a contiguous 1..0 order, and "" re-renders to "". The first
    live run scored 1.00 on all three for a case whose synthesis had 429'd, which made
    a total failure look like partial success.

    Scored 0 rather than None: the question this eval asks is whether the artifact is
    good, and there being no artifact is the strongest possible no.
    """
    return not outputs.get("structured")


# --------------------------------------------------------------------------
# 1. grounding_no_invention
# --------------------------------------------------------------------------


def grounding_no_invention(outputs: dict, reference_outputs: dict | None = None, **_: Any) -> dict:
    """No user-specific claim in a fact-asserting field may be unsupported.

    Three concrete sub-checks rather than one fuzzy similarity score, because each
    corresponds to a way the artifact can lie about someone:

    1. **Money.** A salary or day rate stated when `salary_expectation` was never
       collected is fabrication, full stop. When it *was* collected, the figure must
       actually come from it.
    2. **Seniority.** A years-of-experience number that disagrees with
       `years_experience`, or appears when it was never collected.
    3. **Claimed capability.** Every `strengths_to_leverage` entry must share a
       content token with something in the profile. This is where "has AWS
       experience" gets caught, while `skill_priorities` — where "learn AWS" belongs
       — is never inspected.

    Sub-check 3 is token overlap rather than exact matching so the model may rephrase
    ("debugging distributed systems" for "systems debugging") without being punished.
    """
    if _no_artifact(outputs):
        return {"key": "grounding_no_invention", "score": 0, "comment": "no artifact was produced"}

    reference = reference_outputs or {}
    facts = reference.get("confirmed_facts") or {}
    structured = outputs.get("structured") or {}

    violations: list[str] = []
    fact_text = _fact_text(structured)
    profile_tokens = _profile_tokens(facts)

    # 1. Money.
    stated_salary = facts.get("salary_expectation")
    money = CURRENCY_PATTERN.findall(fact_text)
    if money:
        if not stated_salary:
            violations.append(f"money claimed but salary was never collected: {money}")
        else:
            allowed = _tokens(str(stated_salary)) | set(
                re.findall(r"\d[\d,.]*", str(stated_salary))
            )
            for figure in money:
                digits = re.findall(r"\d[\d,.]*", figure)
                if digits and not any(d in allowed for d in digits):
                    violations.append(f"salary figure not in the collected value: {figure!r}")

    # 2. Seniority.
    stated_years = facts.get("years_experience")
    for match in YEARS_PATTERN.finditer(fact_text):
        digit, word = match.group(1), match.group(2)
        value = int(digit) if digit else WORD_NUMBERS.get((word or "").lower())
        if value is None:
            continue
        if stated_years is None:
            violations.append(f"years of experience claimed but never collected: {value}")
        elif value != int(stated_years):
            violations.append(f"years of experience {value} disagrees with {stated_years}")

    # 3. Claimed capability.
    profile_text = _profile_text(facts)
    for strength in structured.get("strengths_to_leverage") or []:
        if not _strength_is_supported(str(strength), profile_tokens, profile_text):
            violations.append(f"strength has no support in the profile: {strength!r}")

    return {
        "key": "grounding_no_invention",
        "score": int(not violations),
        "comment": ("; ".join(violations) if violations else "every user-specific claim traces to the profile"),
    }


# --------------------------------------------------------------------------
# 2. action_items_preserved
# --------------------------------------------------------------------------


def action_items_preserved(outputs: dict, reference_outputs: dict | None = None, **_: Any) -> dict:
    """Every confirmed step survives exactly, in source order, with nothing added.

    Exact string equality, not similarity: the user approved that wording, and
    Stage 2 makes rewording structurally impossible by never letting the model
    supply the text. This check is what would catch that guarantee regressing.
    """
    if _no_artifact(outputs):
        return {"key": "action_items_preserved", "score": 0, "comment": "no artifact was produced"}

    reference = reference_outputs or {}
    expected = list(reference.get("confirmed_action_items") or [])
    items = (outputs.get("structured") or {}).get("action_items") or []
    actual = [str(item.get("step", "")) for item in items if isinstance(item, dict)]

    if actual == expected:
        return {
            "key": "action_items_preserved",
            "score": 1,
            "comment": f"all {len(expected)} confirmed steps preserved in order",
        }

    problems: list[str] = []
    missing = [step for step in expected if step not in actual]
    added = [step for step in actual if step not in expected]
    if missing:
        problems.append(f"dropped or reworded: {missing}")
    if added:
        problems.append(f"not confirmed by the user: {added}")
    if not problems:
        problems.append(f"reordered: expected {expected}, got {actual}")

    return {
        "key": "action_items_preserved",
        "score": 0,
        "comment": "; ".join(problems),
    }


# --------------------------------------------------------------------------
# 3. unknowns_honest
# --------------------------------------------------------------------------


def unknowns_honest(outputs: dict, reference_outputs: dict | None = None, **_: Any) -> dict:
    """Missing fields are declared missing, and never contradicted elsewhere.

    Two halves, and the second is the one with teeth:

    * The `unknowns` list must equal the deterministic derivation. Since PR 5
      computes it rather than asking the model, this half only fails if that
      derivation regresses.
    * Nothing in a fact-asserting field may assert a value for a field the same
      document calls unknown. A artifact that lists "Salary expectation was never
      discussed" under one heading and quotes a target salary under another is not
      honest, however fluent it reads — and that self-contradiction is exactly what
      a grounding regression produces.
    """
    if _no_artifact(outputs):
        return {"key": "unknowns_honest", "score": 0, "comment": "no artifact was produced"}

    reference = reference_outputs or {}
    expected_unknowns = list(reference.get("unknowns") or [])
    facts = reference.get("confirmed_facts") or {}
    structured = outputs.get("structured") or {}
    reported = list(structured.get("unknowns") or [])

    problems: list[str] = []

    for entry in expected_unknowns:
        if entry not in reported:
            problems.append(f"missing field not declared: {entry!r}")
    for entry in reported:
        if entry not in expected_unknowns:
            problems.append(f"declared unknown but it was collected: {entry!r}")

    fact_text = _fact_text(structured)
    if not facts.get("salary_expectation") and CURRENCY_PATTERN.search(fact_text):
        problems.append("salary is declared unknown yet a figure is asserted")
    if facts.get("years_experience") is None and YEARS_PATTERN.search(fact_text):
        problems.append("experience is declared unknown yet a duration is asserted")

    return {
        "key": "unknowns_honest",
        "score": int(not problems),
        "comment": ("; ".join(problems) if problems else "gaps declared and never contradicted"),
    }


# --------------------------------------------------------------------------
# 4. artifact_complete
# --------------------------------------------------------------------------


def artifact_complete(outputs: dict, reference_outputs: dict | None = None, **_: Any) -> dict:
    """Every populated section is materially represented somewhere in the document.

    "Materially" means at least one of the section's collected values is traceable in
    the rendered artifact, not that a heading exists. Checked against the Markdown
    because a value may legitimately surface in prose rather than in a list field.
    """
    if _no_artifact(outputs):
        return {"key": "artifact_complete", "score": 0, "comment": "no artifact was produced"}

    reference = reference_outputs or {}
    facts = reference.get("confirmed_facts") or {}
    expected_sections = list(reference.get("expected_sections") or [])
    markdown_tokens = _tokens(outputs.get("markdown") or "")

    from fo_dataset import SECTION_FIELDS  # local: keeps this module import-light

    unrepresented: list[str] = []
    for section in expected_sections:
        section_tokens: set[str] = set()
        for field_name in SECTION_FIELDS.get(section, []):
            value = facts.get(field_name)
            if value is None or value == [] or value == "":
                continue
            if isinstance(value, list):
                for entry in value:
                    section_tokens |= _tokens(str(entry))
            else:
                section_tokens |= _tokens(str(value))
        if section_tokens and not (section_tokens & markdown_tokens):
            unrepresented.append(section)

    return {
        "key": "artifact_complete",
        "score": int(not unrepresented),
        "comment": (
            f"populated sections absent from the artifact: {unrepresented}"
            if unrepresented
            else f"all {len(expected_sections)} populated sections represented"
        ),
    }


# --------------------------------------------------------------------------
# 5. priority_valid
# --------------------------------------------------------------------------


def priority_valid(outputs: dict, reference_outputs: dict | None = None, **_: Any) -> dict:
    """Priorities are exactly 1..n, each once, and rendered in that order.

    A `FinalOutput` validator already enforces the numbering, so this is a
    belt-and-braces check on the *rendered* result — it would catch the renderer
    emitting items out of priority order, which the validator cannot see.
    """
    if _no_artifact(outputs):
        return {"key": "priority_valid", "score": 0, "comment": "no artifact was produced"}

    structured = outputs.get("structured") or {}
    items = structured.get("action_items") or []
    priorities = [item.get("priority") for item in items if isinstance(item, dict)]
    expected = list(range(1, len(priorities) + 1))

    problems: list[str] = []
    if sorted(priorities) != expected:
        problems.append(f"priorities are not a contiguous total order: {priorities}")

    markdown = outputs.get("markdown") or ""
    positions = []
    for index in expected:
        marker = f"{index}. **"
        positions.append(markdown.find(marker))
    if positions and any(pos == -1 for pos in positions):
        problems.append("not every priority is rendered as a numbered item")
    elif positions != sorted(positions):
        problems.append("rendered order disagrees with priority order")

    return {
        "key": "priority_valid",
        "score": int(not problems),
        "comment": ("; ".join(problems) if problems else f"contiguous 1..{len(priorities)}, rendered in order"),
    }


# --------------------------------------------------------------------------
# 6. render_deterministic
# --------------------------------------------------------------------------


def render_deterministic(outputs: dict, reference_outputs: dict | None = None, **_: Any) -> dict:
    """Re-rendering the same structured output must reproduce the same bytes.

    The runner records `render_repeat` by calling `render_final_output` a second
    time. If they ever differ, every other Markdown-level assertion in this file
    becomes unreliable, so this is the check that protects the others.
    """
    if _no_artifact(outputs):
        return {"key": "render_deterministic", "score": 0, "comment": "no artifact was produced"}

    markdown = outputs.get("markdown")
    repeat = outputs.get("render_repeat")

    if markdown is None or repeat is None:
        return {
            "key": "render_deterministic",
            "score": 0,
            "comment": "runner did not provide a repeat render",
        }

    identical = markdown == repeat
    return {
        "key": "render_deterministic",
        "score": int(identical),
        "comment": (
            "byte-identical on re-render"
            if identical
            else f"re-render differed ({len(markdown)} vs {len(repeat)} chars)"
        ),
    }


DETERMINISTIC_EVALUATORS = [
    grounding_no_invention,
    action_items_preserved,
    unknowns_honest,
    artifact_complete,
    priority_valid,
    render_deterministic,
]
