"""Background candidate facts, extracted once from the whole resume at upload.

Structured extraction, not retrieval: "what is their current role, how many years,
what education" are questions about the whole document, and top-k chunks can miss
entries. So the four Background fields are read from the full text, once, with the
same `BackgroundExtract` schema and the same strict chain `memory_updater` uses for
the Background section.

The result is a **candidate**, stored on the resume row and never in `user_data`.
JobBuddy proposes it; the user confirms or corrects; the existing extraction of the
user's own replies is what finally records it.

Failure here is not an upload failure. Candidates make Background shorter; without
them the section simply asks as it always has.
"""

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..models import BackgroundExtract

logger = logging.getLogger(__name__)

# A ten-page resume is well under this; the bound keeps one upload's cost bounded.
MAX_RESUME_CHARS = 40_000

RESUME_BACKGROUND_RULES = """You read a resume and extract four Background facts about the person
who wrote it. You do not talk to anyone and you do not write prose.

WHAT TO EXTRACT
- current_role: their current or most recent role, as the resume names it
  (include the organisation if the resume gives it).
- years_experience: ONLY a total number of years of experience the resume states
  in words, for example "8 years of experience". Never calculate it from dates.
  A number of years spent on one employer, product or technology ("four years
  embedded in infotainment development") is not it. If the resume states no
  total, return null.
- highest_education: the highest completed qualification, as written.
- work_history: up to four roles, most recent first, one line each:
  role, organisation, dates as written.

NEVER GUESS
- Return null for anything the resume does not state.
- Do not infer seniority, years, or roles from job titles or dates.
- Copy facts as written. Do not summarise achievements or add detail.
"""

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}


# How far from "N years" the word "experience" may sit and still be describing it.
_EXPERIENCE_WINDOW = 40


def years_stated(text: str, years: int) -> bool:
    """Whether the resume states `years` years *of experience* — "8 years of
    experience", "8+ yrs experience", "Experience: eight years".

    The rule "never infer years of experience from dates" cannot be left to a
    prompt alone. Anchoring on the word "experience" also rejects a number of
    years the resume states about something else: on a real resume reading "four
    years embedded in infotainment product development", 4 was proposed as that
    person's total experience, which was wrong and which they would have had to
    correct. A dropped number is asked for as an ordinary question instead.
    """
    words = [word for word, value in _NUMBER_WORDS.items() if value == years]
    forms = [re.escape(str(years))] + words
    pattern = rf"\b(?:{'|'.join(forms)})\s*\+?\s*(?:years?|yrs?)\b"
    for match in re.finditer(pattern, text, re.IGNORECASE):
        nearby = (
            text[max(0, match.start() - _EXPERIENCE_WINDOW) : match.start()],
            text[match.end() : match.end() + _EXPERIENCE_WINDOW],
        )
        if any(re.search(r"\bexperience\b", part, re.IGNORECASE) for part in nearby):
            return True
    return False


def _empty(value: Any) -> bool:
    return value is None or value == [] or (isinstance(value, str) and not value.strip())


def _chain():
    from ..nodes.memory_updater import _extraction_chain

    return _extraction_chain(BackgroundExtract)


async def extract_background_candidates(text: str, *, chain: Any = None) -> dict[str, Any] | None:
    """Unconfirmed Background candidates from the resume text, or None. Never raises."""
    if not text or not text.strip():
        return None

    runnable = chain if chain is not None else _chain()
    messages = [
        SystemMessage(content=RESUME_BACKGROUND_RULES.strip()),
        HumanMessage(content=f"RESUME\n\n{text[:MAX_RESUME_CHARS]}"),
    ]
    try:
        result = await runnable.ainvoke(messages)
    except Exception:
        logger.exception("resume candidates: extraction call failed; uploading without candidates")
        return None

    parsed = result.get("parsed") if isinstance(result, dict) else result
    if parsed is None:
        error = result.get("parsing_error") if isinstance(result, dict) else None
        logger.warning("resume candidates: no parse (%s); uploading without candidates", error)
        return None

    try:
        facts = BackgroundExtract.model_validate(parsed).model_dump()
    except Exception:  # noqa: BLE001
        logger.warning("resume candidates: output did not match BackgroundExtract")
        return None

    years = facts.get("years_experience")
    if years is not None and not years_stated(text, int(years)):
        logger.info("resume candidates: dropped years_experience=%s (not stated in the resume)", years)
        facts["years_experience"] = None

    if isinstance(facts.get("work_history"), list):
        facts["work_history"] = [line for line in facts["work_history"] if not _empty(line)] or None

    return facts if any(not _empty(value) for value in facts.values()) else None
