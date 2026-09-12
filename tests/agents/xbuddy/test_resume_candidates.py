"""Background candidate extraction at upload: structured, unconfirmed, never guessed.

Offline: a fake chain stands in for the model. The years-of-experience guard is
deterministic and is tested against the corpus text directly.
"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from agents.xbuddy.models import BackgroundExtract
from agents.xbuddy.resume.candidates import (
    MAX_RESUME_CHARS,
    RESUME_BACKGROUND_RULES,
    extract_background_candidates,
    years_stated,
)

CORPUS = Path(__file__).resolve().parents[3] / "evals" / "resume_retrieval" / "corpus"
RESUME = (CORPUS / "backend_to_ai.txt").read_text(encoding="utf-8")


class FakeChain:
    def __init__(self, parsed=None, raises=None, parsing_error=None):
        self.parsed, self.raises, self.parsing_error = parsed, raises, parsing_error
        self.calls: list = []

    async def ainvoke(self, messages, config=None):
        self.calls.append(messages)
        if self.raises:
            raise self.raises
        return {"raw": AIMessage(content=""), "parsed": self.parsed, "parsing_error": self.parsing_error}


def extract(**fields):
    return BackgroundExtract(**{"current_role": None, "years_experience": None,
                                "highest_education": None, "work_history": None, **fields})


# --------------------------------------------------------------------------
# Years: stated, never inferred
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "8 years of experience",
    "8+ years of professional experience building APIs",
    "eight years of experience",
    "8 yrs experience",
    "Experience: 8 years",
    "Backend engineer with 8 years of experience in Python",
])
def test_stated_years_of_experience_are_recognised(text):
    assert years_stated(text, 8)


@pytest.mark.parametrize("text", ["2017 - 2025", "Mar 2020 – Present", "8 projects", "18 years", "80 years"])
def test_dates_and_other_numbers_are_not_a_statement_of_years(text):
    assert not years_stated(text, 8)


@pytest.mark.parametrize(("text", "years"), [
    # From a real resume: the number is real, but it describes one posting.
    ("Backed by four years embedded in infotainment product development, validating user flows.", 4),
    ("Led the platform for 8 years at Northwind Logistics", 8),
    ("Three years on the payments core", 3),
    # Far enough away to be a different statement.
    ("Hands-on experience shipping LLM features. Backed by four years in automotive QA.", 4),
])
def test_years_stated_about_something_else_are_not_years_of_experience(text, years):
    """A wrong proposal costs the user a correction; asking costs one question."""
    assert not years_stated(text, years)


@pytest.mark.asyncio
async def test_years_the_resume_states_are_kept():
    chain = FakeChain(extract(current_role="Senior Backend Engineer", years_experience=8))
    facts = await extract_background_candidates(RESUME, chain=chain)  # "8 years of experience" is in the text
    assert facts["years_experience"] == 8


@pytest.mark.asyncio
async def test_years_calculated_from_dates_are_dropped():
    """The model returned 9 — the span of the dates — but the resume never says 9 years."""
    chain = FakeChain(extract(current_role="Senior Backend Engineer", years_experience=9))
    facts = await extract_background_candidates(RESUME, chain=chain)
    assert facts["years_experience"] is None
    assert facts["current_role"] == "Senior Backend Engineer"


# --------------------------------------------------------------------------
# Shape and failure
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_the_four_background_fields_are_returned():
    chain = FakeChain(extract(current_role="r", highest_education="BASc", work_history=["a", " ", "b"]))
    facts = await extract_background_candidates(RESUME, chain=chain)
    assert set(facts) == {"current_role", "years_experience", "highest_education", "work_history"}
    assert facts["work_history"] == ["a", "b"]


@pytest.mark.asyncio
async def test_an_all_empty_extraction_is_no_candidates():
    assert await extract_background_candidates(RESUME, chain=FakeChain(extract())) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("chain", [
    FakeChain(raises=RuntimeError("openai 503")),
    FakeChain(parsed=None, parsing_error=ValueError("bad json")),
])
async def test_a_failed_extraction_is_no_candidates_not_an_error(chain):
    assert await extract_background_candidates(RESUME, chain=chain) is None


@pytest.mark.asyncio
async def test_blank_text_makes_no_call():
    chain = FakeChain(extract(current_role="r"))
    assert await extract_background_candidates("   ", chain=chain) is None
    assert chain.calls == []


@pytest.mark.asyncio
async def test_the_model_gets_the_rules_and_a_bounded_resume():
    chain = FakeChain(extract(current_role="r"))
    await extract_background_candidates("x" * (MAX_RESUME_CHARS + 5000), chain=chain)
    system, human = chain.calls[0]
    assert system.content == RESUME_BACKGROUND_RULES.strip()
    assert len(human.content) <= MAX_RESUME_CHARS + len("RESUME\n\n")


def test_the_rules_forbid_inferring_years_from_dates():
    assert "Never calculate it from dates" in RESUME_BACKGROUND_RULES


def test_the_default_chain_is_memory_updaters_background_chain(monkeypatch):
    """Reuse, not a second structured-extraction setup."""
    from agents.xbuddy.nodes import memory_updater
    from agents.xbuddy.resume import candidates

    built = []
    monkeypatch.setattr(memory_updater, "_extraction_chain", lambda model: built.append(model) or "chain")
    assert candidates._chain() == "chain" and built == [BackgroundExtract]
