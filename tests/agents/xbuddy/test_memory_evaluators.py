"""Offline tests for the memory eval's evaluators and seeded defect.

The evaluators are pure, so they can be exercised without a key or a network.
That matters twice over: it keeps ordinary pytest offline, and it lets the
seeded-bug proof run here as well — the mutation is applied to the real
`extraction` module and the resulting data is scored, with no model call.
"""

import sys
from pathlib import Path

import pytest

EVALS_DIR = Path(__file__).resolve().parents[3] / "evals" / "memory"
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from dataset import CASES, as_langsmith_examples
from evaluators import (
    EVALUATORS,
    advances_unfilled_field,
    asks_about,
    extraction_no_clobber,
    known_block_complete,
    no_redundant_question,
)

from agents.xbuddy.enums import SectionID
from agents.xbuddy.models import EXTRACT_MODELS, XBuddyData
from agents.xbuddy.prompts import get_section_template


def score(evaluator, **outputs) -> int:
    reference = outputs.pop("reference", None)
    return evaluator(outputs, reference)["score"]


# --------------------------------------------------------------------------
# Dataset integrity
# --------------------------------------------------------------------------


def test_dataset_size_and_section_coverage():
    assert 8 <= len(CASES) <= 12, f"expected 8-12 cases, got {len(CASES)}"
    assert len({case["id"] for case in CASES}) == len(CASES), "case ids must be unique"

    sections = {case["section"] for case in CASES}
    assert sections == {section.value for section in SectionID}, (
        f"every section should appear; missing {sorted({s.value for s in SectionID} - sections)}"
    )


def test_dataset_fields_are_real_and_consistent():
    """Guards the dataset against drift in XBuddyData or the extract models."""
    known_fields = set(XBuddyData.model_fields)

    for case in CASES:
        section = SectionID(case["section"])
        unknown = set(case["known"]) - known_fields
        assert not unknown, f"{case['id']}: unknown XBuddyData fields {sorted(unknown)}"

        extract_fields = set(EXTRACT_MODELS[section].model_fields)
        stray = set(case["extraction"]) - extract_fields
        assert not stray, f"{case['id']}: extraction keys not on the schema {sorted(stray)}"

        required = set(get_section_template(section).required_fields)
        bad_missing = set(case["missing"]) - required
        assert not bad_missing, f"{case['id']}: 'missing' names non-required {sorted(bad_missing)}"

        # A field cannot be both already-known and still-missing.
        assert not (set(case["missing"]) & set(case["known"])), case["id"]


def test_refinement_pending_names_stored_required_fields():
    """`refinement_pending` means "stored, but the prompt prescribes a follow-up".

    So each entry must be a required field of its section that is already in `known`
    and therefore *not* in `missing`. A stray entry would silently exempt a genuine
    re-ask from `no_redundant_question`, which is the check that matters most.
    """
    for case in CASES:
        refinement = case.get("refinement_pending", [])
        assert isinstance(refinement, list), case["id"]
        required = set(get_section_template(SectionID(case["section"])).required_fields)
        for field in refinement:
            assert field in required, f"{case['id']}: {field} is not required by the section"
            assert field in case["known"], f"{case['id']}: {field} is not stored, so nothing to refine"
            assert field not in case["missing"], f"{case['id']}: {field} cannot be missing too"


def test_cases_with_a_prescribed_follow_up_declare_it():
    """The two prompt-mandated follow-ups must be annotated, or the eval mis-scores.

    Both were false failures in the first live run: the agent was following its
    section prompt and the dataset had not said so. Derived from the prompts rather
    than hard-coded to case ids, so a new case inherits the requirement.
    """
    for case in CASES:
        refinement = case.get("refinement_pending", [])

        # Job Preferences: "If they say hybrid, ask how many days on-site."
        if "hybrid" in [mode.lower() for mode in case["known"].get("preferred_work_modes", [])]:
            assert "preferred_work_modes" in refinement, (
                f"{case['id']}: hybrid is stored, so the days-on-site follow-up is prescribed"
            )

        # Skill Assessment: "ask for an example alongside each [strength]."
        if case["section"] == SectionID.SKILL_ASSESSMENT.value and case["known"].get("strengths"):
            assert "strengths" in refinement, (
                f"{case['id']}: strengths are stored, so the example follow-up is prescribed"
            )


def test_dataset_includes_a_correction_and_a_complete_section():
    assert any(case["extraction"] for case in CASES), "need at least one correction case"
    assert any(not case["missing"] for case in CASES), "need a fully-complete section case"
    assert any(case["missing"] and case["known"] for case in CASES), "need a partial section"


def test_langsmith_examples_line_up():
    inputs, outputs = as_langsmith_examples()
    assert len(inputs) == len(outputs) == len(CASES)
    assert inputs[0]["case_id"] == CASES[0]["id"]


def test_langsmith_examples_carry_what_the_evaluators_read():
    """The evaluators receive `reference_outputs`, so anything they read must ship.

    `refinement_pending` and `section` are consumed by `no_redundant_question` and
    `advances_unfilled_field`; if the export drops them the exemptions silently stop
    applying in the live run while every offline test still passes.
    """
    _, outputs = as_langsmith_examples()

    for case, exported in zip(CASES, outputs, strict=True):
        assert exported["section"] == case["section"]
        assert exported["refinement_pending"] == case.get("refinement_pending", [])
        assert exported["missing_fields"] == case["missing"]


# --------------------------------------------------------------------------
# asks_about — the shared heuristic
# --------------------------------------------------------------------------


def test_asks_about_detects_a_direct_question():
    assert asks_about("What kind of role are you targeting?", "target_roles") is True


def test_asks_about_ignores_statements():
    """A deferral sitting beside a question must not read as that question.

    The question names `target_roles`, so it is not topically bare and the
    preceding sentence is never consulted — "salary" stays unseen.
    """
    assert asks_about("Salary comes later. What role are you after?", "salary_expectation") is False


def test_asks_about_reads_the_preceding_sentence_for_a_bare_question():
    """Tier 2: a referential question takes its topic from the sentence before it."""
    reply = "Let's capture your highest completed education next. What is it?"
    assert asks_about(reply, "highest_education") is True


def test_asks_about_does_not_look_back_past_one_sentence():
    """The lookback is one sentence, never the whole reply."""
    reply = "Your education is on file. Let's move on. What is it you'd like to add?"
    assert asks_about(reply, "highest_education") is False


def test_asks_about_ignores_an_acknowledgement_before_a_question_about_something_else():
    """The live regression that motivated gating tier 2 on a bare question.

    An unconditional two-sentence window read "we'll aim for that timeline" as a
    timeline re-ask, even though the question asks for the role.
    """
    reply = "Great, we'll aim for that timeline. What role are you looking to move into next?"
    assert asks_about(reply, "target_timeline", "within 3 months") is False
    assert asks_about(reply, "target_roles") is True


@pytest.mark.parametrize(
    ("reply", "field"),
    [
        # The Career Goal prompt's own words for career_goal_summary.
        ("What do you want this move to change?", "career_goal_summary"),
        # The Skill Assessment prompt requires evidence alongside each strength.
        ("Can you share an example of a project where that helped?", "strengths"),
    ],
)
def test_asks_about_recognises_the_shipped_prompt_phrasings(reply, field):
    """Terms must match how the PR 2 prompts actually ask.

    Both phrasings scored as advancing nothing in the first live run, which was an
    evaluator gap rather than agent behaviour.
    """
    assert asks_about(reply, field) is True


def test_asks_about_treats_an_echoed_value_as_acknowledgement():
    """Mentioning a known value while asking about something else is not a re-ask."""
    reply = "For a Senior SRE role, when would you like to move?"
    assert asks_about(reply, "target_roles", ["Senior SRE"]) is False
    assert asks_about(reply, "target_timeline") is True


def test_asks_about_flags_a_re_ask_that_does_not_echo():
    assert asks_about("What job title are you aiming for?", "target_roles", ["Senior SRE"]) is True


@pytest.mark.parametrize(
    ("reply", "field"),
    [
        # A bare noun mentioned while asking about something else must not flag.
        ("When would you like to be in the new role?", "target_roles"),
        ("Where would you like to start on that plan?", "preferred_locations"),
        # "plan" and "action" alone are no longer terms; only real solicitations are.
        ("That plan looks solid. What's your salary expectation?", "action_items"),
    ],
)
def test_asks_about_does_not_flag_incidental_nouns(reply, field):
    """Regression guard for over-broad terms.

    "role" once matched a timeline question, "where" an action-plan question.
    Phrase-level terms fix it; this fails if anyone reintroduces a bare noun.
    """
    assert asks_about(reply, field) is False


# --------------------------------------------------------------------------
# known_block_complete
# --------------------------------------------------------------------------


def test_known_block_complete_passes_when_everything_is_rendered():
    from agents.xbuddy.context import render_known_data

    data = XBuddyData(target_roles=["Senior SRE"], target_timeline="3 months")
    assert (
        score(known_block_complete, known_block=render_known_data(data), after=data.model_dump())
        == 1
    )


def test_known_block_complete_fails_when_a_field_is_missing_from_the_block():
    data = XBuddyData(target_roles=["Senior SRE"], target_timeline="3 months")
    partial = "- Target role(s): Senior SRE"  # timeline omitted
    result = known_block_complete({"known_block": partial, "after": data.model_dump()}, None)

    assert result["score"] == 0
    assert "target_timeline" in result["comment"]


# --------------------------------------------------------------------------
# extraction_no_clobber
# --------------------------------------------------------------------------


def test_no_clobber_passes_for_an_unchanged_merge():
    data = XBuddyData(target_roles=["SRE"], target_timeline="3 months").model_dump()
    assert score(extraction_no_clobber, before=data, after=data) == 1


def test_no_clobber_allows_a_correction_overwrite():
    before = XBuddyData(target_timeline="3 months").model_dump()
    after = XBuddyData(target_timeline="6 months").model_dump()
    assert score(extraction_no_clobber, before=before, after=after) == 1


@pytest.mark.parametrize("wiped", [None, [], ""])
def test_no_clobber_fails_when_a_value_is_emptied(wiped):
    before = XBuddyData(target_roles=["SRE"]).model_dump()
    after = dict(before)
    after["target_roles"] = wiped

    result = extraction_no_clobber({"before": before, "after": after}, None)
    assert result["score"] == 0
    assert "target_roles" in result["comment"]


# --------------------------------------------------------------------------
# no_redundant_question
# --------------------------------------------------------------------------


def test_no_redundant_question_passes_when_asking_something_new():
    before = XBuddyData(target_roles=["Senior SRE"]).model_dump()
    reply = "Got it. When would you like to be in the new role?"
    assert score(no_redundant_question, reply=reply, before=before) == 1


def test_no_redundant_question_fails_on_a_re_ask():
    before = XBuddyData(target_roles=["Senior SRE"]).model_dump()
    reply = "Great. What job title are you aiming for?"
    result = no_redundant_question({"reply": reply, "before": before}, None)

    assert result["score"] == 0
    assert "target_roles" in result["comment"]


def test_no_redundant_question_exempts_a_prescribed_refinement():
    """`refinement_pending` is a follow-up the section prompt requires, not a re-ask.

    "hybrid" is stored, and Job Preferences says to ask how many days on-site — so
    the work-mode question is correct behaviour here.
    """
    before = XBuddyData(preferred_work_modes=["hybrid"]).model_dump()
    reply = "How many days on-site would you be comfortable with?"

    assert score(no_redundant_question, reply=reply, before=before, reference={}) == 0
    assert (
        score(
            no_redundant_question,
            reply=reply,
            before=before,
            reference={"refinement_pending": ["preferred_work_modes"]},
        )
        == 1
    )


def test_no_redundant_question_exemption_is_per_case_not_global():
    """A case without the annotation still counts a work-mode question as a re-ask.

    The reply must not restate "hybrid", or echo suppression would pass it for a
    different reason and the test would not be about the exemption at all.
    """
    before = XBuddyData(preferred_work_modes=["hybrid"]).model_dump()
    reply = "Would you rather be fully remote, or in the office?"
    reference = {"refinement_pending": ["strengths"]}

    result = no_redundant_question({"reply": reply, "before": before}, reference)
    assert result["score"] == 0
    assert "preferred_work_modes" in result["comment"]


def test_no_redundant_question_judges_against_pre_turn_memory():
    """What the user already said, whether or not the merge kept it.

    Scoring against post-merge data would let a clobbering bug hide: the field
    would look unknown, so re-asking it would look legitimate.
    """
    before = XBuddyData(target_timeline="3 months").model_dump()
    after = XBuddyData().model_dump()  # the bug wiped it
    reply = "When are you hoping to move?"

    result = no_redundant_question({"reply": reply, "before": before, "after": after}, None)
    assert result["score"] == 0


# --------------------------------------------------------------------------
# advances_unfilled_field
# --------------------------------------------------------------------------


def test_advances_unfilled_field_passes_when_targeting_a_gap():
    reply = "When would you like to start the new role?"
    assert (
        score(advances_unfilled_field, reply=reply, reference={"missing_fields": ["target_timeline"]})
        == 1
    )


def test_advances_unfilled_field_fails_when_asking_nothing_relevant():
    reply = "That sounds great, thanks for sharing."
    result = advances_unfilled_field({"reply": reply}, {"missing_fields": ["target_timeline"]})
    assert result["score"] == 0


def test_advances_unfilled_field_is_not_applicable_when_complete():
    result = advances_unfilled_field({"reply": "All set."}, {"missing_fields": []})
    assert result["score"] == 1
    assert "complete" in result["comment"]


def test_advances_unfilled_field_counts_a_prescribed_refinement_as_progress():
    """Refining a stored value moves the section forward as much as filling a gap."""
    reply = "You said hybrid — how many days on-site would work?"
    result = advances_unfilled_field(
        {"reply": reply},
        {"missing_fields": ["salary_expectation"], "refinement_pending": ["preferred_work_modes"]},
    )
    assert result["score"] == 1
    assert "preferred_work_modes" in result["comment"]


def test_advances_unfilled_field_exempts_action_plan():
    """Action Plan proposes its items; asking the user to invent them is the defect.

    The prompt says "Propose a first draft yourself... Do not ask the user to invent
    the plan from nothing", so a reply that delivers a plan and asks which steps feel
    realistic is ideal — scoring it as a stall measures the wrong thing.
    """
    reply = "Here is a first draft: 1) ship a Kubernetes side project 2) get the CKA."
    result = advances_unfilled_field(
        {"reply": reply}, {"missing_fields": ["action_items"], "section": "action_plan"}
    )
    assert result["score"] == 1
    assert "action_plan" in result["comment"]


def test_advances_unfilled_field_exemption_does_not_leak_to_other_sections():
    """Only Action Plan is exempt; the four collection sections keep full strength."""
    reply = "Here is a first draft: 1) ship a Kubernetes side project 2) get the CKA."
    result = advances_unfilled_field(
        {"reply": reply}, {"missing_fields": ["target_timeline"], "section": "career_goal"}
    )
    assert result["score"] == 0


def test_all_four_evaluators_are_registered():
    keys = [
        evaluator({"reply": "", "known_block": "", "before": {}, "after": {}}, {})["key"]
        for evaluator in EVALUATORS
    ]
    assert keys == [
        "known_block_complete",
        "extraction_no_clobber",
        "no_redundant_question",
        "advances_unfilled_field",
    ]


# --------------------------------------------------------------------------
# The seeded bug, proven offline against the real merge path
# --------------------------------------------------------------------------


def build_extract(section: SectionID, values: dict):
    model = EXTRACT_MODELS[section]
    payload = dict.fromkeys(model.model_fields, None)
    payload.update(values)
    return model(**payload)


def test_clean_merge_keeps_everything_for_every_case():
    """Baseline: with healthy code no case loses a stored value."""
    from agents.xbuddy.context import render_known_data
    from agents.xbuddy.extraction import merge_extraction

    for case in CASES:
        section = SectionID(case["section"])
        before = XBuddyData.model_validate(case["known"])
        after = merge_extraction(build_extract(section, case["extraction"]), before)

        outputs = {
            "reply": "",
            "known_block": render_known_data(after),
            "before": before.model_dump(),
            "after": after.model_dump(),
        }
        assert extraction_no_clobber(outputs, None)["score"] == 1, case["id"]
        assert known_block_complete(outputs, None)["score"] == 1, case["id"]


def test_seeded_bug_is_detected_by_the_evaluators(monkeypatch):
    """The proof, without any model call.

    `seed_memory_bug` mutates the real `_is_no_op`, so `merge_extraction` applies
    null values instead of skipping them and previously stored fields are wiped.
    The evaluators must notice — otherwise the eval cannot demonstrate it detects
    a genuine regression.
    """
    from run_experiment import seed_memory_bug

    from agents.xbuddy import extraction as extraction_module
    from agents.xbuddy.context import render_known_data

    monkeypatch.setattr(extraction_module, "_is_no_op", extraction_module._is_no_op)
    seed_memory_bug()

    clobbered_cases: list[str] = []
    block_failures: list[str] = []

    for case in CASES:
        if not case["known"]:
            continue
        section = SectionID(case["section"])
        before = XBuddyData.model_validate(case["known"])
        after = extraction_module.merge_extraction(
            build_extract(section, case["extraction"]), before
        )

        outputs = {
            "reply": "",
            "known_block": render_known_data(after),
            "before": before.model_dump(),
            "after": after.model_dump(),
        }
        if extraction_no_clobber(outputs, None)["score"] == 0:
            clobbered_cases.append(case["id"])
        if known_block_complete(outputs, None)["score"] == 0:
            block_failures.append(case["id"])

    assert clobbered_cases, "the seeded bug must trip extraction_no_clobber"
    # Losing the value also removes it from the rendering the model is shown,
    # which is the mechanism that makes the agent re-ask.
    assert not block_failures, (
        "known_block_complete scores the *post-merge* data, so a wiped field is "
        "consistently absent from both — the re-ask is caught by "
        "no_redundant_question, which scores against pre-turn memory"
    )


def test_seeded_bug_makes_a_re_ask_score_as_redundant():
    """The second required failure, shown end to end on real merged data."""
    from run_experiment import seed_memory_bug

    from agents.xbuddy import extraction as extraction_module

    seed_memory_bug()

    case = next(c for c in CASES if c["id"] == "role_known_timeline_missing")
    before = XBuddyData.model_validate(case["known"])
    after = extraction_module.merge_extraction(
        build_extract(SectionID(case["section"]), case["extraction"]), before
    )

    assert after.target_roles in (None, []), "the bug should have wiped the role"

    # With the role no longer in memory, the agent would naturally ask for it.
    reply = "Sure — what job title are you aiming for?"
    outputs = {"reply": reply, "before": before.model_dump(), "after": after.model_dump()}

    assert no_redundant_question(outputs, None)["score"] == 0
    assert extraction_no_clobber(outputs, None)["score"] == 0


@pytest.fixture(autouse=True)
def restore_is_no_op():
    """Undo the seeded mutation so it cannot leak into other tests."""
    from agents.xbuddy import extraction as extraction_module

    original = extraction_module._is_no_op
    yield
    extraction_module._is_no_op = original
