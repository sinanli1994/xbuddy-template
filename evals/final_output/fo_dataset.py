"""Cases for the JobBuddy final-artifact eval.

Each case is a **completed** conversation: five sections done, `XBuddyData`
populated to whatever degree that case is about, and a confirmed action plan. The
eval runs real synthesis over that profile and scores the artifact.

Every case carries a stable `case_id`, which is the dataset synchronization key —
see `evals/memory/sync.py`, reused here rather than reimplemented.

The `fo_` prefix on this eval's module names is load-bearing. Both eval
directories are put on `sys.path`, and PR 4's `evals/memory/` also contains a
`dataset.py` and an `evaluators.py`. Without distinct names, whichever imports
first wins and the other eval silently gets the wrong module — which is exactly
what happened when both test files ran in one pytest session. The runner is named
`run_final_output_eval.py` for the same reason: PR 4's test imports
`seed_memory_bug` from `run_experiment`, and a second module of that name shadowed it.

What the reference data distinguishes
------------------------------------
* **confirmed facts** — the populated `XBuddyData` fields. Anything user-specific in
  the artifact must trace back to these.
* **confirmed action items** — Section 5's agreed plan. Authoritative and closed:
  it must survive exactly and in order.
* **deterministic unknowns** — computed by the production `derive_unknowns`, not
  hand-written, so the expectation cannot drift from the code.
* **expected sections** — which populated sections must be materially represented.

Deliberately no expected prose. The artifact's wording is the model's; only its
grounding, completeness, and ordering are the contract.
"""

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agents.xbuddy.models import XBuddyData
from agents.xbuddy.synthesis import derive_unknowns

DATASET_NAME = "jobbuddy-final-output"

# Which XBuddyData fields belong to which section. Used to decide whether a section
# is "populated" for this case, and therefore whether the artifact must represent it.
SECTION_FIELDS: dict[str, list[str]] = {
    "career_goal": ["target_roles", "career_goal_summary", "target_timeline"],
    "background": ["current_role", "years_experience", "highest_education", "work_history"],
    "job_preferences": [
        "preferred_locations",
        "preferred_work_modes",
        "target_industries",
        "employment_types",
        "salary_expectation",
    ],
    "skill_assessment": ["strengths", "current_skills", "skill_gaps"],
    "action_plan": ["action_items"],
}


CASES: list[dict[str, Any]] = [
    {
        "id": "fully_populated_sre",
        "profile": {
            "target_roles": ["Senior SRE"],
            "career_goal_summary": "Wants platform ownership and more autonomy",
            "target_timeline": "within 6 months",
            "current_role": "QA Analyst",
            "years_experience": 4,
            "highest_education": "BSc Computer Science",
            "work_history": ["QA Analyst at Acme", "Junior tester at Bolt"],
            "preferred_locations": ["Berlin"],
            "preferred_work_modes": ["hybrid"],
            "target_industries": ["fintech"],
            "employment_types": ["full-time"],
            "salary_expectation": "75-85k EUR",
            "strengths": ["systems debugging", "test automation"],
            "current_skills": ["Python", "Postgres", "Docker"],
            "skill_gaps": ["Kubernetes", "Terraform"],
            "action_items": [
                "Rewrite the CV summary to lead with the test-automation platform work",
                "Ship a small Kubernetes project and write it up",
                "Message three former colleagues at Berlin fintechs this week",
            ],
        },
        "notes": "Nothing missing. The baseline: every section must be represented.",
    },
    {
        "id": "partial_no_preferences",
        "profile": {
            "target_roles": ["Data Engineer"],
            "career_goal_summary": "Move from reporting into pipelines",
            "target_timeline": "within 3 months",
            "current_role": "BI Analyst",
            "years_experience": 6,
            "highest_education": "MSc Statistics",
            "strengths": ["SQL modelling"],
            "current_skills": ["SQL", "dbt", "Excel"],
            "skill_gaps": ["Airflow", "Spark"],
            "action_items": [
                "Build one end-to-end dbt project on public data",
                "Take the Airflow fundamentals course",
                "Apply to four pipeline roles a week",
            ],
        },
        "notes": (
            "No preferences at all: location, work mode, industry, employment type "
            "and salary are unknown. The strongest test of honest unknowns."
        ),
    },
    {
        "id": "career_transition_teacher",
        "profile": {
            "target_roles": ["Instructional Designer", "Learning Experience Designer"],
            "career_goal_summary": "Leave classroom teaching for product-based learning design",
            "target_timeline": "within 9 months",
            "current_role": "Secondary school teacher",
            "years_experience": 11,
            "highest_education": "PGCE",
            "work_history": ["Head of department", "Classroom teacher"],
            "preferred_locations": ["Remote"],
            "preferred_work_modes": ["remote"],
            "employment_types": ["full-time"],
            "strengths": ["curriculum design", "explaining hard ideas simply"],
            "current_skills": ["curriculum design", "assessment writing"],
            "skill_gaps": ["Figma", "e-learning authoring tools"],
            "action_items": [
                "Rebuild two lesson sequences as a portfolio case study",
                "Learn Articulate Storyline to a working level",
                "Speak to two instructional designers who used to teach",
            ],
        },
        "notes": (
            "Career changer. Positioning must build a bridge from teaching without "
            "inventing industry experience the user does not have."
        ),
    },
    {
        "id": "aggressive_timeline",
        "profile": {
            "target_roles": ["Backend Engineer"],
            "career_goal_summary": "Needs a role quickly after a layoff",
            "target_timeline": "within 4 weeks",
            "current_role": "Backend Engineer (recently redundant)",
            "years_experience": 5,
            "preferred_locations": ["Manchester", "Remote"],
            "preferred_work_modes": ["remote", "hybrid"],
            "employment_types": ["full-time", "contract"],
            "strengths": ["Go services", "incident response"],
            "current_skills": ["Go", "PostgreSQL", "AWS"],
            "action_items": [
                "Apply to eight roles a day from a shortlist",
                "Ask four ex-colleagues for referrals today",
                "Prepare one strong incident story for interviews",
            ],
        },
        "notes": (
            "Four weeks. Timeframes must be sized to that, and the plan must not be "
            "paced as if there were six months."
        ),
    },
    {
        "id": "strong_skills_thin_preferences",
        "profile": {
            "target_roles": ["ML Engineer"],
            "career_goal_summary": "Move from research scripts to production ML",
            "target_timeline": "within 6 months",
            "current_role": "Research Assistant",
            "years_experience": 3,
            "highest_education": "PhD Physics",
            "strengths": ["numerical modelling", "PyTorch", "writing up results"],
            "current_skills": ["Python", "PyTorch", "NumPy", "LaTeX"],
            "skill_gaps": ["MLOps", "cloud deployment"],
            "action_items": [
                "Deploy one model behind an API and document the latency",
                "Convert the thesis chapter into a public write-up",
                "Join two ML engineering communities and ask for CV feedback",
            ],
        },
        "notes": "Deep skills, no preferences. Search targets must stay honest about that.",
    },
    {
        "id": "clear_preferences_thin_skills",
        "profile": {
            "target_roles": ["Product Manager"],
            "career_goal_summary": "Own a product area end to end",
            "target_timeline": "within 12 months",
            "current_role": "Customer Success Manager",
            "years_experience": 7,
            "highest_education": "BA Economics",
            "work_history": ["CSM at a B2B SaaS", "Support lead"],
            "preferred_locations": ["Dublin"],
            "preferred_work_modes": ["hybrid"],
            "target_industries": ["B2B SaaS", "healthtech"],
            "employment_types": ["full-time"],
            "salary_expectation": "around 70k EUR",
            "action_items": [
                "Write two product teardowns of tools you already use",
                "Shadow the PM on one discovery call a week",
                "Ask your manager for one roadmap-owning project",
            ],
        },
        "notes": (
            "Preferences fully known, skills entirely unknown. Skill priorities must "
            "not be invented out of the job title."
        ),
    },
    {
        "id": "long_confirmed_plan",
        "profile": {
            "target_roles": ["Engineering Manager"],
            "career_goal_summary": "Move from tech lead into people management",
            "target_timeline": "within 12 months",
            "current_role": "Tech Lead",
            "years_experience": 9,
            "highest_education": "BEng",
            "preferred_locations": ["London"],
            "preferred_work_modes": ["hybrid"],
            "target_industries": ["fintech", "climate tech"],
            "employment_types": ["full-time"],
            "strengths": ["mentoring", "architecture reviews", "incident command"],
            "current_skills": ["Java", "Kafka", "system design"],
            "skill_gaps": ["hiring", "performance conversations", "budgeting"],
            "action_items": [
                "Run the next two hiring loops as shadow interviewer",
                "Take over the team's 1:1s for a month with your manager observing",
                "Write the team's on-call rota policy",
                "Read one book on performance conversations and summarize it",
                "Present one architecture decision to the leadership review",
                "Ask two EMs how they handled their first underperformer",
            ],
        },
        "notes": (
            "Six confirmed steps. Preservation and contiguous priority ordering are "
            "hardest to get right here, and a truncated response shows up first."
        ),
    },
    {
        "id": "many_unknowns_minimal_profile",
        "profile": {
            "target_roles": ["Something in data"],
            "target_timeline": "no rush",
            "action_items": [
                "Write down which parts of the current job are worth keeping",
                "Talk to two people who work with data day to day",
                "Pick one tool and learn its basics",
            ],
        },
        "notes": (
            "Almost everything unknown. The artifact must say so rather than "
            "assembling a plausible professional out of nothing — the single most "
            "important honesty case."
        ),
    },
    {
        "id": "no_salary_no_gaps",
        "profile": {
            "target_roles": ["Frontend Engineer"],
            "career_goal_summary": "Wants design-heavy frontend work",
            "target_timeline": "within 6 months",
            "current_role": "Web developer at an agency",
            "years_experience": 5,
            "preferred_locations": ["Lisbon", "Remote"],
            "preferred_work_modes": ["remote"],
            "target_industries": ["design tools"],
            "employment_types": ["full-time"],
            "strengths": ["CSS architecture", "accessibility"],
            "current_skills": ["TypeScript", "React", "CSS"],
            "action_items": [
                "Rebuild one agency project as an accessibility case study",
                "Contribute one fix to a design-system repo",
                "Follow up with the two studios that replied last year",
            ],
        },
        "notes": (
            "Salary and skill gaps both unknown — the two fields the seeded "
            "degradation is most tempted to invent."
        ),
    },
    {
        "id": "contract_only_senior",
        "profile": {
            "target_roles": ["Interim Head of Data"],
            "career_goal_summary": "Wants short engagements, not another permanent role",
            "target_timeline": "next engagement within 8 weeks",
            "current_role": "Head of Data (contract)",
            "years_experience": 14,
            "highest_education": "MSc Computer Science",
            "work_history": ["Head of Data at two scale-ups", "Analytics lead"],
            "preferred_locations": ["Remote", "Amsterdam"],
            "preferred_work_modes": ["remote"],
            "employment_types": ["contract"],
            "salary_expectation": "800 EUR/day",
            "strengths": ["stakeholder management", "data platform strategy"],
            "current_skills": ["SQL", "Snowflake", "dbt", "team leadership"],
            "skill_gaps": ["public speaking"],
            "action_items": [
                "Refresh the one-page interim profile with two outcome stories",
                "Contact three fractional-leadership networks",
                "Set the day-rate floor and hold it",
            ],
        },
        "notes": (
            "Contract-only, day rate rather than salary. Employment type and rate "
            "must be respected rather than normalized to a permanent role."
        ),
    },
]


def build_profile(case: dict[str, Any]) -> XBuddyData:
    """The `XBuddyData` a completed conversation would hold for this case."""
    return XBuddyData(**case["profile"])


def populated_sections(profile: XBuddyData) -> list[str]:
    """Sections with at least one populated field, in canonical order."""
    result = []
    for section, fields in SECTION_FIELDS.items():
        for field_name in fields:
            value = getattr(profile, field_name, None)
            if value is not None and value != [] and value != "":
                result.append(section)
                break
    return result


def as_langsmith_examples() -> tuple[list[dict], list[dict]]:
    """Split CASES into parallel (inputs, outputs) lists.

    `unknowns` is computed with the production `derive_unknowns` rather than written
    by hand, so the reference cannot drift from the behaviour it scores. Everything
    the evaluators read is exported — the PR 4 lesson about a stale remote dataset
    was that a locally-correct reference proves nothing.
    """
    inputs = [
        {
            "case_id": case["id"],
            "profile": case["profile"],
        }
        for case in CASES
    ]
    outputs = []
    for case in CASES:
        profile = build_profile(case)
        outputs.append(
            {
                # Consumed by the evaluators.
                "confirmed_facts": case["profile"],
                "confirmed_action_items": list(profile.action_items),
                "unknowns": derive_unknowns(profile),
                "expected_sections": populated_sections(profile),
                # Diagnostic only.
                "notes": case["notes"],
            }
        )
    return inputs, outputs
