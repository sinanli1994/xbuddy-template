"""Cases for the JobBuddy memory eval.

Each case is a *second* turn: some structured memory already exists, this turn's
extraction is applied to it, and the reply is generated from the resulting state.
That ordering is what makes the eval able to catch a memory regression — a merge
that drops a field shows up as the agent re-asking for it.

Every case is deterministic: the stored data, the extraction result, and the user
message are all fixed, so the only variable is the model's reply.

A field is in one of three states, not two — the distinction the first live run
exposed:

* **complete** — stored and finished; asking again is a memory failure.
* **needs refinement** (`refinement_pending`) — stored, but the section prompt
  prescribes a follow-up before moving on. "hybrid" needs a days-on-site answer;
  a strength needs an example. Asking here is *correct*, so these fields are
  exempt from the re-ask check and count as legitimate progress.
* **missing** (`missing`) — not collected; the next question should target it.

`known` maps to XBuddyData fields already collected. `extraction` is what the
extraction model returns this turn — almost always all-null, meaning "nothing new
was said", which is exactly the situation a broken merge destroys.
"""

from typing import Any

DATASET_NAME = "jobbuddy-memory-regression"

CASES: list[dict[str, Any]] = [
    # ---------------- Career Goal ----------------
    {
        "id": "role_known_timeline_missing",
        "section": "career_goal",
        "known": {"target_roles": ["Senior SRE"], "career_goal_summary": "Wants more autonomy"},
        "extraction": {},
        "user_message": "Yeah, autonomy is the big one for me.",
        "missing": ["target_timeline"],
        "notes": "Role and reason known. Must not re-ask the role; should ask about timing.",
    },
    {
        "id": "timeline_known_role_missing",
        "section": "career_goal",
        "known": {"target_timeline": "within 3 months"},
        "extraction": {},
        "user_message": "Three months feels right, yes.",
        "missing": ["target_roles", "career_goal_summary"],
        "notes": "Timeline known. Must not re-ask when; should move toward the role.",
    },
    {
        "id": "career_goal_complete",
        "section": "career_goal",
        "known": {
            "target_roles": ["Senior SRE"],
            "career_goal_summary": "Wants more autonomy and platform ownership",
            "target_timeline": "within 3 months",
        },
        "extraction": {},
        "user_message": "That all sounds right.",
        "missing": [],
        "notes": "Nothing left in this section. Must not re-ask anything at all.",
    },
    {
        "id": "correction_timeline",
        "section": "career_goal",
        "known": {"target_roles": ["Senior SRE"], "target_timeline": "within 3 months"},
        "extraction": {"target_timeline": "within 6 months"},
        "user_message": "Actually, make that six months — I need more runway.",
        "missing": ["career_goal_summary"],
        "notes": "The one supported overwrite: the new timeline replaces the old.",
    },
    # ---------------- Background ----------------
    {
        "id": "role_and_experience_known",
        "section": "background",
        "known": {"current_role": "QA Analyst", "years_experience": 4},
        "extraction": {},
        "user_message": "Four years, mostly manual testing moving into automation.",
        "missing": ["highest_education", "work_history"],
        "notes": "Must not re-ask the current role or how many years.",
    },
    {
        "id": "background_partial",
        "section": "background",
        "known": {"current_role": "QA Analyst", "highest_education": "BSc Computer Science"},
        "extraction": {},
        "user_message": "Yes, a BSc in Computer Science.",
        "missing": ["years_experience", "work_history"],
        "notes": "Education just confirmed. Next question should target years or history.",
    },
    # ---------------- Job Preferences ----------------
    {
        "id": "location_and_mode_known",
        "section": "job_preferences",
        "known": {"preferred_locations": ["Berlin", "Remote"], "preferred_work_modes": ["remote"]},
        "extraction": {},
        "user_message": "Remote, ideally, though Berlin works if needed.",
        "missing": ["target_industries", "employment_types", "salary_expectation"],
        "notes": "Must not re-ask where or which work mode.",
    },
    {
        "id": "preferences_partial_salary_missing",
        "section": "job_preferences",
        "known": {
            "preferred_locations": ["Berlin"],
            "preferred_work_modes": ["hybrid"],
            "target_industries": ["fintech"],
            "employment_types": ["full-time"],
        },
        "extraction": {},
        "user_message": "Full-time is what I'm after.",
        "missing": ["salary_expectation"],
        "refinement_pending": ["preferred_work_modes"],
        "notes": (
            "Only pay is genuinely missing, but the Job Preferences prompt says "
            "'If they say hybrid, ask how many days on-site they would accept.' "
            "That follow-up is prescribed, so it is neither a re-ask nor a stall."
        ),
    },
    # ---------------- Skill Assessment ----------------
    {
        "id": "skills_known_gaps_missing",
        "section": "skill_assessment",
        "known": {
            "strengths": ["systems debugging"],
            "current_skills": ["Python", "Postgres", "Docker"],
        },
        "extraction": {},
        "user_message": "Python and Postgres are where I'm strongest.",
        "missing": ["skill_gaps"],
        "refinement_pending": ["strengths"],
        "notes": (
            "Must not re-ask which skills they have. Gaps are missing, but the "
            "Skill Assessment prompt says 'ask for an example alongside each "
            "[strength] ... Only then move to gaps' — XBuddyData stores strengths "
            "as bare strings, so 'evidenced' is not representable and the "
            "follow-up must be treated as prescribed refinement."
        ),
    },
    # ---------------- Action Plan ----------------
    {
        "id": "action_plan_with_full_history",
        "section": "action_plan",
        "known": {
            "target_roles": ["Senior SRE"],
            "target_timeline": "within 3 months",
            "current_role": "QA Analyst",
            "years_experience": 4,
            "preferred_locations": ["Berlin"],
            "current_skills": ["Python"],
            "skill_gaps": ["Kubernetes"],
        },
        "extraction": {},
        "user_message": "Where should I start?",
        "missing": ["action_items"],
        "notes": (
            "Every earlier section is filled. The strongest re-ask test: a broken "
            "merge or rendering makes the agent revisit Career Goal or Background."
        ),
    },
]


def as_langsmith_examples() -> tuple[list[dict], list[dict]]:
    """Split CASES into two parallel lists: (inputs, outputs).

    `case_id` is the stable synchronization key — see `sync.plan_sync`.

    The evaluators read exactly three keys out of `reference_outputs`:
    `missing_fields`, `refinement_pending`, and `section`. Those three are what
    must never go stale in the uploaded dataset; a missing `refinement_pending`
    silently disables both exemptions and scores correct agent behaviour as a
    failure, which is what happened in the second live run.

    `known` and `extraction` are exported alongside them even though no evaluator
    reads them, so a stored example fully describes the turn it scores. Reading a
    failure in the LangSmith UI otherwise means cross-referencing this file at the
    revision the run used.
    """
    inputs = [
        {
            "case_id": case["id"],
            "section": case["section"],
            "known": case["known"],
            "extraction": case["extraction"],
            "user_message": case["user_message"],
        }
        for case in CASES
    ]
    outputs = [
        {
            # Consumed by the evaluators.
            "missing_fields": case["missing"],
            "refinement_pending": case.get("refinement_pending", []),
            "section": case["section"],
            # Diagnostic only: makes a stored example self-describing.
            "known": case["known"],
            "known_fields": sorted(case["known"]),
            "extraction": case["extraction"],
            "notes": case["notes"],
        }
        for case in CASES
    ]
    return inputs, outputs
