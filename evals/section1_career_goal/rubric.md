# Section 1 (Career Goal) — Human Comparison Rubric

The deterministic evaluators in `run_experiment.py` measure **rule compliance**:
did the reply ask one question, avoid placeholders, stay in section, and aim at
an unfilled field. They cannot tell whether the turn was actually *good*.

Score each of the 10 cases for **both** variants, 1-5 per dimension.

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| **Answerability** | The user has to guess what's being asked, or the question is too abstract to answer in one turn | Answerable, but they may need a beat to work out what shape of answer is wanted | Immediately answerable in one turn, with no clarification needed |
| **Progress** | No movement toward `target_roles`, `career_goal_summary`, or `target_timeline` | Loosely related to an unfilled field | Clearly advances a specific unfilled field |
| **Section discipline** | Drifts into Background / Preferences / Skills / Action Plan | Briefly acknowledges off-section material but lingers on it | Acknowledges and parks off-section material, then returns to Career Goal |
| **Non-leading guidance** | Examples put words in the user's mouth, or narrow the field to only what was listed | Examples are reasonable but slightly steer the answer | Orients the user without constraining them; visibly non-exhaustive |
| **Conciseness** | Padding, restates the whole framework, motivational filler | A little longer than needed | Every sentence earns its place |

## What to look for

**Answerability and Non-leading guidance are the load-bearing pair.** Variant B
(anchored) is expected to win Answerability — that is the whole point of adding
examples. The reason this rubric exists is to check what B *pays* for that:

- Does B's example list quietly become the menu, so a user who wanted something
  else picks from it anyway?
- On `career_changer` and `dont_know`, does B suggest a destination the user
  never mentioned? That is a Non-leading score of 1 and a violation of the
  BASE_RULES "never invent facts" rule, and no deterministic evaluator sees it.
- On `over_specific`, does either variant re-ask something already known?

If B wins Answerability by 1+ point while losing Non-leading guidance by 1+
point, the anchored approach is trading user comfort for steered answers and
should not ship as-is — narrow the examples or make them more obviously
open-ended.

## Cases where the two approaches should diverge most

| Case | Why it discriminates |
|---|---|
| `dont_know` | B's examples are either a lifeline or an imposition |
| `career_changer` | Strongest test of whether examples invent a destination |
| `field_not_role` | Both must convert a field into a title; B may shortcut it |
| `asks_back` | The user *invites* leading; a good reply still declines to decide for them |
| `volunteers_later_section` | Pure section-discipline check, independent of the variant axis |

## Recording

Fill in `scores_template.csv` (20 rows: 10 cases × 2 variants). Then report:

1. Per-dimension mean for A and for B.
2. The per-dimension winner, and the margin.
3. Any case where the variants disagreed sharply, with a one-line reason.
4. A recommendation: ship A, ship B, or ship B with narrowed examples.
