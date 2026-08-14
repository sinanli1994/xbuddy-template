# Memory Regression Eval

Detects the memory failure a JobBuddy user would actually notice: **the agent
re-asks something they already answered.**

That single symptom is downstream of every memory defect in PR 4 — extraction
dropping a field, a merge clobbering a stored value, the `KNOWN SO FAR` block not
being rebuilt, or a restart losing `user_data`. So it is the thing worth
measuring, and it is measurable deterministically.

## What each case actually exercises

Every case is a **second** turn, run through production code:

```
case.known ──► XBuddyData
                   │
                   ▼   real merge_extraction()
        case.extraction ──► after
                   │
                   ▼   real build_system_prompt() + render_known_data()
              system prompt (contains KNOWN SO FAR)
                   │
                   ▼   the model
                 reply
```

That ordering is the whole design. A memory bug in the merge propagates into the
prompt and therefore into the reply, exactly as it would in the graph — so the
evaluators observe *behaviour*, not a mocked score.

No Supabase involvement: this is the in-memory path only.

## A field is in one of three states

The first live run exposed a flaw in the original two-state model. "Known" and
"missing" is not enough, because some stored values carry a **prompt-mandated
follow-up**:

| State | Meaning | Asking about it is |
|---|---|---|
| **complete** | stored and finished | a memory failure |
| **needs refinement** (`refinement_pending`) | stored, but the section prompt prescribes a follow-up before moving on | **correct** — and counts as progress |
| **missing** (`missing`) | never collected | correct — the next question should target it |

Two cases need the middle state, and both were scored as failures in the first run
while the agent was doing exactly what its PR 2 prompt says:

- **Job Preferences** — *"If they say hybrid, ask how many days on-site they would
  accept."* With `preferred_work_modes: ["hybrid"]` stored, "how many days on-site?"
  is the prescribed next move, not a re-ask.
- **Skill Assessment** — *"ask for an example alongside each [strength] ... Only then
  move to gaps."* `XBuddyData` stores strengths as bare strings, so "evidenced" is not
  representable in the schema; the follow-up has to be modelled as refinement.

So `refinement_pending` fields are exempt from `no_redundant_question` and count as
legitimate progress for `advances_unfilled_field`. The exemption is **declared per
case, never applied globally** — a work-mode question in a case without the
annotation still scores as a re-ask, and `test_no_redundant_question_exemption_is_per_case_not_global`
pins that. Two further tests derive the annotations from the prompts themselves, so a
new hybrid or strengths case inherits the requirement instead of quietly under-scoring.

## Dataset — 10 cases, all five sections

| # | Case | Section | Tests |
|---|---|---|---|
| 1 | `role_known_timeline_missing` | Career Goal | role known; must ask timing |
| 2 | `timeline_known_role_missing` | Career Goal | timeline known; must ask role |
| 3 | `career_goal_complete` | Career Goal | nothing left — must re-ask nothing |
| 4 | `correction_timeline` | Career Goal | **the supported overwrite**: 3 → 6 months |
| 5 | `role_and_experience_known` | Background | current role + years known |
| 6 | `background_partial` | Background | education known; ask years/history |
| 7 | `location_and_mode_known` | Job Preferences | location + work mode known |
| 8 | `preferences_partial_salary_missing` | Job Preferences | only pay left; **hybrid needs refinement** |
| 9 | `skills_known_gaps_missing` | Skill Assessment | skills known; ask gaps; **strengths need an example** |
| 10 | `action_plan_with_full_history` | Action Plan | **every earlier section filled** — the strongest re-ask test |

Case 10 is the one that catches the widest class of bug: with seven fields stored
across four sections, any failure to carry memory forward shows up as the agent
revisiting Career Goal or Background instead of building the plan.

## Evaluators

| Evaluator | Passes when |
|---|---|
| `known_block_complete` | every populated field appears in the deterministic `KNOWN SO FAR` rendering, label and value |
| `extraction_no_clobber` | no populated value became `None` / `[]` / `""` through the merge (a *new* value is fine — that is a correction) |
| `no_redundant_question` | the reply asks for nothing that pre-turn memory already held, `refinement_pending` aside |
| `advances_unfilled_field` | when work remains, the reply targets a missing field **or** a prescribed refinement (auto-passes when the section is complete, or for Action Plan) |

### Why `advances_unfilled_field` exempts Action Plan

The other four sections *collect* values from the user, so "did the reply ask about
something still open?" is a fair question. Action Plan does the opposite — its prompt
says **"Propose a first draft yourself... Do not ask the user to invent the plan from
nothing."** A reply that delivers five concrete steps and asks which feel realistic is
the *ideal* behaviour, yet it asks about no unfilled field, so the check scored the
best possible reply as a stall. That is measuring the wrong thing, and no amount of
term-tuning fixes it — the check is structurally inapplicable to a section that
proposes rather than asks.

The exemption is scoped to that one section via `PROPOSE_SECTIONS`, and
`test_advances_unfilled_field_exemption_does_not_leak_to_other_sections` proves the
four collection sections keep the full-strength check. Action Plan is still covered by
the other three evaluators — including `no_redundant_question`, which is the strongest
test in the whole set for case 10 (seven fields stored across four sections).

### Three details that make `asks_about` trustworthy rather than decorative

- **Phrase-level terms.** A bare `"role"` matched *"When would you like to be in the
  new role?"* — a timeline question — and flagged it as re-asking the role. Terms are
  phrasings that actually solicit a value.
- **Echoed values are acknowledgement.** *"For a Senior SRE role, when do you want to
  move?"* mentions the role but asks about timing, so it is not a re-ask.
- **Two-tier scoping.** Tier 1 is the question sentence itself. Tier 2 reads the one
  preceding sentence, but **only when the question names no field at all** — a
  referential *"Let's capture your highest completed education next. What is it?"*,
  which question-only scoping scored as advancing nothing.

  The gate on tier 2 is load-bearing. An unconditional two-sentence window flagged
  *"Great, we'll aim for that timeline. What role are you looking to move into next?"*
  as a timeline re-ask — a live case regressed on exactly that. Because the question
  already names `target_roles`, it is not bare, so the lookback never runs. The window
  never extends beyond one sentence; scanning the whole reply brings back every false
  positive the phrase-level terms were introduced to remove.

`no_redundant_question` deliberately scores against **pre-turn** memory. Scoring
post-merge data would let a clobbering bug hide: the field would look unknown, so
re-asking it would look legitimate.

## Seeded-bug proof

A passing clean run proves nothing on its own. `--seed-bug` introduces a real
one-line defect into the real module:

```python
extraction_module._is_no_op = lambda value: False
```

`merge_extraction` skips fields where `_is_no_op(value)` is true. With that
guard disabled it applies **every** extracted field including the nulls — so an
all-null extraction, which means "nothing new was said this turn" and is the
common case, wipes the section's stored values.

This is precisely the regression a contributor would introduce by deleting the
no-op check. Nothing about the evaluators or their scores is touched; the
mutation changes behaviour and the evaluators detect the consequences:

```
merge_extraction loses fields
   → render_known_data omits them from KNOWN SO FAR
   → the model is never told they are known
   → it re-asks them
```

## Canonical results

These are **the** numbers for PR 4. Both arms ran against the same synchronized
dataset, 10 cases each, 10 root runs and 0 errored runs per arm:

| Evaluator | Clean `memory-clean-623d4334` | Seeded `memory-seeded-bug-e66d4674` |
|---|---|---|
| `known_block_complete` | **1.00** | 1.00 (see note) |
| `extraction_no_clobber` | **1.00** | **0.10** — 9/10 detect |
| `no_redundant_question` | **1.00** | **0.40** — 6/10 detect |
| `advances_unfilled_field` | **1.00** | 0.50 (incidental) |

The clean arm is 40/40 feedback rows passing. Quote these two experiment names
alongside the figures — a score without the run it came from cannot be checked.

Note on earlier runs: two live pairs preceded these, and their scores are
deliberately not recorded here. Both exposed defects in the *harness* rather than
the agent — an evaluator contract that mis-scored prompt-prescribed refinement as
redundant questioning, and a dataset-sync bug that left `refinement_pending` absent
from the uploaded examples (§Dataset synchronization). Both were fixed before the
canonical run, which makes the earlier numbers measurements of the old harness and
not of JobBuddy's memory.

Note on `known_block_complete`: it scores the *post-merge* data, so when the bug
wipes a field it is consistently absent from both the data and the rendering —
self-consistent, and therefore not flagged. That is not a gap in the eval; it is
why `no_redundant_question` scores against pre-turn memory instead. The pair is
what makes the detection sound.

Note on the seeded `extraction_no_clobber` floor: **0.10, not 0.00, is correct.**
The merge only applies fields the *active section's* extract model declares, so the
bug can only wipe values belonging to the current section. Case 10
(`action_plan_with_full_history`) stores seven fields across four **earlier** sections
and none for Action Plan, whose extract model declares only `action_items` — there is
nothing in scope to clobber, so that case legitimately passes even with the bug
active. Chasing 0.00 would mean weakening section-scoped extraction, which is a
design property worth keeping, not a defect to score against.

`advances_unfilled_field` dropping under the bug is a side effect rather than a
detection claim: once memory is wiped the agent re-asks known fields instead of
targeting the open ones. It is recorded for completeness, not relied upon.

**These figures assume a synchronized dataset.** A run scored against a stale replica
produces numbers that look plausible and are not — with `refinement_pending` missing
upstream, the clean arm reads 0.80 on both `no_redundant_question` and
`advances_unfilled_field` while the agent is behaving correctly. Always run
`--dry-run-sync` before spending anything. See §Dataset synchronization.

## Dataset synchronization

`dataset.py` is the source of truth. The LangSmith dataset is a **replica**, and every
run reconciles the replica against the source before scoring anything.

### The drift bug this exists to prevent

`ensure_dataset` used to return early whenever the dataset already existed:

```python
if client.has_dataset(dataset_name=DATASET_NAME):
    return client.read_dataset(dataset_name=DATASET_NAME)   # ← wrote nothing, ever
```

So the examples were uploaded once, when the dataset was first created, and **every
later `dataset.py` edit was silently inert.** Nothing failed. Nothing warned.

The cost was a full paid run's worth of untrustworthy scores. `refinement_pending` was
added locally, covered by offline tests, and reviewed — but never reached the cloud, so
the evaluators received `reference_outputs` with no such key and both exemptions
silently did nothing. Two cases were scored as failures for behaviour that is *exactly*
what their section prompts mandate:

| Case | Reply | Scored | Actually |
|---|---|---|---|
| `preferences_partial_salary_missing` | *"You mentioned hybrid. How many days on-site would you be comfortable with?"* | `no_redundant_question` **0** — "re-asked `preferred_work_modes`" | the follow-up Job Preferences explicitly prescribes |
| `skills_known_gaps_missing` | *"Can you share an example of a project where you used these skills?"* | `no_redundant_question` **0** — "re-asked `strengths`" | the evidence Skill Assessment explicitly requires |

Clean `no_redundant_question` and `advances_unfilled_field` both read 0.80 instead of
1.00. The give-away was in the feedback text itself: `targets none of
['salary_expectation']` listed only the missing field, with no refinement candidate
beside it — proof the key was absent rather than merely unsatisfied.

Two lessons are baked into the current design. First, **a local-only test cannot prove
a remote contract**: every offline test passed throughout, because they all checked the
export function and none checked what was stored. Second, the failure mode was
*silence* — a stale replica scores confidently and wrongly, which is worse than an
error, so reconciliation now happens unconditionally and prints what it did.

### `refinement_pending` is part of the evaluator contract

The evaluators read exactly three keys out of `reference_outputs`:

| Key | Read by | Effect if missing |
|---|---|---|
| `missing_fields` | `advances_unfilled_field` | nothing to advance; auto-pass |
| `refinement_pending` | `no_redundant_question`, `advances_unfilled_field` | **both exemptions silently disabled** |
| `section` | `advances_unfilled_field` | Action Plan exemption silently disabled |

`refinement_pending` is the dangerous one, because absence and "no follow-up needed"
are indistinguishable to a naive reader — which is why every case exports the key even
when its value is `[]`, and why a parametrized test asserts all three reach the upload
payload.

`known` and `extraction` are exported alongside them. No evaluator reads those; they
make a stored example self-describing, so reading a failure in the LangSmith UI does
not require cross-referencing `dataset.py` at the revision the run used.

### Identity: `inputs.case_id`

Synchronization is keyed on `inputs["case_id"]` — stable across runs, human-readable,
and already unique (`test_dataset_size_and_section_coverage`). LangSmith's example
UUIDs are deliberately *not* used: they are assigned server-side and nothing local
remembers them, so they cannot survive a fresh checkout. List position is not used
either, since `list_examples` guarantees no ordering.

### Create / update / unchanged

| Situation | Action |
|---|---|
| `case_id` absent from the dataset | **create** |
| present, and any field differs | **update in place** via `update_example`, reusing the stored id |
| present and identical | **untouched** — not rewritten |

Comparison is a JSON round-trip normalization, so a local tuple does not diff against a
stored list on every run. Absent and `[]` are treated as **different** states, which is
precisely the distinction the stale examples turned on. A key present remotely but not
locally also counts as a change, because `update_example` replaces the mapping rather
than merging into it.

Unchanged examples are never rewritten, so LangSmith's example version history stays
meaningful — "what did this run score against?" remains answerable.

### Nothing is ever deleted

Three situations are **reported and left alone**, never acted on:

| Situation | Meaning | Handling |
|---|---|---|
| **Orphan** — stored `case_id` unknown locally | usually a renamed or removed case | reported; the row stays |
| **Duplicate** `case_id`s stored | shouldn't happen; would make scoring ambiguous | first wins, count reported |
| Example with **no `case_id`** | someone else's example in the same dataset | ignored entirely |

Deleting would make a rename indistinguishable from data loss, and the whole point of
this stage is that dataset changes should be visible rather than silent. Removing a
stale row is a deliberate manual act.

### Commands

```bash
uv run python evals/memory/run_experiment.py --dry-run-sync   # read-only report
uv run python evals/memory/run_experiment.py --sync-dataset   # apply, then exit
```

Both are **free** — they touch only the dataset API and never call a model. `--dry-run-sync`
computes the full plan and writes nothing; `--sync-dataset` applies it. Neither is
strictly required, because a clean or `--seed-bug` run synchronizes first anyway, but
running the dry report before spending money tells you exactly what the run is about to
score against.

A typical report looks like:

```
Dataset jobbuddy-memory-regression: create 0 / update 10 / unchanged 0
  ~ preferences_partial_salary_missing: outputs.refinement_pending, outputs.notes
  ! stored but unknown locally (left alone): ['renamed_away']
```

### Offline proof

`tests/agents/xbuddy/test_dataset_sync.py` (21 tests) covers this without a network,
keys, or a model, using a fake client: a stale example is updated, a current one is not
rewritten, a missing one is created, `refinement_pending` survives the whole
export → plan → payload path, identity survives reordering and regenerated UUIDs, and
orphans / duplicates / foreign examples are reported without being touched.

Planning is a pure function in `sync.py` — no `langsmith` import, no I/O — which is what
makes that possible. The create-vs-sync branch lives there rather than in
`run_experiment.py` on purpose: that branch *is* the regression site, and importing the
runner pulls `.env` into `os.environ`, so leaving it there would have kept the one thing
that broke untestable.

## Setup

Repository-root `.env`; the runner calls `load_dotenv(find_dotenv())` itself.

```bash
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=jobbuddy-pr4
```

`LANGSMITH_*` is this project's convention; the legacy `LANGCHAIN_*` aliases still
resolve because `langsmith.utils.get_env_var` checks both namespaces. As with the
PR 2 eval, `LANGSMITH_PROJECT` does not control where results land — `evaluate()`
creates its own project per run, named `<experiment_prefix>-<hex8>`.

## Run

Run these in order. The first three are free; only the last two spend anything.

```bash
# 1. free — resolves keys and tracing, calls nothing
uv run python evals/memory/run_experiment.py --check-env

# 2. free — read-only: what would the dataset sync change?
uv run python evals/memory/run_experiment.py --dry-run-sync

# 3. free — apply it, so both paid arms score against the same current dataset
uv run python evals/memory/run_experiment.py --sync-dataset

# 4. paid — clean run, expect 1.00 x4
uv run python evals/memory/run_experiment.py

# 5. paid — seeded run, MUST fail extraction_no_clobber and no_redundant_question
uv run python evals/memory/run_experiment.py --seed-bug
```

Steps 2 and 3 are the Stage 7D additions, and skipping them is what produced a paid run
of untrustworthy scores (§Dataset synchronization). Step 3 is not strictly required —
steps 4 and 5 synchronize first regardless — but doing it explicitly means the dry report
you read is the state the paid runs actually used, and a `create N / update N` line in
the middle of a paid run is a surprise worth not having.

Never publish a clean run without its `--seed-bug` counterpart: a passing eval that has
never been shown to fail is not evidence of anything.

Experiments appear as `memory-clean-<hex8>` and `memory-seeded-bug-<hex8>` over the
shared dataset `jobbuddy-memory-regression`, so LangSmith's comparison view lines
them up per case.

## Offline coverage

The evaluators are pure functions, so they are unit-tested in the ordinary suite —
`tests/agents/xbuddy/test_memory_evaluators.py`, 41 tests, no network and no keys.
It covers each evaluator's pass and fail paths, both `asks_about` tiers and the gate
between them, the per-case scoping of the refinement exemption, the dataset's
integrity against `XBuddyData` and the extract schemas, the fields
`as_langsmith_examples` must export for the exemptions to apply at all, and **the
seeded bug itself**: the mutation is applied to the real `extraction` module and the
resulting data is scored, so the detection claim above is verified without spending
anything.

Synchronization has its own 21 tests in `tests/agents/xbuddy/test_dataset_sync.py`
(§Dataset synchronization → Offline proof), for **62 offline tests** across the two
files.

Every one of these was verified by reverting the change it guards and confirming the
suite goes red — including the two dataset tests, which were added precisely because
deleting the `refinement_pending` annotations initially broke nothing, and the sync
tests, where reinstating the original early return turns three of them red.

Nothing in this directory is collected by pytest — no filename matches `test_*.py`
and it sits outside `tests/`.

## Files

| File | Purpose |
|---|---|
| `dataset.py` | 10 deterministic cases with `known` / `extraction` / `missing` / `refinement_pending`, and the `as_langsmith_examples` export |
| `evaluators.py` | The four evaluators plus the shared `asks_about` heuristic |
| `sync.py` | Pure `case_id`-keyed sync planner, plus `sync_dataset` / `apply_plan` |
| `run_experiment.py` | Target, `--check-env`, `--dry-run-sync`, `--sync-dataset`, `--seed-bug`, LangSmith wiring |
| `README.md` | This file |

## Reporting into the PR

Quote §Canonical results — `memory-clean-623d4334` and `memory-seeded-bug-e66d4674` —
and for each arm include:

1. All four evaluator scores, named with the experiment they came from.
2. The seeded arm's `extraction_no_clobber` and `no_redundant_question` drop, which is
   the detection proof; a clean run alone demonstrates nothing.
3. Root-run and errored-run counts, so the denominator is auditable.
4. A shared LangSmith trace URL.
