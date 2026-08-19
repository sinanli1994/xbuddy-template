# Final-Artifact Eval

Answers the question PR 5 exists to answer: is the final document **grounded,
complete, personalized, coherent, actionable, and prioritized** — not merely
generated without an exception.

Each case is a completed conversation's `XBuddyData`. The target runs the real
`synthesize_final_output` and the real `render_final_output`, so a defect propagates
exactly as it would in `implementation_node`. No Supabase involvement.

PR 5's architecture, lifecycle, persistence contract, and PR 6 deferrals are recorded
in [`docs/pr5-final-output.md`](../../docs/pr5-final-output.md). This file covers the
eval only. The structured-first design is what makes these checks possible at all:
"every strength traces to a collected fact" is an assertion over a list of strings,
where over free-form Markdown it would be a regex.

## Two families of evaluator, and why the split matters

| Family | Count | Decides |
|---|---|---|
| **Deterministic** | 6 | Anything derivable from data: grounding, preservation, honesty, completeness, ordering, render stability |
| **LLM judge** | 4 | Only what genuinely requires reading: coherence, relevance, actionability, prioritization quality |

A judge is never asked something a function can answer. That is not just cost
discipline — it is what makes the central finding legible: **the seeded arm is
expected to keep coherence high while grounding collapses.** A fluent, confident,
fabricated document is the failure mode a user cannot detect themselves, and it only
shows up as a *gap between the two families*. Blend them into one "quality" score and
the gap disappears.

## Dataset — 10 cases

| Case | What it tests |
|---|---|
| `fully_populated_sre` | nothing missing; every section must be represented |
| `partial_no_preferences` | all five preference fields unknown |
| `career_transition_teacher` | must bridge from teaching without inventing industry experience |
| `aggressive_timeline` | 4 weeks; timeframes must respect it |
| `strong_skills_thin_preferences` | deep skills, no preferences |
| `clear_preferences_thin_skills` | skills entirely unknown — must not infer them from the job title |
| `long_confirmed_plan` | 6 confirmed steps; preservation and ordering are hardest here |
| `many_unknowns_minimal_profile` | 13 unknowns — the strongest honesty case |
| `no_salary_no_gaps` | the two fields the seeded bug most wants to invent |
| `contract_only_senior` | day rate, contract-only; must not normalize to a permanent role |

Coverage spans 0 → 13 unknowns, 2 → 5 populated sections, 3 → 6 confirmed steps.

`unknowns` in the reference is **computed by the production `derive_unknowns`**, never
hand-written, so the expectation cannot drift from the behaviour it scores.

## Deterministic evaluators

| Evaluator | Passes when |
|---|---|
| `grounding_no_invention` | no unsupported money figure, seniority claim, or claimed capability |
| `action_items_preserved` | every confirmed step survives exactly, in order, nothing added |
| `unknowns_honest` | missing fields declared, populated fields not, and no unknown contradicted elsewhere |
| `artifact_complete` | every populated section is materially represented |
| `priority_valid` | priorities are contiguous 1..n *and* rendered in that order |
| `render_deterministic` | re-rendering the same structured output is byte-identical |

### The grounding boundary

The hardest judgement here, and the one most likely to be got wrong in either
direction. Fields are split explicitly:

- **Fact-asserting** — `headline`, `positioning_summary`, `strengths_to_leverage`,
  `risks_or_constraints`, and each `action_items[].rationale`. These describe the
  person, so user-specific claims must trace to `XBuddyData`.
- **Recommendation** — `skill_priorities`, `search_targets`, and each
  `action_items[].step` / `timeframe`. These propose what to *do*. Naming a
  technology the user does not yet know is the product working.

So "learn Kubernetes" in `skill_priorities` is never flagged, while "Kubernetes
operations at scale" in `strengths_to_leverage` is. Two tests pin both directions,
including one where a *recommendation* mentions a duration ("the depth most teams
expect after two years") that would be misread as a seniority claim if the boundary
ever moved.

Grounding is three concrete sub-checks rather than one similarity score, because each
maps to a way the document can lie about someone: an unsupported **money** figure, a
**seniority** number that disagrees with `years_experience`, and a **claimed
capability** with no support anywhere in the profile. Capability support is token
overlap, so the model may rephrase without being punished.

## LLM judges

Bounded 1–5 with a written description of every level, normalized to 0–1 for
aggregation with the raw score and the judge's reasoning kept in the comment. Vague
"is this good?" scoring drifts between runs; naming what a 2 looks like versus a 4
does not.

The judge is told explicitly **not** to compensate for other checks: a fluent
invention should still score high on coherence, because grounding is not its job.

A judge failure scores `None`, never `0` — a `0` would be indistinguishable from a
genuinely bad document and would quietly corrupt the aggregate.

## Seeded degradation

`--seed-grounding-bug` changes real synthesis behaviour. It replaces
`build_synthesis_context` so that the **UNKNOWNS block is withheld** and the model is
instead told:

> *Where a detail was not captured above, infer the most likely professional value and
> state it plainly as part of their profile — a market-rate salary target …, the cloud
> and container tooling someone at this level would be expected to have, and a typical
> amount of experience. Present the finished picture.*

This is not contrived. It is the change someone makes when a stakeholder says the
artifact "looks unfinished". Nothing about the evaluators or their scores is touched:

```
UNKNOWNS withheld + "infer the likely value"
   → the model states a salary that was never discussed
   → grounding_no_invention and unknowns_honest fall
   → coherence stays high, because the document reads well
```

Only cases with a real gap can degrade — `fully_populated_sre` has nothing to invent,
which is why the offline proof asserts on the subset with gaps rather than all ten.

## Measured results

`finalout-clean-2ad07af4` and `finalout-seeded-grounding-3f10d273`, 10 cases each,
10 root runs, 0 errored, 0 synthesis failures.

| Evaluator | Clean | Seeded | Delta |
|---|---|---|---|
| `grounding_no_invention` | **1.000** | **0.700** | **−0.300** (3/10 fail) |
| `unknowns_honest` | **1.000** | **0.700** | **−0.300** (3/10 fail) |
| `action_items_preserved` | 1.000 | 1.000 | 0.000 |
| `artifact_complete` | 1.000 | 1.000 | 0.000 |
| `priority_valid` | 1.000 | 1.000 | 0.000 |
| `render_deterministic` | 1.000 | 1.000 | 0.000 |
| `coherence` | 0.900 | 0.800 | −0.100 |
| `relevance_to_user` | 0.694 | 0.700 | +0.006 |
| `actionability` | 0.875 | 0.950 | **+0.075** |
| `prioritization_quality` | 0.700 | 0.925 | **+0.225** |

The seeded failures are concrete invented salaries on cases where salary was never
collected — `$70,000–$90,000`, `$110,000–$130,000`, `$120,000–$150,000` — each
contradicting the same document's own "never discussed" line.

**This is the finding.** Grounding and honesty each fall 0.300 while *actionability
rises 0.075 and prioritization rises 0.225*. The fabricated documents read as better
career advice. Had the two families been blended into one "quality" score, the seeded
arm would have looked roughly unchanged, or better. Fluent hallucination is only
visible as the gap between them.

## Dataset sync

Reuses `evals/memory/sync.py` — PR 4's `case_id`-keyed planner, loaded by file path
rather than by adding that directory to `sys.path`, because `evals/memory/` also
contains a `dataset.py` and an `evaluators.py` that shadow this eval's own modules.
(They did, on the first run.)

That module exists because PR 4 ran a paid experiment against a stale remote dataset
and got numbers that looked plausible and were not. Every run here reconciles first.

```bash
uv run python evals/final_output/run_final_output_eval.py --dry-run-sync   # read-only report
uv run python evals/final_output/run_final_output_eval.py --sync-dataset   # apply
```

## Run

```bash
# free
uv run python evals/final_output/run_final_output_eval.py --check-env
uv run python evals/final_output/run_final_output_eval.py --dry-run-sync
uv run python evals/final_output/run_final_output_eval.py --sync-dataset

# paid
uv run python evals/final_output/run_final_output_eval.py
uv run python evals/final_output/run_final_output_eval.py --seed-grounding-bug

# paid, cheaper: deterministic evaluators only
uv run python evals/final_output/run_final_output_eval.py --no-judges
```

Each paid arm is 10 synthesis calls plus 40 judge calls. `--no-judges` drops it to 10.

Never publish a clean run without its seeded counterpart: an eval that has never been
shown to fail is not evidence of anything.

## Token budget

`max_tokens=3000` caps the **model's output**, which is the `FinalOutputDraft` JSON —
*not* the rendered document. Two Stage 2 design choices keep that output small: the
model never echoes action-item step text (it returns annotations keyed by
`step_number`), and it never authors `unknowns` (they are derived).

Measured offline across all 10 cases with a deliberately verbose draft — long
rationales, timeframes on every step, six entries in every list:

| Quantity | Worst case |
|---|---|
| Synthesis context sent | ~236 tokens |
| **Draft JSON returned (what `max_tokens` caps)** | **~832 tokens** |
| Rendered Markdown | ~918 tokens |
| Headroom against 3000 | **~2168 tokens (3.6×)** |

Confirmed against both paid arms: structured output 1448–2250 characters (~360–560
tokens), rendered Markdown 1358–2075 characters, **zero truncation flags and zero
parse failures across 20 runs**. The largest was `long_confirmed_plan` at ~518 tokens,
as predicted.

**Conclusion: no evidence that 3000 is insufficient, so it is unchanged.**

## Offline coverage

`tests/agents/xbuddy/test_final_output_eval.py` — 61 tests, no network, no keys.
Dataset integrity, both pass and fail paths for all six deterministic evaluators, the
grounding boundary in both directions, judge prompt construction and failure handling,
survival through the PR 4 sync planner, and an offline simulation of the seeded
degradation proving both grounding metrics catch it across every case with a gap.

Verified by mutation: 19 defects introduced one at a time, all 19 caught. Two initially
slipped through — a recommendation-field guard that used technology names (which
contain no money or duration pattern, so the mutation was nearly inert) and the
priority-contiguity branch (unreachable through a valid `FinalOutput`, since its
validator forbids it). Both now have targeted tests.

Nothing in this directory is collected by pytest — no filename matches `test_*.py` and
it sits outside `tests/`.

## Files

| File | Purpose |
|---|---|
| `dataset.py` | 10 cases; reference facts, plan, derived unknowns, expected sections |
| `evaluators.py` | The six deterministic evaluators and the grounding boundary |
| `judges.py` | Four rubric-based judges, bounded and normalized |
| `run_experiment.py` | Target, seeded degradation, sync, token metrics, LangSmith wiring |
| `README.md` | This file |

## Reporting into the PR

1. Both experiment names, and all ten evaluator scores per arm.
2. The clean → seeded delta per metric, with the grounding drop called out.
3. That coherence stayed high while grounding fell — the fluent-hallucination proof.
4. Per-case failures for anything below 1.00 on the clean arm.
5. The token-budget table and the `max_tokens` decision.
