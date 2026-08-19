# PR 5 — Final Output

JobBuddy could hold a five-section conversation and remember it (PR 4), but the
conversation ended without producing anything. PR 5 turns the collected memory into a
durable job search strategy.

`implementation_node` was a documented pass-through returning `{}`; it now synthesizes,
renders, persists, and announces. Nothing else in the graph changed shape.

## The quality bar

Seven dimensions, and what actually holds each one:

| # | Dimension | Held by |
|---|---|---|
| 1 | **Grounded** | the FACTS / UNKNOWNS prompt boundary + `grounding_no_invention` |
| 2 | **Relevant** | the synthesis prompt + the `relevance_to_user` judge |
| 3 | **Coherent** | fixed section order (structural) + the `coherence` judge |
| 4 | **Actionable** | required `step` and `rationale` + the `actionability` judge |
| 5 | **Prioritized** | a **model invariant** — `FinalOutput` rejects anything but 1..n |
| 6 | **Honest about gaps** | `unknowns` derived from state, never authored; heading always rendered |
| 7 | **Complete** | every non-empty field reaches the document + `artifact_complete` |

Dimensions 5 and 6 are invariants rather than aspirations. That is deliberate: it frees
the eval to spend its budget on grounding instead of arithmetic.

## Architecture: structured-first, then deterministic rendering

```
XBuddyData ──► build_synthesis_context ──► one tagged model call
                 (FACTS / CONFIRMED ACTION PLAN / UNKNOWNS)
                                              │
                                    FinalOutputDraft
                                              │
                    assemble_final_output ────┤ + confirmed steps (verbatim)
                                              │ + derive_unknowns (computed)
                                          FinalOutput
                                              │
                              render_final_output ──► Markdown
```

Markdown, not Tiptap: the editor's `convertToTiptapFormat` already parses Markdown, and
its own saves write Markdown into both content columns.

**Why structured-first rather than asking for Markdown directly.** The grounding checks
that matter only work against fields. "Every strength traces to a collected fact" is an
assertion over a list of strings; over prose it is a regex. The cost is prose bounded by
the renderer's template — a structured report rather than an essay, which for a career
roadmap is the right trade. Third application of a pattern PR 2 and PR 4 already used
(`render_known_data`, `render_section_content`).

### Two things the model is never allowed to author

- **`action_items`** — `ActionAnnotation` has no `step` field, so rewording a confirmed
  step is *unrepresentable*, not merely discouraged. The model returns annotations keyed
  by `step_number`; `assemble_final_output` attaches them to source strings and sets
  `priority` from source order. `XBuddyData.action_items` order already means "what to do
  first" (the Action Plan prompt says so), so index+1 is derived, not invented.
- **`unknowns`** — computed by `derive_unknowns` over all 16 `XBuddyData` fields. A model
  asked which facts are missing can both overlook one and invent one; state knows exactly.
  `[]` counts as missing, carrying PR 4's `[]`-as-no-op limitation forward rather than
  inventing an explicit-none semantic.

## Invalidation and regeneration lifecycle

`final_output is None` is the **entire** representation of "no valid artifact". No stale
flag, no version counter in graph state.

| Event | Result |
|---|---|
| All five sections DONE, no artifact | synthesize once; append one readiness message |
| Artifact exists | `{}` — no model call, no message |
| **Real source change** (`extraction_changed`) | clear `final_output`; demote the section DONE → IN_PROGRESS; set `should_generate_final_output=False`; mark the durable row **stale** |
| Modify *intent* with no extracted change | artifact preserved — regenerating would buy nothing |
| Reconfirmed → all five DONE again | regenerate once; a second readiness message is intentional |
| Correction **and** confirmation in one turn | promotion wins; regenerates that turn |

The trigger is `merge_extraction` + `extraction_changed` — a value comparison, never a
model opinion. `agent_output.should_save_content` is deliberately **not** consulted:
`DECISION_RULES` never defines it, so the model emits it with no rule to apply.

## Durable persistence and user-edit protection

New table `final-outputs` (hyphenated because the existing frontend already reads and
writes exactly that name). Every backend-only column is defaulted or nullable, so the
frontend's six-column upsert still works — and because `ON CONFLICT DO UPDATE` assigns
only the supplied columns, a user's save changes `content` and leaves
`generated_content_hash` alone. **That divergence is the edit signal.**

`content_fingerprint` = SHA-256 of canonical Markdown. `updated_at` cannot answer "was
this edited?" — the agent's own write moves it — and a model would make a data-integrity
decision non-reproducible.

| Existing row | Hash vs content | Action |
|---|---|---|
| none | — | insert, `status=current` |
| present | recorded hash == new artifact's | no write (idempotent) |
| present | matches stored content | overwrite, `status=current`, hash updated |
| present | **differs** | **refuse**, preserve the row, report non-fatally |
| present | hash NULL | refuse — conservative; the frontend never writes that column |

**Preserving edits beats automatic synchronization.** The graph keeps the new artifact
either way; only the durable row lags. A refusal is not retried (it would refuse every
turn); a genuine failure is, via one scalar `final_output_pending` ∈ {`write`,`stale`,
`None`}. That is *not* folded into PR 4's `persistence_pending`, whose retry loop looks
each entry up in `section_states` and silently drops anything it cannot find — a
`"final_output"` token there would read as already-succeeded.

Invalidation marks the row stale and **never deletes**; a stale-marking failure never
rolls back the graph-side invalidation, because keeping a document the agent knows is
wrong is the worse outcome.

## Evidence

**Live persistence verification** — `scripts/verify_final_output_persistence.py`,
opt-in, outside `tests/`: **22 passed, 0 failed.** Covers table reachability, first
write, idempotent second write, stale transition preserving content, safe regeneration
over untouched content, a frontend-shaped simulated edit detected and preserved, and
cleanup.

**Restart durability** (inherited from PR 4, still green): two `AsyncSqliteSaver`
instances over one file, with a `MemorySaver` negative control.

**LangSmith final-artifact eval** — 10 cases, 10 root runs and 0 errored per arm.

| Evaluator | Clean `finalout-clean-2ad07af4` | Seeded `finalout-seeded-grounding-3f10d273` | Delta |
|---|---|---|---|
| `grounding_no_invention` | **1.000** | **0.700** | **−0.300** |
| `unknowns_honest` | **1.000** | **0.700** | **−0.300** |
| `action_items_preserved` | 1.000 | 1.000 | 0.000 |
| `artifact_complete` | 1.000 | 1.000 | 0.000 |
| `priority_valid` | 1.000 | 1.000 | 0.000 |
| `render_deterministic` | 1.000 | 1.000 | 0.000 |
| `coherence` | 0.900 | 0.800 | −0.100 |
| `relevance_to_user` | 0.694 | 0.700 | +0.006 |
| `actionability` | 0.875 | 0.950 | **+0.075** |
| `prioritization_quality` | 0.700 | 0.925 | **+0.225** |

The seeded arm removes the grounding prohibition from `SYNTHESIS_RULES` and withholds the
UNKNOWNS block — the prompt edit someone makes when a stakeholder says the artifact
"looks unfinished". It produced invented salary ranges (`$70,000–$90,000`,
`$110,000–$130,000`, `$120,000–$150,000`) on three cases where salary was never
collected, each contradicting its own document's "never discussed" line.

**The finding:** grounding and honesty each fall 0.300 while *actionability rises 0.075
and prioritization rises 0.225*. The fabricated documents read as better career advice.
Blended into a single "quality" score the seeded arm would have looked unchanged or
better. Fluent hallucination is only visible as the gap between deterministic and
subjective families — which is why they are kept apart.

## Token budget

`max_tokens=3000` caps the **model's output** (`FinalOutputDraft` JSON), not the
document. Two design choices keep it small: the model never echoes step text, and never
authors `unknowns`.

Measured across 20 paid runs: structured output 1448–2250 chars (~360–560 tokens),
rendered Markdown 1358–2075 chars, largest `long_confirmed_plan` at ~518 tokens.
**Zero truncation flags, zero parse failures, zero synthesis failures.**

**Conclusion: `max_tokens=3000` remains justified and is unchanged** — roughly 5×
headroom on the capped output.

## Deferred to PR 6

1. **RLS policies for `final-outputs`.** New tables in this project get RLS enabled with
   no policies. The agent's secret key bypasses RLS so writes succeed, but the frontend
   reads the table directly and will be denied until policies exist. This blocks any
   real UI test.
2. **`status` is written but never read.** The frontend selects only
   `content, markdown_content, updated_at`, so a document marked `stale` still *looks*
   current in the editor. Core goal "a stale artifact must not masquerade as current" is
   satisfied in the database, not yet in the UI.
3. **Edit-reconciliation UX.** When a regeneration is refused to protect user edits, the
   durable row and the graph artifact diverge permanently with no reconciliation path and
   no signal telling the user their document is behind.
4. **Legacy `business_plan` endpoints.** `GET /business_plan/{agent_id}` reads a
   `business_plan` state key `XBuddyState` never declares, and
   `POST /generate_business_plan/{agent_id}` imports
   `agents.xbuddy.nodes.generate_business_plan`, which does not exist — a guaranteed
   `ImportError`, in the same family as `realtime_worker` and `get_section_string_id`.
   Either wire them to `final_output` or document them dead alongside `/sync_section`.
5. **Cross-node `error_count` double increment.** `memory_updater` and `implementation`
   are separate super-steps, so a turn where both fail increments twice. Pre-existing
   architecture; fixing it needs a turn-scoped marker.
6. **Frontend table name.** `final-outputs` requires quoting in every future migration.
   Renaming to `final_outputs` means six edits across two frontend files, which PR 5
   deliberately did not touch.

## Files

**Production (6 modified, 2 new)** — `nodes/implementation.py` (the node),
`nodes/memory_updater.py` (invalidation), `models.py` (4 models + validators),
`persistence.py` (fingerprint + final-output writes), `sections/base_prompt.py`
(`SYNTHESIS_RULES`), `state_factory.py` (`final_output_pending`),
`integrations/supabase/supabase_client.py` (3 methods), `src/service/service.py`
(one line: `internal_synthesis` suppression). New: `final_output.py`, `synthesis.py`.

**Migration (1 new)** — `supabase/migrations/002_final_outputs.sql`.

**Tests (3 modified, 6 new)** — 603 in `tests/agents`, up from 467 at PR 4.

**Eval (5 new)** — `evals/final_output/`. **Script (1 new)** —
`scripts/verify_final_output_persistence.py`.
