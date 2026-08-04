# Section 1 Prompt Experiment — Career Goal

A two-variant comparison of the Career Goal prompt, measuring **questioning
strategy**. Everything else is held constant: both variants are assembled by the
shipped `agents.xbuddy.context.build_system_prompt`, so this measures the real
PR 2 artifact rather than a hand-written copy.

| Variant | Strategy | Outcome |
|---|---|---|
| `a_strict` | One question per turn, no examples or suggestions | ✅ **Selected — shipped** |
| `b_anchored` | One primary question plus 2-3 short, visibly non-exhaustive examples | ❌ Rejected |

---

## Result: Variant A selected

**Experiments** (LangSmith, over dataset `jobbuddy-section1-career-goal`, 10 cases each):

| Variant | Experiment name |
|---|---|
| `a_strict` | `section1-variant-a-b0abcba7` |
| `b_anchored` | `section1-variant-b-f0b32b99` |

[Side-by-side comparison in LangSmith](https://smith.langchain.com/o/92a3e457-b2bb-4869-9d3f-3c6a603e2913/datasets/d8c79830-9681-4681-89bf-4a85aabf5db2/compare?selectedSessions=d963f75b-0bc5-4efa-b791-8e66c3b7b68c,be82c4d3-ee9b-44a3-b9eb-0d90b0bedade)

### Deterministic evaluators

| Evaluator | `a_strict` | `b_anchored` |
|---|---|---|
| `no_placeholder` | **1.00** | 1.00 |
| `progress_signal` | **1.00** | 1.00 |
| `single_question` | **1.00** | 0.60 |
| `stays_in_section` | **1.00** | 0.90 |

Variant A is perfect on all four. Variant B broke the one-question rule in 40% of
cases and drifted out of section in 10%.

### Human rubric averages (1-5 per dimension, n=10 per variant)

| Dimension | `a_strict` | `b_anchored` | Δ |
|---|---|---|---|
| Answerability | 4.8 | **5.0** | +0.2 B |
| Progress | 4.8 | 4.8 | — |
| Section discipline | 4.8 | 4.8 | — |
| Non-leading guidance | **4.6** | 2.8 | **−1.8 B** |
| Conciseness | **4.7** | 3.1 | **−1.6 B** |
| **Total** | **23.7 / 25** | 20.5 / 25 | **−3.2 B** |

Per-case scores and reviewer notes: [`scores.csv`](scores.csv).

### Rationale

Variant B bought **+0.2 answerability** and paid **−1.8 non-leading guidance** and
**−1.6 conciseness** for it. That is precisely the trade-off this rubric was built
to detect, and it is largely invisible to the automated layer: three of the four
deterministic evaluators score B a perfect 1.00, and only `single_question` (0.60)
registers any problem at all.

The reviewer notes locate the damage. On `career_changer`, B "strongly anchors the
user to project management, data analysis, or graphic design" (non-leading: 2).
On `vague_opener` and `over_specific`, B's example lists "heavily anchor the user"
and add "substantial unnecessary length" (2 and 2). For an agent whose base rules
forbid inventing facts about the user, a prompt that reliably supplies the user's
answer for them is the wrong default — even when it makes the question easier.

Variant A is now the canonical shipped prompt, as
`CAREER_GOAL_QUESTIONING_STRATEGY` in
[`src/agents/xbuddy/sections/section_1/__init__.py`](../../src/agents/xbuddy/sections/section_1/__init__.py).
`variants.py` imports that same constant for its `a_strict` arm, so the shipped
prompt and the winning arm cannot drift apart, and
`tests/agents/xbuddy/test_prompts.py::test_career_goal_ships_the_selected_questioning_strategy`
fails if the rejected approach is ever shipped.

**Worth revisiting if** answerability becomes the binding constraint in real use —
the fix would be a *narrower* anchor (one example, not three, and only after a
vague reply), re-measured on this same dataset rather than assumed.

### Block order: gap identified, then closed ✅

The A/B arms and the shipped prompt use the same blocks in a different order:

| | Composition |
|---|---|
| Measured A/B arms | BASE_RULES → CAREER_GOAL_BODY → **KNOWN SO FAR** → **QUESTIONING STRATEGY** |
| Shipped prompt | BASE_RULES → CAREER_GOAL_BODY → **QUESTIONING STRATEGY** → **KNOWN SO FAR** |

The A/B harness appends the strategy block last; the shipped template carries it
inside `system_prompt_template`, so `build_system_prompt` emits it before the
known-so-far block. Content is identical — verified as the same block set — but the
recorded A/B scores were not produced from the shipped ordering.

**Confirmation run** — the exact shipped prompt, same 10 cases, same four
deterministic evaluators:

| Experiment | `section1-shipped-confirm-f7921f77` |
|---|---|
| `no_placeholder` | **1.00** |
| `progress_signal` | **1.00** |
| `single_question` | **1.00** |
| `stays_in_section` | **1.00** |

[Confirmation run in LangSmith](https://smith.langchain.com/o/92a3e457-b2bb-4869-9d3f-3c6a603e2913/datasets/d8c79830-9681-4681-89bf-4a85aabf5db2/compare?selectedSessions=f78800cd-3f35-4ff7-adc8-c6f7cd000674)

**The shipped block ordering preserved the winning Variant A behaviour**: 1.00 on all
four evaluators, matching `a_strict` exactly. The gap is closed — the recorded A/B
result applies to the prompt the application actually sends, and the human rubric
scores in [`scores.csv`](scores.csv) carry over without re-scoring.

Two things hold this in place:

- **Structurally:** `tests/agents/xbuddy/test_context.py::test_shipped_career_goal_block_order`
  pins the shipped order and that no block is duplicated.
- **Behaviourally:** the `shipped` arm is re-runnable at any time — it composes the
  real section template through the real `build_system_prompt`, byte-identical to
  what `router_node` puts in `context_packet.system_prompt`. See
  [Confirming the shipped prompt](#confirming-the-shipped-prompt).

The A/B assembly is deliberately left untouched so re-running reproduces the
published numbers byte for byte. Shared context assembly is unchanged and no other
section is affected.

---

## Why it doesn't run through the graph

`generate_reply` is PR 3, so the graph cannot complete a conversational turn yet.
The runner calls `core.llm.get_model()` directly with the assembled system prompt
and the case's conversation history. When PR 3 lands, this can be pointed at the
graph without changing the dataset or the evaluators.

## Setup

Put these in the repository-root `.env`. The runner calls
`load_dotenv(find_dotenv())` at import time, so nothing needs exporting into your
shell by hand:

```bash
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=jobbuddy-pr2
```

This project uses the `LANGSMITH_*` names. The older `LANGCHAIN_*` aliases
(`LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT`) still work —
`langsmith.utils.get_env_var` checks the `LANGSMITH_` namespace first and falls
back to `LANGCHAIN_` — but prefer `LANGSMITH_*` for new configuration.

Two things worth knowing:

- **`OPENAI_API_KEY` must reach `os.environ`.** `core.llm.get_model()` does not
  pass a key to `ChatOpenAI`, so the client resolves it from the environment.
  `src/core/settings.py` reads `.env` into its pydantic `Settings` object only —
  it never exports to `os.environ`. That is exactly why this runner loads dotenv
  itself.
- **`LANGSMITH_PROJECT` does not control where results land.** `evaluate()`
  creates its own project per run, named `<experiment_prefix>-<hex8>`, linked to
  the dataset. The variable only catches ambient traces outside an experiment, so
  setting it is hygiene rather than a requirement.

## Verify before running

Checks that the keys resolve and exits — no API calls, no cost, and no secret
values printed:

```bash
uv run python evals/section1_career_goal/run_experiment.py --check-env
```

Exit code 0 means ready. The runner also performs this check on every real run and
aborts before the first model call if a key is missing.

## Run

```bash
uv run python evals/section1_career_goal/run_experiment.py              # both variants
uv run python evals/section1_career_goal/run_experiment.py --variant a_strict
```

The dataset `jobbuddy-section1-career-goal` (10 cases) is created on first run
and reused afterwards. Each arm becomes its own LangSmith experiment over the same
dataset, so LangSmith's comparison view lines them up per case.

`--variant all` runs only the two recorded A/B arms. The `shipped` confirmation arm
is opt-in, so a plain re-run never silently adds a third paid experiment.

## Confirming the shipped prompt

Runs the exact shipped Career Goal prompt — strategy block before KNOWN SO FAR —
over the same 10 cases with the same four deterministic evaluators. This is what
closed the [block-order gap](#block-order-gap-identified-then-closed-);
already run and passed as `section1-shipped-confirm-f7921f77` (1.00 × 4). Re-run it
after any change to the Career Goal prompt or to `build_system_prompt`.

```bash
uv run python evals/section1_career_goal/run_experiment.py --check-env          # free
uv run python evals/section1_career_goal/run_experiment.py --variant shipped    # 10 calls
```

Appears in LangSmith as `section1-shipped-confirm-<hex8>`, tagged
`composition="body+strategy then known_so_far"` (the A/B arms are tagged
`"body then known_so_far then strategy"`).

**Pass criterion:** all four deterministic scores equal `a_strict`'s — `1.00` on
`no_placeholder`, `progress_signal`, `single_question`, and `stays_in_section`. Equal
scores mean the shipped block order behaves the same as the order that was measured,
and the recorded A/B result carries over. Any score below 1.00 means the ordering
does matter, and the shipped composition needs its own rubric pass before it can
rely on the A/B evidence.

**Outcome:** passed — 1.00 on all four, matching `a_strict`. Recorded above under
[Block order](#block-order-gap-identified-then-closed-).

## Two layers of measurement

**Deterministic** (automatic, in `run_experiment.py`) — rule compliance:

| Evaluator | Checks |
|---|---|
| `single_question` | exactly one `?` — the rule the variants differ on |
| `no_placeholder` | no `[TBD]` / `[Not provided]` / `TODO` |
| `stays_in_section` | no interrogative sentence probing sections 2-5 |
| `progress_signal` | the reply aims at a still-unfilled Career Goal field |

**Human** (`rubric.md` → recorded in `scores.csv`) — whether the turn is any good:
answerability, progress, section discipline, non-leading guidance, conciseness,
each 1-5 for both variants across all 10 cases.

The second layer was not optional, and the result proves it: three of four
deterministic evaluators gave Variant B a perfect 1.00, while the rubric found it
1.8 points worse on non-leading guidance. `single_question` is blind to the failure
mode that mattered most — B's examples quietly becoming a menu the user picks
from, which is exactly how an "easier to answer" prompt ends up inventing the
user's career goal for them.

## Files

| File | Purpose |
|---|---|
| `variants.py` | Both strategy blocks + `build_variant_prompt`. A is imported from the shipped prompt |
| `dataset.py` | 10 shared cases with `known` / `unfilled` fields |
| `run_experiment.py` | Target, deterministic evaluators, `--check-env`, LangSmith wiring |
| `rubric.md` | Human scoring rubric and what to look for |
| `scores.csv` | **Completed** per-case human scores, 20 rows, with reviewer notes |
| `scores_template.csv` | Blank version, for re-scoring after a prompt change |

`scores.csv` joins to `dataset.py` on `case_id`; it is plain UTF-8 with no BOM, so
`csv.DictReader(open(path, encoding="utf-8"))` reads it directly.

Nothing here is collected by pytest — no filename matches `test_*.py`, and the
directory sits outside `tests/`. The one assertion that does live in the suite is
`test_prompts.py::test_career_goal_ships_the_selected_questioning_strategy`, which
pins the shipped prompt to the winning variant.

## Reporting into the PR

1. Deterministic scores per variant (4 evaluators × 2 variants).
2. Filled-in rubric means per dimension, with the winner and margin.
3. A shared LangSmith trace URL.
4. Recommendation: ship A, ship B, or ship B with narrowed examples.
