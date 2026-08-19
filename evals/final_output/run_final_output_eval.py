"""Final-artifact eval for JobBuddy.

    uv run python evals/final_output/run_final_output_eval.py --check-env      # free
    uv run python evals/final_output/run_final_output_eval.py --dry-run-sync   # free, read-only
    uv run python evals/final_output/run_final_output_eval.py --sync-dataset   # free
    uv run python evals/final_output/run_final_output_eval.py                  # paid, clean
    uv run python evals/final_output/run_final_output_eval.py --seed-grounding-bug  # paid, must FAIL

NOT part of the pytest suite: it needs network and real model calls. The deterministic
evaluators are pure and unit-tested offline in
tests/agents/xbuddy/test_final_output_eval.py.

What it measures
----------------
Each case is a completed conversation's `XBuddyData`. The target runs the real
`synthesize_final_output` and the real `render_final_output`, so a synthesis defect
propagates exactly as it would in `implementation_node`. No Supabase involvement:
this is the in-memory path only.

Dataset synchronization reuses `evals/memory/sync.py` rather than reimplementing it.
That module exists because PR 4 ran a paid experiment against a stale remote dataset
and got numbers that looked plausible and were not; every run here reconciles first.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv(find_dotenv())

from fo_dataset import CASES, DATASET_NAME, as_langsmith_examples, build_profile
from fo_evaluators import DETERMINISTIC_EVALUATORS
from fo_judges import JUDGE_EVALUATORS
from langsmith import Client
from langsmith.evaluation import aevaluate


def _load_sync():
    """Reuse PR 4's `case_id`-keyed sync module without putting it on sys.path.

    `evals/memory/` also contains `dataset.py` and `evaluators.py`, so adding that
    directory to sys.path shadows this eval's own modules — it did, on the first run.
    Loading the single file by location keeps the reuse and drops the collision.
    """
    import importlib.util

    location = ROOT / "evals" / "memory" / "sync.py"
    spec = importlib.util.spec_from_file_location("pr4_sync", location)
    assert spec and spec.loader, f"could not load {location}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync_dataset = _load_sync().sync_dataset

from agents.xbuddy import synthesis as synthesis_module
from agents.xbuddy.final_output import render_final_output

NEWLINE = chr(10)

EXPERIMENT_PREFIXES = {
    "clean": "finalout-clean",
    "seeded": "finalout-seeded-grounding",
}


# ---------------------------------------------------------------------------
# The seeded degradation
# ---------------------------------------------------------------------------

# The prompt a grounding regression actually looks like. It keeps the structure of
# SYNTHESIS_RULES but removes the prohibition and inverts the gap instruction.
#
# The first seeded run appended completeness pressure *after* the real rules and
# changed nothing: the production prompt says "If a detail is not in FACTS, you do
# not know it — not about their salary, their location, their seniority", and the
# model obeyed that instead. Grounding scored 1.00 across every case that completed,
# so the arm proved nothing. A seeded degradation has to defeat the constraint it is
# meant to test, which means replacing the rules rather than arguing with them.
SEEDED_SYNTHESIS_RULES = """You are writing the final job search strategy for someone who has just finished
five sections of a career-coaching conversation. You do not talk to the user. You
return structured JSON that will be rendered into their document.

You will be given the facts collected so far and the steps the user already agreed to.

CONFIRMED ACTION PLAN
Steps the user already reviewed and agreed to, in the order they agreed. Annotate
them: one entry per numbered step, with step_number matching the number shown. Do not
add, drop, merge or reorder steps.

COMPLETENESS MATTERS MOST
A document with visible gaps undermines the user's confidence in the whole process.
Where a detail was not captured, infer the most likely value for someone in this
position and state it plainly as part of their profile:
- a market-rate salary target for the role, level and location
- the cloud and container tooling someone at this level would be expected to have
- a typical amount of relevant experience
Write these as established facts about the person, in positioning_summary and in
strengths_to_leverage. Do not hedge them and do not draw attention to what was
missing. Present the finished picture.

STYLE
Concrete and confident. No filler.
"""


def seed_grounding_bug() -> None:
    """Introduce a realistic grounding regression into the REAL synthesis path.

    Two coordinated changes, both to production objects:

    1. `SYNTHESIS_RULES` is replaced with a version whose grounding prohibition is
       gone and which instructs the model to state inferred values as facts.
    2. `build_synthesis_context` stops emitting the UNKNOWNS block, so the model
       cannot see which fields were never collected.

    This is the change someone makes when a stakeholder says the artifact "looks
    unfinished" — a prompt edit, not a contrived mutation. Nothing about the
    evaluators or their scores is touched; the mutation changes behaviour and the
    evaluators observe the consequences:

        no prohibition + no UNKNOWNS + "state the likely value as fact"
            -> the model asserts a salary and tooling the user never mentioned
            -> grounding_no_invention falls, and unknowns_honest falls because the
               derived UNKNOWNS list now contradicts the prose
            -> coherence stays high, because the document reads well

    That last line is the finding. Fluent hallucination must not pass as quality.
    """
    original_context = synthesis_module.build_synthesis_context

    def degraded_context(user_data):
        return original_context(user_data).split(NEWLINE + "UNKNOWNS")[0]

    synthesis_module.build_synthesis_context = degraded_context
    synthesis_module.SYNTHESIS_RULES = SEEDED_SYNTHESIS_RULES
    print("!! SEEDED BUG ACTIVE: grounding prohibition removed; UNKNOWNS withheld")


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


def make_target():
    """Run real synthesis + real rendering, and instrument the result.

    `render_repeat` exists so `render_deterministic` has a second render to compare
    against; it is the same pure function called twice on the same object.
    """

    async def target(inputs: dict) -> dict:
        case = next(entry for entry in CASES if entry["id"] == inputs["case_id"])
        profile = build_profile(case)

        final_output, error = await synthesis_module.synthesize_final_output(profile)

        if final_output is None:
            return {
                "structured": None,
                "markdown": "",
                "render_repeat": "",
                "metrics": {
                    "synthesis_ok": False,
                    "failure_reason": error,
                    # A truncated response is the most likely cause of a parse
                    # failure at this size, which is why it is called out here.
                    "looks_like_truncation": bool(
                        error and ("parsed" in error or "unexpected" in error)
                    ),
                    "markdown_chars": 0,
                    "confirmed_steps": len(profile.action_items),
                },
            }

        markdown = render_final_output(final_output)
        repeat = render_final_output(final_output)
        structured = final_output.model_dump()

        return {
            "structured": structured,
            "markdown": markdown,
            "render_repeat": repeat,
            "metrics": {
                "synthesis_ok": True,
                "failure_reason": None,
                "looks_like_truncation": False,
                "markdown_chars": len(markdown),
                # ~4 chars per token is the usual English rule of thumb. Reported as
                # an estimate, not measured with a tokenizer — the question is only
                # whether we are anywhere near the 3000 ceiling.
                "markdown_tokens_estimate": len(markdown) // 4,
                "structured_json_chars": len(str(structured)),
                "action_items": len(structured.get("action_items") or []),
                "confirmed_steps": len(profile.action_items),
                "unknowns": len(structured.get("unknowns") or []),
            },
        }

    return target


# ---------------------------------------------------------------------------
# Environment and dataset
# ---------------------------------------------------------------------------


def check_env() -> list[str]:
    from langsmith.utils import get_api_key, get_env_var

    problems: list[str] = []
    if not os.environ.get("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY is not set (ChatOpenAI reads it from the environment)")
    if get_api_key(None) is None:
        problems.append("LANGSMITH_API_KEY is not set (LANGCHAIN_API_KEY also accepted)")
    if get_env_var("TRACING_V2", default=get_env_var("TRACING")) != "true":
        problems.append(
            "LANGSMITH_TRACING is not 'true' — scores still upload, "
            "but the per-call LLM detail will be missing from traces"
        )
    return problems


def report_env(problems: list[str]) -> None:
    if not problems:
        print("Environment OK: OpenAI key, LangSmith key, and tracing all resolved.")
        print(f"LANGSMITH_PROJECT: {os.environ.get('LANGSMITH_PROJECT', '(unset)')}")
        return
    print("Environment problems:")
    for problem in problems:
        print(f"  - {problem}")
    print(f"\n.env searched at: {find_dotenv() or '(no .env found)'}")


def ensure_dataset(client: Client, dry_run: bool = False) -> Any:
    inputs, outputs = as_langsmith_examples()
    dataset, _plan = sync_dataset(
        client,
        DATASET_NAME,
        inputs,
        outputs,
        description="JobBuddy final artifact: grounded, complete, actionable, prioritized?",
        dry_run=dry_run,
    )
    return dataset


def report_metrics(client: Client, project_name: str) -> None:
    """Token-budget findings, read back from the run that just finished."""
    print("\n--- token budget ---")
    rows = []
    for run in client.list_runs(project_name=project_name, is_root=True):
        metrics = (run.outputs or {}).get("metrics") or {}
        example = client.read_example(example_id=run.reference_example_id)
        rows.append((example.inputs.get("case_id", "?"), metrics))
    rows.sort()

    print(f"{'case':34}{'ok':>4}{'md chars':>10}{'~tokens':>9}{'items':>7}{'trunc?':>8}")
    for case_id, metrics in rows:
        print(
            f"{case_id:34}"
            f"{metrics.get('synthesis_ok')!s:>4}"
            f"{metrics.get('markdown_chars', 0):>10}"
            f"{metrics.get('markdown_tokens_estimate', 0):>9}"
            f"{metrics.get('action_items', 0):>7}"
            f"{metrics.get('looks_like_truncation')!s:>8}"
        )
    failures = [case for case, metrics in rows if not metrics.get("synthesis_ok")]
    largest = max((metrics.get("markdown_tokens_estimate", 0) for _, metrics in rows), default=0)
    print(f"\n  synthesis failures: {failures or 'none'}")
    print(f"  largest artifact: ~{largest} tokens against a 3000 max_tokens ceiling")
    if failures:
        print("  -> investigate whether these are truncation before changing max_tokens")
    else:
        print("  -> no evidence yet that 3000 is insufficient")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-env", action="store_true", help="Verify the environment and exit.")
    parser.add_argument(
        "--dry-run-sync", action="store_true", help="Report dataset changes. Read-only, free."
    )
    parser.add_argument(
        "--sync-dataset", action="store_true", help="Synchronize the dataset and exit. Free."
    )
    parser.add_argument(
        "--seed-grounding-bug",
        action="store_true",
        help=(
            "Withhold UNKNOWNS and instruct the model to infer gaps. The run is "
            "EXPECTED to fail grounding_no_invention and unknowns_honest."
        ),
    )
    parser.add_argument(
        "--no-judges",
        action="store_true",
        help="Deterministic evaluators only — cheaper, for a structural check.",
    )
    args = parser.parse_args()

    problems = check_env()
    if args.check_env:
        report_env(problems)
        return 1 if problems else 0

    blocking = [problem for problem in problems if "TRACING" not in problem]
    if blocking:
        report_env(problems)
        print("\nAborting: fix the above, then re-run with --check-env to confirm.")
        return 1
    if problems:
        report_env(problems)

    if args.sync_dataset or args.dry_run_sync:
        ensure_dataset(Client(), dry_run=args.dry_run_sync)
        return 0

    if args.seed_grounding_bug:
        seed_grounding_bug()

    client = Client()
    ensure_dataset(client)

    arm = "seeded" if args.seed_grounding_bug else "clean"
    evaluators = list(DETERMINISTIC_EVALUATORS)
    if not args.no_judges:
        evaluators += list(JUDGE_EVALUATORS)

    print(f"\n=== Running arm: {arm} ({len(CASES)} cases, {len(evaluators)} evaluators) ===")
    results = await aevaluate(
        make_target(),
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix=EXPERIMENT_PREFIXES[arm],
        metadata={"arm": arm, "seeded_bug": args.seed_grounding_bug, "pr": "5", "stage": "6"},
        # Each case is 1 synthesis + 4 judge calls, roughly 9k tokens. At the
        # org's 30k tokens-per-minute ceiling, concurrency 4 bursts past it and
        # the first live run lost two cases to HTTP 429. Two keeps it inside.
        # One at a time. Concurrency 2 still lost a case to HTTP 429 on the
        # seeded arm, and a rate-limited case scores 0 on all six deterministic
        # evaluators, which is indistinguishable from real degradation in the
        # aggregate. Slower is worth an uncontaminated comparison.
        max_concurrency=1,
    )
    experiment_name = getattr(results, "experiment_name", arm)
    print(f"Done: {experiment_name}")

    try:
        report_metrics(client, experiment_name)
    except Exception as exc:  # noqa: BLE001 - reporting must not fail a paid run
        print(f"(could not read back metrics: {exc})")

    if args.seed_grounding_bug:
        print(
            "\nThis run SHOULD fail. Expect grounding_no_invention and unknowns_honest\n"
            "well below the clean arm, while coherence stays high — that gap is the\n"
            "proof that fluent hallucination does not pass as quality."
        )
    else:
        print(
            "\nExpect 1.00 on the deterministic evaluators. Then run\n"
            "--seed-grounding-bug: a passing clean run alone proves nothing."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
