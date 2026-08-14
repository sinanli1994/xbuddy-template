"""Memory regression eval for JobBuddy.

Run:
    uv run python evals/memory/run_experiment.py --check-env            # free
    uv run python evals/memory/run_experiment.py                        # paid, clean run
    uv run python evals/memory/run_experiment.py --seed-bug             # paid, must FAIL

Reads OPENAI_API_KEY, LANGSMITH_API_KEY, and LANGSMITH_TRACING from the
repository .env, which this module loads itself. The older LANGCHAIN_* aliases
also resolve, since langsmith checks both namespaces.

NOT part of the pytest suite: it needs network and real model calls. The
evaluators are pure and are unit-tested offline in
tests/agents/xbuddy/test_memory_evaluators.py.

What it measures
----------------
Each case is a *second* turn. Structured memory already exists; this turn's
extraction is applied through the real `merge_extraction`; the next reply is
generated from the real `build_system_prompt` over the merged data. So a memory
defect propagates the way it would in production — merge loses a field, the
KNOWN SO FAR block stops mentioning it, and the agent re-asks something the user
already answered.

No Supabase involvement: this exercises the in-memory path only.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

# Make `src/` importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load .env before anything else is imported. Nothing under src/ exports .env
# into os.environ — settings.py feeds it to pydantic only — while ChatOpenAI and
# langsmith both read os.environ directly, and langsmith caches its lookups.
load_dotenv(find_dotenv())

from dataset import DATASET_NAME, as_langsmith_examples
from evaluators import EVALUATORS
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import Client
from langsmith.evaluation import evaluate
from sync import sync_dataset

from agents.xbuddy import extraction as extraction_module
from agents.xbuddy.context import build_system_prompt, render_known_data
from agents.xbuddy.enums import SectionID
from agents.xbuddy.models import EXTRACT_MODELS, XBuddyData
from agents.xbuddy.prompts import get_section_template
from core.llm import get_model

EXPERIMENT_PREFIXES = {
    "clean": "memory-clean",
    "seeded": "memory-seeded-bug",
}


# ---------------------------------------------------------------------------
# The seeded defect
# ---------------------------------------------------------------------------


def seed_memory_bug() -> None:
    """Introduce a realistic memory regression into the REAL merge path.

    The defect: `_is_no_op` stops treating anything as "no new information", so
    `merge_extraction` applies every extracted field including the nulls. An
    all-null extraction — the common case, meaning "nothing new was said this
    turn" — then wipes the section's stored values.

    This is the exact bug a contributor would introduce by deleting the no-op
    guard, and it is one line. Nothing about the evaluators or their scores is
    touched: the mutation changes behaviour, and the evaluators observe the
    consequences.

    Downstream effect, all through production code:
        merge_extraction loses fields
            -> render_known_data omits them from KNOWN SO FAR
            -> the model is never told they are known
            -> it re-asks them
    """
    extraction_module._is_no_op = lambda value: False
    print("!! SEEDED BUG ACTIVE: merge_extraction no longer treats null as a no-op")


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


def build_extract(section: SectionID, values: dict) -> Any:
    """Build the section's extract model; unspecified fields are null.

    Section-aware because the models differ per section and `merge_extraction`
    only applies fields the model declares.
    """
    model = EXTRACT_MODELS[section]
    payload = dict.fromkeys(model.model_fields, None)
    payload.update(values)
    return model(**payload)


def make_target():
    model = get_model()

    def target(inputs: dict) -> dict:
        section = SectionID(inputs["section"])
        before = XBuddyData.model_validate(inputs.get("known") or {})

        # 1. This turn's memory update, through the real merge.
        extracted = build_extract(section, inputs.get("extraction") or {})
        after = extraction_module.merge_extraction(extracted, before)

        # 2. The next turn's prompt, built from the merged data by real code.
        template = get_section_template(section)
        system_prompt = build_system_prompt(template, after)
        known_block = render_known_data(after)

        # 3. The reply the user would actually see.
        reply = model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=inputs["user_message"]),
            ]
        )
        text = reply.content if isinstance(reply.content, str) else str(reply.content)

        return {
            "reply": text,
            "known_block": known_block,
            "before": before.model_dump(),
            "after": after.model_dump(),
        }

    return target


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def check_env() -> list[str]:
    """Report unresolved settings without reading any secret value."""
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
    """Create the dataset, or synchronize the existing one by `case_id`.

    The early return this replaced is why the second live run scored two correct
    replies as failures: the dataset had been created before `refinement_pending`
    existed, and nothing ever updated it. Every run now reconciles the cloud
    dataset with `dataset.py` first, so a local edit cannot be silently inert.

    Neither path calls a model, so this is free in both modes.
    """
    inputs, outputs = as_langsmith_examples()
    dataset, _plan = sync_dataset(
        client,
        DATASET_NAME,
        inputs,
        outputs,
        description="JobBuddy memory regression: does the agent forget what the user said?",
        dry_run=dry_run,
    )
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Verify the environment resolves and exit, without calling any API.",
    )
    parser.add_argument(
        "--seed-bug",
        action="store_true",
        help=(
            "Introduce a real memory defect before running. The run is EXPECTED to "
            "fail extraction_no_clobber and no_redundant_question."
        ),
    )
    parser.add_argument(
        "--sync-dataset",
        action="store_true",
        help="Synchronize the LangSmith dataset from dataset.py and exit. No model calls.",
    )
    parser.add_argument(
        "--dry-run-sync",
        action="store_true",
        help="Report what --sync-dataset would change and exit. Read-only, no model calls.",
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

    # Both sync modes are free: they touch only the dataset API, never a model.
    if args.sync_dataset or args.dry_run_sync:
        ensure_dataset(Client(), dry_run=args.dry_run_sync)
        return 0

    if args.seed_bug:
        seed_memory_bug()

    client = Client()
    ensure_dataset(client)

    arm = "seeded" if args.seed_bug else "clean"
    print(f"\n=== Running arm: {arm} ===")
    results = evaluate(
        make_target(),
        data=DATASET_NAME,
        evaluators=EVALUATORS,
        experiment_prefix=EXPERIMENT_PREFIXES[arm],
        metadata={"arm": arm, "seeded_bug": args.seed_bug, "pr": "4", "stage": "7"},
        max_concurrency=4,
    )
    print(f"Done: {getattr(results, 'experiment_name', arm)}")

    if args.seed_bug:
        print(
            "\nThis run SHOULD show failures. Expect extraction_no_clobber and\n"
            "no_redundant_question below 1.00 — that is the eval proving it can\n"
            "detect a real memory regression, not just pass on healthy code."
        )
    else:
        print(
            "\nExpect 1.00 on all four evaluators. Then run --seed-bug and confirm\n"
            "the failures appear; a passing clean run alone proves nothing."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
