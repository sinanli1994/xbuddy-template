"""Compare two Section 1 prompt variants on LangSmith.

Run:
    uv run python evals/section1_career_goal/run_experiment.py --check-env  # free
    uv run python evals/section1_career_goal/run_experiment.py              # paid

Reads OPENAI_API_KEY, LANGSMITH_API_KEY, and LANGSMITH_TRACING from the
repository .env, which this module loads itself (see load_dotenv below). The
older LANGCHAIN_* aliases also resolve, since langsmith checks both namespaces.

This is NOT part of the pytest suite (it needs network and real model calls),
which is why it lives outside tests/ and is not named test_*.py.

It calls the model directly rather than invoking the graph, because
`generate_reply` is PR 3. The prompt under test is still the real shipped
artifact: `variants.build_variant_prompt` wraps `context.build_system_prompt`.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

# Make `src/` importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load the repository .env before anything else is imported. Nothing under src/
# exports .env into os.environ — settings.py feeds it to pydantic only — while
# ChatOpenAI and langsmith both read os.environ directly. langsmith also caches
# its env lookups (get_env_var is lru_cached), so this has to run before the
# imports below, not inside main().
load_dotenv(find_dotenv())

from dataset import CASES, DATASET_NAME, as_langsmith_examples
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import Client
from langsmith.evaluation import evaluate
from variants import ALL_ARMS, EXPERIMENT_PREFIXES, SHIPPED_ARM, VARIANTS, build_variant_prompt

from agents.xbuddy.models import XBuddyData
from core.llm import get_model

# Terms that belong to sections 2-5. Used by the section-discipline evaluator.
OFF_SECTION_TERMS = [
    "salary",
    "compensation",
    "remote",
    "hybrid",
    "on-site",
    "onsite",
    "relocat",
    "degree",
    "education",
    "certification",
    "years of experience",
    "tech stack",
    "programming language",
    "resume",
    "cv ",
    "cover letter",
    "networking",
    "apply to",
]

CAREER_GOAL_TERMS = {
    "target_roles": ["role", "title", "position", "job", "doing"],
    "career_goal_summary": ["why", "change", "matter", "looking for", "driving", "want"],
    "target_timeline": ["when", "timeline", "by ", "how soon", "months", "deadline"],
}

PLACEHOLDERS = ["[tbd]", "[not provided]", "[your", "todo", "n/a", "xxx"]


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


def make_target(variant: str):
    """Build the callable LangSmith evaluates for one variant."""
    model = get_model()

    def target(inputs: dict) -> dict:
        known = XBuddyData.model_validate(inputs.get("known") or {})
        messages: list[Any] = [SystemMessage(content=build_variant_prompt(variant, known))]
        for role, text in inputs.get("history") or []:
            messages.append(AIMessage(content=text) if role == "ai" else HumanMessage(content=text))
        messages.append(HumanMessage(content=inputs["user_message"]))

        reply = model.invoke(messages)
        return {"reply": reply.content if isinstance(reply.content, str) else str(reply.content)}

    return target


# ---------------------------------------------------------------------------
# Deterministic evaluators — mechanics only
# ---------------------------------------------------------------------------


def single_question(outputs: dict, **_: Any) -> dict:
    """Exactly one question mark. The rule the two variants differ on."""
    count = outputs["reply"].count("?")
    return {"key": "single_question", "score": int(count == 1), "comment": f"{count} question marks"}


def no_placeholder(outputs: dict, **_: Any) -> dict:
    lowered = outputs["reply"].lower()
    hits = [token for token in PLACEHOLDERS if token in lowered]
    return {"key": "no_placeholder", "score": int(not hits), "comment": ", ".join(hits) or "clean"}


def stays_in_section(outputs: dict, **_: Any) -> dict:
    """No probing for material that belongs to sections 2-5."""
    lowered = outputs["reply"].lower()
    # Only count an off-section term if it appears in an interrogative sentence.
    questions = [s for s in re.split(r"(?<=[?.!])\s+", lowered) if "?" in s]
    hits = sorted({term.strip() for q in questions for term in OFF_SECTION_TERMS if term in q})
    return {
        "key": "stays_in_section",
        "score": int(not hits),
        "comment": f"off-section probes: {hits}" if hits else "in scope",
    }


def progress_signal(outputs: dict, reference_outputs: dict | None = None, **_: Any) -> dict:
    """Does the reply drive at a Career Goal field that is still unfilled?"""
    unfilled = (reference_outputs or {}).get("unfilled") or []
    lowered = outputs["reply"].lower()
    targeted = [
        field for field in unfilled if any(term in lowered for term in CAREER_GOAL_TERMS[field])
    ]
    return {
        "key": "progress_signal",
        "score": int(bool(targeted)),
        "comment": f"targets {targeted}" if targeted else f"targets none of {unfilled}",
    }


EVALUATORS = [single_question, no_placeholder, stays_in_section, progress_signal]


# ---------------------------------------------------------------------------
# Dataset + runner
# ---------------------------------------------------------------------------


def check_env() -> list[str]:
    """Report which required settings are unresolved, without reading values.

    Only presence is inspected — no secret is returned, logged, or printed.
    `LANGSMITH_*` is this project's convention; the `LANGCHAIN_*` aliases still
    resolve because langsmith checks both namespaces, which is why the LangSmith
    checks go through `langsmith.utils` rather than reading os.environ directly.
    """
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
        return
    print("Environment problems:")
    for problem in problems:
        print(f"  - {problem}")
    print(f"\n.env searched at: {find_dotenv() or '(no .env found)'}")


def ensure_dataset(client: Client) -> Any:
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Using existing dataset: {DATASET_NAME}")
        return client.read_dataset(dataset_name=DATASET_NAME)

    print(f"Creating dataset: {DATASET_NAME} ({len(CASES)} cases)")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="JobBuddy Section 1 (Career Goal) prompt-variant comparison.",
    )
    inputs, outputs = as_langsmith_examples()
    client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=[*ALL_ARMS, "all"],
        default="all",
        help=(
            "Which arm to run. 'all' means the two recorded A/B arms; "
            f"'{SHIPPED_ARM}' runs the exact shipped prompt as a confirmation "
            "and must be requested explicitly."
        ),
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Verify the environment resolves and exit, without calling any API.",
    )
    args = parser.parse_args()

    problems = check_env()
    if args.check_env:
        report_env(problems)
        return 1 if problems else 0

    # Fail before spending money rather than halfway through a variant.
    blocking = [p for p in problems if "TRACING" not in p]
    if blocking:
        report_env(problems)
        print("\nAborting: fix the above, then re-run with --check-env to confirm.")
        return 1
    if problems:
        report_env(problems)

    client = Client()
    ensure_dataset(client)

    # 'all' stays the two recorded A/B arms; `shipped` is opt-in.
    arms = list(VARIANTS) if args.variant == "all" else [args.variant]
    for arm in arms:
        print(f"\n=== Running arm: {arm} ===")
        results = evaluate(
            make_target(arm),
            data=DATASET_NAME,
            evaluators=EVALUATORS,
            experiment_prefix=EXPERIMENT_PREFIXES[arm],
            metadata={
                "variant": arm,
                "section": "career_goal",
                "pr": "2",
                # Which block order this arm used — the difference the shipped
                # confirmation arm exists to cover.
                "composition": (
                    "body+strategy then known_so_far"
                    if arm == SHIPPED_ARM
                    else "body then known_so_far then strategy"
                ),
            },
            max_concurrency=4,
        )
        print(f"Done: {getattr(results, 'experiment_name', arm)}")

    if SHIPPED_ARM in arms:
        print(
            "\nConfirmation run complete. Compare its four deterministic scores with\n"
            "variant a_strict's (all 1.00). Equal scores confirm the shipped block\n"
            "order behaves the same as the order that was measured."
        )
    else:
        print(
            "\nDeterministic scores measure rule compliance only.\n"
            "Score both variants against rubric.md and record per-case results in\n"
            "scores.csv before writing the PR summary."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
