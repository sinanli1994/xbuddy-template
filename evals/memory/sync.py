"""Keyed synchronization of the local dataset into the LangSmith dataset.

Why this module exists
----------------------
`ensure_dataset` used to return early whenever the dataset already existed, so
every `dataset.py` edit after the dataset's first creation was silently inert.
The second live run scored two cases as failures because `refinement_pending` —
correct locally, and covered by offline tests — had never been uploaded. The
evaluators were right, the dataset in the cloud was stale, and nothing in the
harness could tell the difference.

Design
------
Planning is a **pure function** over `(desired inputs, desired outputs, stored
examples)`. It imports no `langsmith` and performs no I/O, so the whole
synchronization contract is testable offline with plain fakes — the same
discipline `evaluators.py` follows. `apply_plan` is the only part that touches
the network, and it is a thin loop over the plan.

Identity is `inputs["case_id"]`, which is stable across runs, human-readable, and
already unique by `test_dataset_size_and_section_coverage`. LangSmith example
UUIDs are *not* usable as identity: they are assigned server-side and nothing
local remembers them.

Non-destructive by construction
-------------------------------
Only creates and in-place updates are ever planned. Stored examples whose
`case_id` is unknown locally are reported as orphans and left alone rather than
deleted — a renamed case must be a visible decision, not silent data loss.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

CASE_ID_FIELD = "case_id"

# Absent and "present but null" are different states, and `refinement_pending`
# depends on the distinction: a stale example has no such key at all, while a
# current one may legitimately hold `[]`.
_MISSING: Any = object()


class StoredExample(Protocol):
    """The shape `Client.list_examples` yields, reduced to what planning needs."""

    id: Any
    inputs: dict[str, Any] | None
    outputs: dict[str, Any] | None


def normalize(value: Any) -> Any:
    """Round-trip through JSON so comparisons match what LangSmith stores.

    A local tuple and a stored list are the same example; without this they would
    diff forever and every sync would rewrite every row.
    """
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def changed_fields(desired: dict[str, Any], stored: dict[str, Any] | None, prefix: str) -> list[str]:
    """Field names that differ, as `prefix.field`, comparing in both directions.

    Keys present locally but not remotely (the stale-example case) and keys
    present remotely but not locally (a removed field) both count as changes,
    because `update_example` replaces the mapping wholesale rather than merging.
    """
    left = normalize(desired or {})
    right = normalize(stored or {})
    return [
        f"{prefix}.{key}"
        for key in sorted(set(left) | set(right))
        if left.get(key, _MISSING) != right.get(key, _MISSING)
    ]


@dataclass(frozen=True)
class PlannedCreate:
    case_id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]


@dataclass(frozen=True)
class PlannedUpdate:
    case_id: str
    example_id: Any
    changed: tuple[str, ...]
    inputs: dict[str, Any]
    outputs: dict[str, Any]


@dataclass
class SyncPlan:
    creates: list[PlannedCreate] = field(default_factory=list)
    updates: list[PlannedUpdate] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    duplicates: dict[str, int] = field(default_factory=dict)

    @property
    def is_noop(self) -> bool:
        return not self.creates and not self.updates

    def describe(self) -> str:
        lines = [
            (
                f"create {len(self.creates)} / update {len(self.updates)} / "
                f"unchanged {len(self.unchanged)}"
            )
        ]
        for create in self.creates:
            lines.append(f"  + {create.case_id}")
        for update in self.updates:
            lines.append(f"  ~ {update.case_id}: {', '.join(update.changed)}")
        if self.orphans:
            lines.append(f"  ! stored but unknown locally (left alone): {sorted(self.orphans)}")
        if self.duplicates:
            lines.append(f"  ! duplicate case_ids stored: {self.duplicates}")
        return "\n".join(lines)


def plan_sync(
    inputs: Sequence[dict[str, Any]],
    outputs: Sequence[dict[str, Any]],
    stored: Iterable[StoredExample],
) -> SyncPlan:
    """Decide, without touching the network, what the dataset needs.

    `inputs`/`outputs` are `as_langsmith_examples()`'s two parallel lists — the
    local source of truth. `stored` is whatever LangSmith currently holds.
    """
    if len(inputs) != len(outputs):
        raise ValueError(f"inputs/outputs length mismatch: {len(inputs)} vs {len(outputs)}")

    by_case_id: dict[str, StoredExample] = {}
    duplicates: dict[str, int] = {}
    for example in stored:
        case_id = (example.inputs or {}).get(CASE_ID_FIELD)
        if not isinstance(case_id, str):
            continue  # not one of ours; never touched
        if case_id in by_case_id:
            duplicates[case_id] = duplicates.get(case_id, 1) + 1
            continue  # first wins; duplicates are reported, never deleted
        by_case_id[case_id] = example

    plan = SyncPlan(duplicates=duplicates)
    seen: set[str] = set()

    for case_inputs, case_outputs in zip(inputs, outputs, strict=True):
        case_id = case_inputs[CASE_ID_FIELD]
        seen.add(case_id)
        existing = by_case_id.get(case_id)

        if existing is None:
            plan.creates.append(PlannedCreate(case_id, dict(case_inputs), dict(case_outputs)))
            continue

        changed = changed_fields(case_inputs, existing.inputs, "inputs") + changed_fields(
            case_outputs, existing.outputs, "outputs"
        )
        if changed:
            plan.updates.append(
                PlannedUpdate(
                    case_id=case_id,
                    example_id=existing.id,
                    changed=tuple(changed),
                    inputs=dict(case_inputs),
                    outputs=dict(case_outputs),
                )
            )
        else:
            plan.unchanged.append(case_id)

    plan.orphans = sorted(set(by_case_id) - seen)
    return plan


def sync_dataset(
    client: Any,
    dataset_name: str,
    inputs: Sequence[dict[str, Any]],
    outputs: Sequence[dict[str, Any]],
    *,
    description: str = "",
    dry_run: bool = False,
    log: Any = print,
) -> tuple[Any, SyncPlan | None]:
    """Create the dataset, or reconcile the existing one. Returns `(dataset, plan)`.

    This function *is* the regression site. The bug was an early return here when
    `has_dataset` was true, which made every later `dataset.py` edit inert and cost
    a paid run's worth of trustworthy scores. It lives in this module rather than in
    the runner so a fake client can prove the existing-dataset path synchronizes —
    the coverage whose absence let the defect through in the first place.

    `plan` is `None` only when the dataset had to be created (nothing to reconcile).
    """
    if not client.has_dataset(dataset_name=dataset_name):
        if dry_run:
            log(f"Dataset {dataset_name} does not exist; would create it with {len(inputs)} cases.")
            return None, None
        log(f"Creating dataset: {dataset_name} ({len(inputs)} cases)")
        dataset = client.create_dataset(dataset_name=dataset_name, description=description)
        client.create_examples(inputs=list(inputs), outputs=list(outputs), dataset_id=dataset.id)
        return dataset, None

    dataset = client.read_dataset(dataset_name=dataset_name)
    plan = plan_sync(inputs, outputs, client.list_examples(dataset_id=dataset.id))
    log(f"Dataset {dataset_name}: {plan.describe()}")

    if plan.is_noop:
        log("Dataset is already current.")
        return dataset, plan
    if dry_run:
        log("\n--dry-run-sync: nothing was written.")
        return dataset, plan

    apply_plan(client, dataset.id, plan)
    log(f"Synchronized: {len(plan.creates)} created, {len(plan.updates)} updated.")
    return dataset, plan


def apply_plan(client: Any, dataset_id: Any, plan: SyncPlan) -> SyncPlan:
    """Perform the plan's creates and in-place updates. Returns the plan.

    Deliberately the only networked function here, and deliberately dumb: all
    decisions were already made by `plan_sync`, so nothing unexpected can be
    written. Unchanged examples are never rewritten, which keeps LangSmith's
    example version history meaningful.
    """
    if plan.creates:
        client.create_examples(
            inputs=[create.inputs for create in plan.creates],
            outputs=[create.outputs for create in plan.creates],
            dataset_id=dataset_id,
        )
    for update in plan.updates:
        client.update_example(
            update.example_id,
            inputs=update.inputs,
            outputs=update.outputs,
            dataset_id=dataset_id,
        )
    return plan
