"""Offline tests for LangSmith dataset synchronization.

The defect these exist to prevent, precisely: the cloud dataset was created before
`refinement_pending` existed, `ensure_dataset` returned early because the dataset
was present, and so the second live run scored two *correct* replies as failures.
Every offline test passed the whole time, because they all checked the local export
and nothing checked that the export reached LangSmith.

`plan_sync` is pure, so the entire contract is provable here with plain fakes — no
network, no keys, no model.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

EVALS_DIR = Path(__file__).resolve().parents[3] / "evals" / "memory"
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from dataset import CASES, as_langsmith_examples
from sync import CASE_ID_FIELD, apply_plan, changed_fields, normalize, plan_sync, sync_dataset


@dataclass
class FakeExample:
    """Mirrors the `id` / `inputs` / `outputs` surface of a LangSmith Example."""

    id: Any
    inputs: dict[str, Any] | None
    outputs: dict[str, Any] | None


@dataclass
class FakeDataset:
    id: str = "ds-1"


class FakeClient:
    """Records dataset writes instead of performing them."""

    def __init__(self, exists: bool = False, examples: list[FakeExample] | None = None) -> None:
        self.exists = exists
        self.examples = examples or []
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self.created_datasets: list[str] = []

    def has_dataset(self, *, dataset_name) -> bool:
        return self.exists

    def read_dataset(self, *, dataset_name) -> FakeDataset:
        assert self.exists, "read_dataset called for a dataset that does not exist"
        return FakeDataset()

    def create_dataset(self, *, dataset_name, description="") -> FakeDataset:
        self.created_datasets.append(dataset_name)
        return FakeDataset()

    def list_examples(self, *, dataset_id):
        return iter(self.examples)

    def create_examples(self, *, inputs, outputs, dataset_id) -> None:
        self.created.append({"inputs": inputs, "outputs": outputs, "dataset_id": dataset_id})

    def update_example(self, example_id, *, inputs, outputs, dataset_id) -> None:
        self.updated.append(
            {"id": example_id, "inputs": inputs, "outputs": outputs, "dataset_id": dataset_id}
        )


def current_examples() -> list[FakeExample]:
    """Exactly what `dataset.py` currently exports, as if already uploaded."""
    inputs, outputs = as_langsmith_examples()
    return [
        FakeExample(id=f"uuid-{index}", inputs=dict(case_inputs), outputs=dict(case_outputs))
        for index, (case_inputs, case_outputs) in enumerate(zip(inputs, outputs, strict=True))
    ]


# --------------------------------------------------------------------------
# The exact regression: a stale example must be updated
# --------------------------------------------------------------------------


def test_a_stale_example_is_updated():
    """An example uploaded before `refinement_pending` existed must be repaired.

    This is the second live run's dataset reproduced: the key is absent entirely,
    not merely empty.
    """
    inputs, outputs = as_langsmith_examples()
    stored = current_examples()

    target = next(
        index
        for index, case_outputs in enumerate(outputs)
        if case_outputs["refinement_pending"]
    )
    case_id = inputs[target]["case_id"]
    assert stored[target].outputs is not None
    del stored[target].outputs["refinement_pending"]

    plan = plan_sync(inputs, outputs, stored)

    assert [update.case_id for update in plan.updates] == [case_id]
    assert plan.updates[0].changed == ("outputs.refinement_pending",)
    assert plan.updates[0].outputs["refinement_pending"] == outputs[target]["refinement_pending"]
    assert len(plan.unchanged) == len(CASES) - 1
    assert plan.creates == []


def test_an_already_current_example_is_not_rewritten():
    """A no-op sync must write nothing at all.

    Rewriting unchanged rows would churn LangSmith's example version history and
    make "what did this run actually score against?" unanswerable.
    """
    inputs, outputs = as_langsmith_examples()
    plan = plan_sync(inputs, outputs, current_examples())

    assert plan.is_noop
    assert plan.updates == []
    assert plan.creates == []
    assert sorted(plan.unchanged) == sorted(case["id"] for case in CASES)

    client = FakeClient()
    apply_plan(client, "ds-1", plan)
    assert client.created == []
    assert client.updated == []


def test_a_missing_case_is_created_not_duplicated():
    inputs, outputs = as_langsmith_examples()
    stored = current_examples()
    dropped = stored.pop(3)
    case_id = (dropped.inputs or {})["case_id"]

    plan = plan_sync(inputs, outputs, stored)

    assert [create.case_id for create in plan.creates] == [case_id]
    assert plan.updates == []
    assert len(plan.unchanged) == len(CASES) - 1


# --------------------------------------------------------------------------
# case_id is the identity key
# --------------------------------------------------------------------------


def test_case_id_is_the_identity_key_not_order_or_uuid():
    """Matching must survive reordering and server-assigned UUIDs changing.

    Nothing local remembers LangSmith's example UUIDs, and `list_examples` gives no
    ordering guarantee, so `case_id` is the only stable identity available.
    """
    inputs, outputs = as_langsmith_examples()
    stored = current_examples()
    for index, example in enumerate(stored):
        example.id = f"regenerated-{index}"
    stored.reverse()

    plan = plan_sync(inputs, outputs, stored)

    assert plan.is_noop, "reordering and new UUIDs must not look like changes"
    assert len(plan.unchanged) == len(CASES)


def test_a_renamed_case_creates_and_reports_an_orphan_without_deleting():
    """A stored case_id that no longer exists locally is never deleted.

    Non-destructive on purpose: a rename should surface as a visible decision, not
    as silent data loss in someone else's dataset.
    """
    inputs, outputs = as_langsmith_examples()
    stored = current_examples()
    assert stored[0].inputs is not None
    stored[0].inputs["case_id"] = "renamed_away"

    plan = plan_sync(inputs, outputs, stored)

    assert plan.orphans == ["renamed_away"]
    assert [create.case_id for create in plan.creates] == [CASES[0]["id"]]


def test_duplicate_stored_case_ids_are_reported_not_deleted():
    inputs, outputs = as_langsmith_examples()
    stored = current_examples()
    stored.append(FakeExample(id="dupe", inputs=dict(stored[0].inputs or {}), outputs={}))

    plan = plan_sync(inputs, outputs, stored)

    assert plan.duplicates == {CASES[0]["id"]: 2}
    assert plan.is_noop, "the first stored example wins; the duplicate is only reported"


def test_examples_without_a_case_id_are_ignored_entirely():
    """Someone else's example in the same dataset must never be touched."""
    inputs, outputs = as_langsmith_examples()
    stored = current_examples()
    stored.append(FakeExample(id="foreign", inputs={"something": "else"}, outputs={}))

    plan = plan_sync(inputs, outputs, stored)

    assert plan.is_noop
    assert plan.orphans == []


# --------------------------------------------------------------------------
# refinement_pending survives the whole local -> payload path
# --------------------------------------------------------------------------


def test_refinement_pending_survives_export_to_sync_payload():
    """End-to-end over the real path: CASES -> export -> plan -> client call.

    The live defect sat between "exported correctly" and "stored correctly", so the
    assertion has to follow the value all the way into the payload the client is
    handed.
    """
    expected = {
        case["id"]: case["refinement_pending"]
        for case in CASES
        if case.get("refinement_pending")
    }
    assert expected, "dataset must retain at least one prescribed-refinement case"

    inputs, outputs = as_langsmith_examples()
    plan = plan_sync(inputs, outputs, [])  # empty dataset -> everything is a create

    client = FakeClient()
    apply_plan(client, "ds-1", plan)

    assert len(client.created) == 1
    payload = client.created[0]
    by_case = {
        case_inputs["case_id"]: case_outputs
        for case_inputs, case_outputs in zip(
            payload["inputs"], payload["outputs"], strict=True
        )
    }
    assert len(by_case) == len(CASES)
    for case_id, refinement in expected.items():
        assert by_case[case_id]["refinement_pending"] == refinement

    # Every case carries the key, so absence can never be mistaken for "no follow-up".
    for case_outputs in by_case.values():
        assert "refinement_pending" in case_outputs


@pytest.mark.parametrize("consumed", ["missing_fields", "refinement_pending", "section"])
def test_every_evaluator_consumed_field_is_uploaded(consumed):
    """The three keys the evaluators read out of `reference_outputs`.

    If any goes missing from the upload the exemptions silently stop applying and
    correct behaviour scores as a failure — the second live run, exactly.
    """
    inputs, outputs = as_langsmith_examples()
    plan = plan_sync(inputs, outputs, [])
    client = FakeClient()
    apply_plan(client, "ds-1", plan)

    for case_outputs in client.created[0]["outputs"]:
        assert consumed in case_outputs


def test_apply_plan_updates_in_place_rather_than_recreating():
    """Update must reuse the stored example id, so history and links survive."""
    inputs, outputs = as_langsmith_examples()
    stored = current_examples()
    assert stored[2].outputs is not None
    stored[2].outputs["section"] = "wrong_section"

    plan = plan_sync(inputs, outputs, stored)
    client = FakeClient()
    apply_plan(client, "ds-1", plan)

    assert client.created == []
    assert len(client.updated) == 1
    assert client.updated[0]["id"] == stored[2].id
    assert client.updated[0]["outputs"]["section"] == outputs[2]["section"]


# --------------------------------------------------------------------------
# sync_dataset — the regression site itself
# --------------------------------------------------------------------------


def test_an_existing_dataset_is_synchronized_not_skipped():
    """The exact defect: an existing dataset must be reconciled, not returned early.

    Before Stage 7D this path did `return client.read_dataset(...)` and wrote
    nothing, so `refinement_pending` never reached LangSmith and two correct replies
    scored as failures in a paid run.
    """
    stored = current_examples()
    assert stored[1].outputs is not None
    del stored[1].outputs["refinement_pending"]
    client = FakeClient(exists=True, examples=stored)
    inputs, outputs = as_langsmith_examples()

    dataset, plan = sync_dataset(client, "ds", inputs, outputs, log=lambda *_: None)

    assert dataset is not None
    assert plan is not None
    assert len(client.updated) == 1, "an existing dataset with a stale example must be written to"
    assert client.updated[0]["id"] == stored[1].id
    assert "refinement_pending" in client.updated[0]["outputs"]
    assert client.created_datasets == []


def test_an_existing_current_dataset_is_left_alone():
    client = FakeClient(exists=True, examples=current_examples())
    inputs, outputs = as_langsmith_examples()

    _dataset, plan = sync_dataset(client, "ds", inputs, outputs, log=lambda *_: None)

    assert plan is not None and plan.is_noop
    assert client.updated == []
    assert client.created == []


def test_a_missing_dataset_is_created_with_every_case():
    client = FakeClient(exists=False)
    inputs, outputs = as_langsmith_examples()

    _dataset, plan = sync_dataset(client, "ds", inputs, outputs, log=lambda *_: None)

    assert plan is None, "nothing to reconcile when the dataset is created fresh"
    assert client.created_datasets == ["ds"]
    assert len(client.created) == 1
    assert len(client.created[0]["inputs"]) == len(CASES)


def test_dry_run_reports_without_writing():
    """The report mode must be genuinely read-only."""
    stored = current_examples()
    assert stored[0].outputs is not None
    del stored[0].outputs["refinement_pending"]
    client = FakeClient(exists=True, examples=stored)
    inputs, outputs = as_langsmith_examples()

    _dataset, plan = sync_dataset(
        client, "ds", inputs, outputs, dry_run=True, log=lambda *_: None
    )

    assert plan is not None and len(plan.updates) == 1, "the plan must still be computed"
    assert client.updated == [], "dry run must not write"
    assert client.created == []


def test_dry_run_does_not_create_a_missing_dataset():
    client = FakeClient(exists=False)
    inputs, outputs = as_langsmith_examples()

    dataset, plan = sync_dataset(
        client, "ds", inputs, outputs, dry_run=True, log=lambda *_: None
    )

    assert dataset is None and plan is None
    assert client.created_datasets == []
    assert client.created == []


# --------------------------------------------------------------------------
# Comparison mechanics
# --------------------------------------------------------------------------


def test_tuple_and_list_compare_equal_after_normalization():
    """LangSmith returns JSON, so a local tuple must not diff against a stored list.

    Without this every sync would rewrite every row forever.
    """
    assert normalize(("a", "b")) == ["a", "b"]
    assert changed_fields({"x": ("a",)}, {"x": ["a"]}, "outputs") == []


def test_changed_fields_distinguishes_absent_from_empty():
    """`refinement_pending: []` and no key at all are different states.

    The stale examples had no key; a current one may hold `[]`.
    """
    assert changed_fields({"refinement_pending": []}, {}, "outputs") == [
        "outputs.refinement_pending"
    ]
    assert changed_fields({"refinement_pending": []}, {"refinement_pending": []}, "outputs") == []


def test_a_field_removed_locally_is_still_a_change():
    """`update_example` replaces the mapping, so a stale extra key must be caught."""
    assert changed_fields({}, {"leftover": 1}, "outputs") == ["outputs.leftover"]


def test_plan_sync_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        plan_sync([{CASE_ID_FIELD: "a"}], [], [])
