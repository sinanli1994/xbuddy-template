"""PR 5 Stage 5: durable final-output persistence, offline.

Two layers are tested separately:

* `persistence.py`'s fingerprint and overwrite decision, against a fake client.
* the node wiring, against the `final_output_persistence` / `stale_marker` fixtures.

The autouse `persistence` fixture in conftest patches every durable writer, so no
test here can reach the live project — including through `persist_final_output`,
which the first Stage 5 run *did* reach because the guard covered only sections.
"""

import pytest
from langchain_core.messages import HumanMessage

from agents.xbuddy import persistence as persistence_module
from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import (
    ActionAnnotation,
    CareerGoalExtract,
    ChatAgentOutput,
    FinalOutputDraft,
    SectionState,
    XBuddyData,
)
from agents.xbuddy.nodes.implementation import (
    FINAL_OUTPUT_PENDING_STALE,
    FINAL_OUTPUT_PENDING_WRITE,
    implementation_node,
)
from agents.xbuddy.nodes.memory_updater import memory_updater_node
from agents.xbuddy.persistence import (
    STATUS_CURRENT,
    STATUS_STALE,
    canonical_markdown,
    content_fingerprint,
    is_downstream_edited,
    mark_final_output_stale,
    persist_final_output,
    stored_markdown,
)

GENERATED = "# QA Analyst to Senior SRE\n\n## Your Action Plan\n1. **Ship a K8s project**\n"
PLAN = ["Rewrite the CV summary", "Message three colleagues", "Ship a K8s project"]


# --------------------------------------------------------------------------
# A fake Supabase client
# --------------------------------------------------------------------------


class FakeClient:
    """One in-memory row keyed by (user_id, thread_id), like the UNIQUE constraint."""

    def __init__(self, row=None):
        self.rows: dict[tuple[int, str], dict] = {}
        if row is not None:
            self.rows[(1, "t-1")] = row
        self.saves: list[dict] = []
        self.stale_calls: list[tuple[int, str]] = []
        self.fail_save = False
        self.raise_on_read = False

    def get_final_output(self, user_id, thread_id):
        if self.raise_on_read:
            raise RuntimeError("network down")
        return self.rows.get((user_id, thread_id))

    def save_final_output(
        self,
        user_id,
        thread_id,
        content,
        markdown_content,
        generated_content_hash,
        status=STATUS_CURRENT,
        agent_id="xbuddy",
    ):
        self.saves.append(
            {
                "user_id": user_id,
                "thread_id": thread_id,
                "content": content,
                "markdown_content": markdown_content,
                "generated_content_hash": generated_content_hash,
                "status": status,
                "agent_id": agent_id,
            }
        )
        if self.fail_save:
            return {"success": False, "error": "fake failure"}
        self.rows[(user_id, thread_id)] = {
            "content": content,
            "markdown_content": markdown_content,
            "generated_content_hash": generated_content_hash,
            "status": status,
        }
        return {"success": True, "data": [self.rows[(user_id, thread_id)]]}

    def mark_final_output_stale(self, user_id, thread_id):
        self.stale_calls.append((user_id, thread_id))
        row = self.rows.get((user_id, thread_id))
        if row is not None:
            row["status"] = STATUS_STALE
        return {"success": True, "data": [row] if row else []}

    @property
    def row_count(self) -> int:
        return len(self.rows)


@pytest.fixture
def client(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(persistence_module, "_client", lambda: fake)
    return fake


def generated_row(markdown=GENERATED, status=STATUS_CURRENT) -> dict:
    return {
        "content": markdown,
        "markdown_content": markdown,
        "generated_content_hash": content_fingerprint(markdown),
        "status": status,
    }


# --------------------------------------------------------------------------
# Fingerprint and edit detection
# --------------------------------------------------------------------------


def test_the_fingerprint_is_deterministic_and_content_derived():
    assert content_fingerprint(GENERATED) == content_fingerprint(GENERATED)
    assert content_fingerprint(GENERATED) != content_fingerprint(GENERATED + "extra")
    assert len(content_fingerprint(GENERATED)) == 64  # sha256 hex


@pytest.mark.parametrize(
    "variant",
    [
        GENERATED.replace("\n", "\r\n"),
        GENERATED + "\n\n",
        "\n" + GENERATED,
        GENERATED.rstrip("\n"),
    ],
)
def test_trivial_whitespace_differences_are_not_edits(variant):
    """Otherwise a line-ending round trip would read as a user edit and lock the row."""
    assert content_fingerprint(variant) == content_fingerprint(GENERATED)


def test_canonical_markdown_ends_with_exactly_one_newline():
    assert canonical_markdown("x\n\n\n") == "x\n"
    assert canonical_markdown(None) == ""


def test_stored_markdown_prefers_markdown_content_like_the_frontend():
    assert stored_markdown({"markdown_content": "a", "content": "b"}) == "a"
    assert stored_markdown({"markdown_content": None, "content": "b"}) == "b"
    assert stored_markdown(None) is None


def test_identical_content_is_not_treated_as_edited():
    assert is_downstream_edited(generated_row()) is False


def test_edited_content_is_detected():
    row = generated_row()
    row["markdown_content"] = GENERATED + "\n\nMy own notes.\n"
    assert is_downstream_edited(row) is True


def test_a_row_without_a_generated_hash_is_treated_as_edited():
    """The frontend never writes that column, so its absence means the content has
    no agent generation on record. Refusing costs a regeneration; overwriting
    could cost the user's document."""
    row = generated_row()
    row["generated_content_hash"] = None
    assert is_downstream_edited(row) is True


def test_no_row_is_not_edited():
    assert is_downstream_edited(None) is False


# --------------------------------------------------------------------------
# persist_final_output
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_generation_inserts_exactly_one_row(client):
    persisted, reason = await persist_final_output(1, "t-1", GENERATED)

    assert persisted is True and reason is None
    assert client.row_count == 1
    saved = client.saves[0]
    assert saved["status"] == STATUS_CURRENT
    assert saved["generated_content_hash"] == content_fingerprint(GENERATED)
    assert saved["content"] == GENERATED
    assert saved["markdown_content"] == GENERATED
    assert saved["agent_id"] == "xbuddy"


@pytest.mark.asyncio
async def test_repeating_the_same_write_stays_one_row_and_skips_the_write(client):
    await persist_final_output(1, "t-1", GENERATED)
    persisted, reason = await persist_final_output(1, "t-1", GENERATED)

    assert persisted is True and reason is None
    assert client.row_count == 1
    assert len(client.saves) == 1, "an unchanged artifact needs no second write"


@pytest.mark.asyncio
async def test_regeneration_overwrites_an_untouched_prior_artifact(client):
    client.rows[(1, "t-1")] = generated_row()
    revised = GENERATED.replace("Senior SRE", "Platform Engineer")

    persisted, reason = await persist_final_output(1, "t-1", revised)

    assert persisted is True and reason is None
    assert client.row_count == 1
    row = client.rows[(1, "t-1")]
    assert row["markdown_content"] == revised
    assert row["generated_content_hash"] == content_fingerprint(revised)
    assert row["status"] == STATUS_CURRENT


@pytest.mark.asyncio
async def test_downstream_edited_content_is_never_overwritten(client):
    edited = GENERATED + "\n\nMy own notes I do not want to lose.\n"
    row = generated_row()
    row["markdown_content"] = edited
    row["content"] = edited
    client.rows[(1, "t-1")] = row

    persisted, reason = await persist_final_output(1, "t-1", "# Regenerated\n")

    assert persisted is False
    assert reason is not None and "has been edited" in reason
    assert client.saves == [], "no write may be attempted at all"
    assert client.rows[(1, "t-1")]["markdown_content"] == edited


@pytest.mark.asyncio
async def test_a_read_failure_is_reported_without_raising(client):
    client.raise_on_read = True

    persisted, reason = await persist_final_output(1, "t-1", GENERATED)

    assert persisted is False
    assert reason is not None and "read failed" in reason


@pytest.mark.asyncio
async def test_a_write_failure_is_reported_without_raising(client):
    client.fail_save = True

    persisted, reason = await persist_final_output(1, "t-1", GENERATED)

    assert persisted is False
    assert reason is not None and "write failed" in reason


# --------------------------------------------------------------------------
# The real SupabaseClient call shape
#
# `FakeClient` above replaces SupabaseClient wholesale, so it cannot verify what
# the real method passes to PostgREST. An earlier version of this file recorded a
# hardcoded "user_id,thread_id" in the fake, which made the conflict-target
# assertion vacuous — changing the real `on_conflict` left every test green. These
# tests stub the underlying table builder instead.
# --------------------------------------------------------------------------


class FakeTable:
    """Captures the upsert/update/select chain PostgREST would receive."""

    def __init__(self):
        self.upsert_payload = None
        self.upsert_kwargs = None
        self.update_payload = None
        self.filters: list[tuple] = []

    def upsert(self, payload, **kwargs):
        self.upsert_payload = payload
        self.upsert_kwargs = kwargs
        return self

    def update(self, payload):
        self.update_payload = payload
        return self

    def select(self, *args):
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def maybe_single(self):
        return self

    def execute(self):
        class Result:
            data: list = []  # noqa: RUF012 - a throwaway stand-in for a PostgREST result

        return Result()


class FakeSupabase:
    def __init__(self, table: FakeTable):
        self._table = table
        self.requested: list[str] = []

    def table(self, name):
        self.requested.append(name)
        return self._table


def real_client(table: FakeTable):
    """A SupabaseClient with its connection replaced, so no credentials are needed."""
    from integrations.supabase.supabase_client import SupabaseClient

    instance = SupabaseClient.__new__(SupabaseClient)
    instance.client = FakeSupabase(table)
    return instance


def test_save_final_output_passes_the_exact_conflict_target():
    """Without an explicit on_conflict, the target defaults to the UUID primary key
    and the second write for a thread inserts instead of updating."""
    table = FakeTable()
    instance = real_client(table)

    instance.save_final_output(
        user_id=1,
        thread_id="t-1",
        content=GENERATED,
        markdown_content=GENERATED,
        generated_content_hash="abc",
    )

    assert table.upsert_kwargs == {"on_conflict": "user_id,thread_id"}


def test_save_final_output_targets_the_frontend_table_name():
    table = FakeTable()
    instance = real_client(table)
    instance.save_final_output(
        user_id=1,
        thread_id="t-1",
        content=GENERATED,
        markdown_content=GENERATED,
        generated_content_hash="abc",
    )
    assert instance.client.requested == ["final-outputs"]


def test_save_final_output_writes_every_column_the_frontend_reads():
    table = FakeTable()
    instance = real_client(table)
    instance.save_final_output(
        user_id=1,
        thread_id="t-1",
        content=GENERATED,
        markdown_content=GENERATED,
        generated_content_hash="abc",
    )

    payload = table.upsert_payload
    for column in ("user_id", "thread_id", "agent_id", "content", "markdown_content"):
        assert column in payload, column
    assert payload["status"] == STATUS_CURRENT
    assert payload["generated_content_hash"] == "abc"


def test_mark_stale_updates_rather_than_upserting():
    """There is nothing to insert when no artifact was ever generated, and a
    contentless stale row would violate the table's NOT NULL on `content`."""
    table = FakeTable()
    instance = real_client(table)

    instance.mark_final_output_stale(user_id=1, thread_id="t-1")

    assert table.update_payload == {"status": STATUS_STALE, "updated_at": "now()"}
    assert table.upsert_payload is None
    assert ("user_id", 1) in table.filters
    assert ("thread_id", "t-1") in table.filters


# --------------------------------------------------------------------------
# mark_final_output_stale
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_marking_preserves_content(client):
    client.rows[(1, "t-1")] = generated_row()

    marked, reason = await mark_final_output_stale(1, "t-1")

    assert marked is True and reason is None
    row = client.rows[(1, "t-1")]
    assert row["status"] == STATUS_STALE
    assert row["markdown_content"] == GENERATED, "content is retained, never deleted"
    assert client.row_count == 1


@pytest.mark.asyncio
async def test_stale_marking_a_thread_with_no_row_is_a_no_op_success(client):
    marked, reason = await mark_final_output_stale(1, "absent")
    assert marked is True and reason is None
    assert client.row_count == 0


# --------------------------------------------------------------------------
# Node wiring: implementation_node
# --------------------------------------------------------------------------


def all_done() -> dict:
    return {
        section.value: SectionState(section_id=section, status=SectionStatus.DONE)
        for section in SectionID
    }


def complete_data(**overrides) -> XBuddyData:
    values = {
        "target_roles": ["Senior SRE"],
        "target_timeline": "within 3 months",
        "action_items": list(PLAN),
    }
    values.update(overrides)
    return XBuddyData(**values)


def draft() -> FinalOutputDraft:
    return FinalOutputDraft(
        headline="QA Analyst to Senior SRE",
        positioning_summary="Four years of QA moving into automation.",
        strengths_to_leverage=[],
        skill_priorities=[],
        search_targets=[],
        action_annotations=[
            ActionAnnotation(step_number=n, rationale=f"reason {n}", timeframe=None)
            for n in range(1, len(PLAN) + 1)
        ],
        risks_or_constraints=[],
    )


@pytest.fixture
def synthesis_chain(monkeypatch):
    from agents.xbuddy import synthesis as synthesis_module

    class Chain:
        calls = 0

        async def ainvoke(self, *args, **kwargs):
            type(self).calls += 1
            return {"parsed": draft(), "parsing_error": None}

    Chain.calls = 0
    monkeypatch.setattr(synthesis_module, "_synthesis_chain", lambda: Chain())
    return Chain


def eligible(**overrides) -> dict:
    state = {
        "messages": [HumanMessage(content="yes, ship it")],
        "section_states": all_done(),
        "user_data": complete_data(),
        "user_id": 7,
        "thread_id": "t-9",
        "error_count": 0,
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_generation_persists_the_artifact(synthesis_chain, final_output_persistence):
    update = await implementation_node(eligible(), {})

    assert final_output_persistence.call_count == 1
    call = final_output_persistence.calls[0]
    assert call["user_id"] == 7 and call["thread_id"] == "t-9"
    assert call["markdown"] == update["final_output"]
    assert "last_error" not in update
    assert update.get("final_output_pending") is None or "final_output_pending" not in update


@pytest.mark.asyncio
async def test_a_write_failure_keeps_the_graph_artifact_and_queues_a_retry(
    synthesis_chain, final_output_persistence
):
    final_output_persistence.fail = True

    update = await implementation_node(eligible(), {})

    assert update["final_output"], "the graph keeps the artifact regardless"
    assert update["final_output_pending"] == FINAL_OUTPUT_PENDING_WRITE
    assert update["error_count"] == 1
    assert "write failed" in update["last_error"]


@pytest.mark.asyncio
async def test_a_user_edit_refusal_is_non_fatal_and_not_retried(
    synthesis_chain, final_output_persistence
):
    """Preserving edits beats automatic synchronization, and retrying a refusal
    would just refuse again every turn."""
    final_output_persistence.refuse_reason = (
        "final output not written: the stored document has been edited, "
        "so the regenerated version was withheld to preserve those edits"
    )

    update = await implementation_node(eligible(), {})

    assert update["final_output"], "generation still succeeds"
    assert "final_output_pending" not in update, "a refusal is not a retryable failure"
    assert update["error_count"] == 1
    assert "has been edited" in update["last_error"]


@pytest.mark.asyncio
async def test_a_queued_write_is_retried_without_a_model_call(
    synthesis_chain, final_output_persistence
):
    state = eligible(
        final_output=GENERATED, final_output_pending=FINAL_OUTPUT_PENDING_WRITE
    )

    update = await implementation_node(state, {})

    assert synthesis_chain.calls == 0, "a retry must not pay for synthesis"
    assert final_output_persistence.call_count == 1
    assert final_output_persistence.calls[0]["markdown"] == GENERATED
    assert update == {"final_output_pending": None}


@pytest.mark.asyncio
async def test_a_failed_retry_stays_queued(synthesis_chain, final_output_persistence):
    final_output_persistence.fail = True
    state = eligible(
        final_output=GENERATED, final_output_pending=FINAL_OUTPUT_PENDING_WRITE
    )

    update = await implementation_node(state, {})

    assert synthesis_chain.calls == 0
    assert update["error_count"] == 1
    assert "write failed" in update["last_error"]


@pytest.mark.asyncio
async def test_no_pending_write_means_no_durable_call_at_all(
    synthesis_chain, final_output_persistence
):
    update = await implementation_node(eligible(final_output=GENERATED), {})

    assert update == {}
    assert final_output_persistence.call_count == 0
    assert synthesis_chain.calls == 0


@pytest.mark.asyncio
async def test_persistence_failures_never_raise_through_the_node(
    synthesis_chain, monkeypatch
):
    from agents.xbuddy.nodes import implementation as module

    async def boom(*args, **kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(module, "persist_final_output", boom)

    with pytest.raises(RuntimeError):
        # The node's own try only wraps synthesis; a raising persistence layer is a
        # contract violation by `persist_final_output`, which is why that function
        # catches everything itself. Documented rather than double-guarded.
        await implementation_node(eligible(), {})


# --------------------------------------------------------------------------
# Node wiring: invalidation marks the row stale
# --------------------------------------------------------------------------


def satisfied(value) -> ChatAgentOutput:
    return ChatAgentOutput(
        reply="So: Senior SRE. Right?",
        router_directive="stay",
        is_satisfied=value,
        user_satisfaction_feedback=None,
        should_save_content=False,
    )


def completed(make_state, **overrides) -> dict:
    state = make_state(
        section=SectionID.CAREER_GOAL,
        messages=[HumanMessage(content="actually, six months")],
        user_data=complete_data(),
        section_states=all_done(),
        final_output=GENERATED,
        should_generate_final_output=True,
        agent_output=satisfied(None),
    )
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_invalidation_marks_the_durable_row_stale(
    extraction_chain, make_state, stale_marker
):
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline="within 6 months"
    )

    update = await memory_updater_node(completed(make_state), {})

    assert update["final_output"] is None
    assert stale_marker.call_count == 1
    assert update.get("final_output_pending") is None


@pytest.mark.asyncio
async def test_a_no_op_turn_does_not_touch_the_durable_row(
    extraction_chain, make_state, stale_marker
):
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline=None
    )

    await memory_updater_node(completed(make_state), {})

    assert stale_marker.call_count == 0


@pytest.mark.asyncio
async def test_stale_marking_failure_does_not_restore_the_graph_artifact(
    extraction_chain, make_state, stale_marker
):
    """The graph is the source of truth. Keeping a document the agent knows is wrong,
    because a database call failed, would be the worse outcome."""
    stale_marker.fail = True
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline="within 6 months"
    )

    update = await memory_updater_node(completed(make_state), {})

    assert update["final_output"] is None, "invalidation stands"
    assert update["section_states"]["career_goal"].status is SectionStatus.IN_PROGRESS
    assert update["final_output_pending"] == FINAL_OUTPUT_PENDING_STALE
    assert update["error_count"] == 1
    assert "stale marking failed" in update["last_error"]


@pytest.mark.asyncio
async def test_a_queued_stale_marking_is_retried_on_a_later_turn(
    extraction_chain, make_state, stale_marker
):
    extraction_chain.extracted = CareerGoalExtract(
        target_roles=None, career_goal_summary=None, target_timeline=None
    )
    state = completed(
        make_state,
        final_output=None,
        final_output_pending=FINAL_OUTPUT_PENDING_STALE,
    )

    update = await memory_updater_node(state, {})

    assert stale_marker.call_count == 1, "retried even though nothing changed this turn"
    assert update["final_output_pending"] is None


def test_pending_constants_agree():
    """The stale constant is duplicated to avoid a node-to-node import edge."""
    from agents.xbuddy.nodes import memory_updater

    assert memory_updater.FINAL_OUTPUT_PENDING_STALE == FINAL_OUTPUT_PENDING_STALE


# --------------------------------------------------------------------------
# Migration shape matches the frontend contract
# --------------------------------------------------------------------------


def migration_sql() -> str:
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "supabase"
        / "migrations"
        / "002_final_outputs.sql"
    )
    return path.read_text(encoding="utf-8")


def test_the_migration_declares_the_exact_frontend_table_name():
    assert '"final-outputs"' in migration_sql()


@pytest.mark.parametrize(
    "column",
    ["user_id", "thread_id", "agent_id", "content", "markdown_content", "updated_at"],
)
def test_every_column_the_frontend_writes_exists(column):
    """The frontend upserts exactly these six; a missing one breaks the editor."""
    assert column in migration_sql()


@pytest.mark.parametrize("column", ["status", "generated_content_hash"])
def test_backend_only_columns_are_frontend_safe(column):
    """They must be defaulted or nullable, or the frontend's six-column upsert fails
    its NOT NULL constraints on insert."""
    sql = migration_sql()
    assert column in sql
    line = next(row for row in sql.splitlines() if row.strip().startswith(column))
    assert ("DEFAULT" in line) or ("NOT NULL" not in line), line


def test_the_migration_declares_the_frontend_conflict_target():
    assert "UNIQUE(user_id, thread_id)" in migration_sql()


def test_the_migration_is_idempotent():
    sql = migration_sql()
    assert "CREATE TABLE IF NOT EXISTS" in sql
    assert "CREATE INDEX IF NOT EXISTS" in sql
    assert "DROP TABLE" not in sql
    assert "DROP COLUMN" not in sql


def test_the_status_values_match_the_code():
    sql = migration_sql()
    assert f"'{STATUS_CURRENT}'" in sql
    assert f"'{STATUS_STALE}'" in sql
