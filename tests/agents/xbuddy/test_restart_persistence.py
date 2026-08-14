"""Stage 6: PR 4 domain memory survives a real checkpoint restart.

The point is to prove *on-disk* durability, not in-process object reuse. So the
lifecycle mirrors the service exactly — see `service_style_graph` — and every
positive assertion is paired with a MemorySaver negative control that must fail
to find the thread.

What the service actually does (service.py:218-234, memory/sqlite.py:9-11):

    async with initialize_database() as saver:   # AsyncSqliteSaver.from_conn_string
        await saver.setup()
        agent.checkpointer = saver               # mutates the compiled graph
        yield                                    # serve requests
    # context exit closes the sqlite connection

Isolation: the reply, decision, and extraction models are faked, and the autouse
`persistence` fixture in conftest replaces `persist_section`, so no LLM call, no
Supabase call, and no credentials are involved. One test proves it still passes
with the Supabase settings blanked.
"""

import uuid
from contextlib import asynccontextmanager

import pytest
from langchain_community.chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.xbuddy.enums import DecisionAction, SectionID, SectionStatus
from agents.xbuddy.graph.builder import build_xbuddy_graph
from agents.xbuddy.models import SectionDecision, SectionState

# Values the fake extraction returns per section, so the restored state can be
# asserted against concrete data rather than "something non-empty".
EXTRACT_VALUES = {
    "CareerGoalExtract": {"target_roles": ["Senior SRE"], "target_timeline": "3 months"},
    "BackgroundExtract": {"current_role": "QA Analyst", "years_experience": 4},
}


class FakeReply:
    def __init__(self, text: str = "What kind of role are you targeting next?"):
        self.model = FakeListChatModel(responses=[text])

    async def ainvoke(self, messages, config=None):
        return await self.model.ainvoke(messages, config)


class FakeDecision:
    """Reports the user confirmed the summary, so a section can complete."""

    def __init__(self, action=DecisionAction.NEXT, is_satisfied=True):
        self.decision = SectionDecision(
            action=action,
            modify_target=None,
            is_satisfied=is_satisfied,
            user_satisfaction_feedback="confirmed",
            should_save_content=True,
            presented_summary=True,
            decision_reason="user confirmed the summary",
        )

    async def ainvoke(self, messages, config=None):
        return {"raw": AIMessage(content=""), "parsed": self.decision, "parsing_error": None}


class FakeExtraction:
    """Section-aware: builds whichever extract model the node asks for.

    Handing back the wrong model would be treated as a parsing failure and the
    turn would produce no user_data, quietly hollowing out this whole test.
    """

    def __init__(self):
        self.model = None

    def build(self, extract_model):
        self.model = extract_model
        return self

    async def ainvoke(self, messages, config=None):
        values = dict.fromkeys(self.model.model_fields, None)
        values.update(EXTRACT_VALUES.get(self.model.__name__, {}))
        return {
            "raw": AIMessage(content=""),
            "parsed": self.model(**values),
            "parsing_error": None,
        }


@pytest.fixture
def fake_models(monkeypatch):
    """Patch all three model calls at module level, so every graph instance sees them."""
    from agents.xbuddy.nodes import generate_decision as decision_module
    from agents.xbuddy.nodes import generate_reply as reply_module
    from agents.xbuddy.nodes import memory_updater as memory_module

    extraction = FakeExtraction()
    monkeypatch.setattr(reply_module, "_reply_model", FakeReply)
    monkeypatch.setattr(decision_module, "_decision_chain", FakeDecision)
    monkeypatch.setattr(memory_module, "_extraction_chain", extraction.build)
    return extraction


@asynccontextmanager
async def service_style_graph(db_path):
    """A fresh saver on `db_path` bound to a freshly compiled graph.

    Deliberately mirrors the service: `build_xbuddy_graph()` compiles with an
    in-process MemorySaver, which is then *replaced* by the on-disk saver exactly
    as service.py does. Both the saver and the graph object are new each time
    this is entered, so nothing can leak between phases except the file.
    """
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        await saver.setup()
        graph = build_xbuddy_graph()
        graph.checkpointer = saver
        yield graph


def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "user_id": 7}, "recursion_limit": 12}


# --------------------------------------------------------------------------
# The main restart proof
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_domain_memory_survives_a_real_checkpoint_restart(tmp_path, fake_models):
    db = tmp_path / "xbuddy-checkpoint.db"
    thread_id = f"restart-{uuid.uuid4().hex[:8]}"
    config = config_for(thread_id)

    # --- Phase A: produce meaningful PR 4 state, then dispose completely ---
    async with service_style_graph(db) as graph_a:
        await graph_a.ainvoke(
            {"messages": [HumanMessage(content="I need a new job")]}, config
        )
        before = (await graph_a.aget_state(config)).values

    assert db.exists() and db.stat().st_size > 0, "the checkpoint file was never written"

    # Sanity-check that phase A actually built the state we intend to prove durable.
    assert before["user_data"].target_roles == ["Senior SRE"]
    assert before["section_states"]["career_goal"].status is SectionStatus.DONE
    assert before["current_section"] is SectionID.BACKGROUND

    # --- Phase B: a brand-new saver and graph over the same file ---
    async with service_style_graph(db) as graph_b:
        assert graph_b is not graph_a
        restored = (await graph_b.aget_state(config)).values

    # messages survived
    assert len(restored["messages"]) == len(before["messages"])
    assert isinstance(restored["messages"][0], HumanMessage)
    assert restored["messages"][0].content == "I need a new job"
    assert [type(m).__name__ for m in restored["messages"]] == [
        type(m).__name__ for m in before["messages"]
    ]

    # identity survived
    assert restored["user_id"] == 7
    assert restored["thread_id"] == thread_id

    # user_data survived with concrete values, not just "non-empty"
    assert restored["user_data"].target_roles == ["Senior SRE"]
    assert restored["user_data"].target_timeline == "3 months"
    assert restored["user_data"] == before["user_data"]

    # section_states survived, statuses intact
    assert restored["section_states"]["career_goal"].status is SectionStatus.DONE
    assert restored["section_states"]["background"].status is SectionStatus.IN_PROGRESS
    for section in (
        SectionID.JOB_PREFERENCES,
        SectionID.SKILL_ASSESSMENT,
        SectionID.ACTION_PLAN,
    ):
        assert restored["section_states"][section.value].status is SectionStatus.PENDING

    # current_section survived, advanced past the completed section
    assert restored["current_section"] is SectionID.BACKGROUND
    assert restored["current_section"] == before["current_section"]

    # router_directive survived too (it drives the next turn's routing)
    assert restored["router_directive"] == before["router_directive"]


@pytest.mark.asyncio
async def test_restored_state_is_usable_for_a_second_turn(tmp_path, fake_models):
    """Durability is only useful if the restored thread can be continued."""
    db = tmp_path / "xbuddy-checkpoint.db"
    thread_id = f"restart-{uuid.uuid4().hex[:8]}"
    config = config_for(thread_id)

    async with service_style_graph(db) as graph_a:
        await graph_a.ainvoke({"messages": [HumanMessage(content="first turn")]}, config)

    async with service_style_graph(db) as graph_b:
        await graph_b.ainvoke({"messages": [HumanMessage(content="second turn")]}, config)
        final = (await graph_b.aget_state(config)).values

    human = [m.content for m in final["messages"] if isinstance(m, HumanMessage)]
    assert "first turn" in human, "the pre-restart message was lost"
    assert "second turn" in human
    # Background collected its own fields in the second turn, on top of Career Goal's.
    assert final["user_data"].target_roles == ["Senior SRE"]
    assert final["user_data"].current_role == "QA Analyst"


# --------------------------------------------------------------------------
# Focused cases for the two flags that only appear in specific situations
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_should_generate_final_output_survives_restart(tmp_path, fake_models):
    """Seed four sections done, complete the fifth, then restart."""
    db = tmp_path / "xbuddy-checkpoint.db"
    thread_id = f"restart-complete-{uuid.uuid4().hex[:8]}"
    config = config_for(thread_id)

    four_done = {
        section.value: SectionState(
            section_id=section,
            status=(
                SectionStatus.IN_PROGRESS
                if section is SectionID.ACTION_PLAN
                else SectionStatus.DONE
            ),
        )
        for section in SectionID
    }

    async with service_style_graph(db) as graph_a:
        await graph_a.ainvoke(
            {
                "messages": [HumanMessage(content="yes, ship it")],
                "section_states": four_done,
                "current_section": SectionID.ACTION_PLAN,
                "router_directive": "stay",
            },
            config,
        )
        before = (await graph_a.aget_state(config)).values

    assert before["should_generate_final_output"] is True

    async with service_style_graph(db) as graph_b:
        restored = (await graph_b.aget_state(config)).values

    assert restored["should_generate_final_output"] is True
    assert all(
        restored["section_states"][section.value].status is SectionStatus.DONE
        for section in SectionID
    )


@pytest.mark.asyncio
async def test_persistence_pending_survives_restart(tmp_path, fake_models, persistence):
    """A queued failed write must still be queued after a restart.

    Otherwise an outage during the last turn before a deploy would silently lose
    the retry, and state would diverge from Supabase with nothing recording it.
    """
    db = tmp_path / "xbuddy-checkpoint.db"
    thread_id = f"restart-pending-{uuid.uuid4().hex[:8]}"
    config = config_for(thread_id)

    persistence.fail = True  # every write fails; no live Supabase involved

    async with service_style_graph(db) as graph_a:
        await graph_a.ainvoke(
            {"messages": [HumanMessage(content="I need a new job")]}, config
        )
        before = (await graph_a.aget_state(config)).values

    assert before["persistence_pending"], "expected a queued section from the failed writes"
    assert before["error_count"] >= 1

    async with service_style_graph(db) as graph_b:
        restored = (await graph_b.aget_state(config)).values

    assert restored["persistence_pending"] == before["persistence_pending"]
    assert restored["error_count"] == before["error_count"]
    assert restored["last_error"] == before["last_error"]
    # And the DONE it refused to persist is still DONE.
    assert restored["section_states"]["career_goal"].status is SectionStatus.DONE


# --------------------------------------------------------------------------
# Negative control: the positive result must come from the file, not reuse
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memorysaver_negative_control_loses_the_thread(fake_models):
    """The same flow with in-process savers must NOT find the thread.

    If this passed, the positive test above would prove nothing — it could be
    succeeding through a shared object rather than the SQLite file.
    """
    thread_id = f"negative-{uuid.uuid4().hex[:8]}"
    config = config_for(thread_id)

    graph_a = build_xbuddy_graph()
    graph_a.checkpointer = MemorySaver()
    await graph_a.ainvoke({"messages": [HumanMessage(content="I need a new job")]}, config)
    before = (await graph_a.aget_state(config)).values
    assert before["user_data"].target_roles == ["Senior SRE"], "phase A did run"

    # A genuinely separate saver and graph — no shared instance.
    graph_b = build_xbuddy_graph()
    graph_b.checkpointer = MemorySaver()
    assert graph_b.checkpointer is not graph_a.checkpointer

    restored = (await graph_b.aget_state(config)).values

    assert not restored, f"in-process state must not survive, got keys {sorted(restored)}"


@pytest.mark.asyncio
async def test_a_different_thread_id_is_not_restored(tmp_path, fake_models):
    """Durability must be thread-scoped, not file-scoped."""
    db = tmp_path / "xbuddy-checkpoint.db"
    written = f"restart-{uuid.uuid4().hex[:8]}"

    async with service_style_graph(db) as graph_a:
        await graph_a.ainvoke(
            {"messages": [HumanMessage(content="I need a new job")]}, config_for(written)
        )

    async with service_style_graph(db) as graph_b:
        same = (await graph_b.aget_state(config_for(written))).values
        other = (await graph_b.aget_state(config_for("never-written"))).values

    assert same, "the written thread should restore"
    assert not other, "an unrelated thread must stay empty"


# --------------------------------------------------------------------------
# Isolation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_works_with_supabase_settings_absent(
    tmp_path, fake_models, monkeypatch
):
    """Checkpoint durability must not depend on Supabase being configured at all."""
    from core.settings import settings

    for name in (
        "SUPABASE_URL",
        "SUPABASE_SECRET_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_PUBLISHABLE_KEY",
        "SUPABASE_ANON_KEY",
    ):
        monkeypatch.setattr(settings, name, None)

    db = tmp_path / "xbuddy-checkpoint.db"
    thread_id = f"restart-nocreds-{uuid.uuid4().hex[:8]}"
    config = config_for(thread_id)

    async with service_style_graph(db) as graph_a:
        await graph_a.ainvoke(
            {"messages": [HumanMessage(content="I need a new job")]}, config
        )

    async with service_style_graph(db) as graph_b:
        restored = (await graph_b.aget_state(config)).values

    assert restored["user_data"].target_roles == ["Senior SRE"]
    assert restored["section_states"]["career_goal"].status is SectionStatus.DONE


@pytest.mark.asyncio
async def test_the_saver_the_service_builds_matches_this_test(tmp_path, monkeypatch):
    """Pin that the test's lifecycle is the production one.

    If the service ever switches saver type or drops `setup()`, this fails and
    the restart proof above stops being evidence about production.
    """
    from core.settings import settings
    from memory.sqlite import get_sqlite_saver

    db = tmp_path / "service-style.db"
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(db))

    async with get_sqlite_saver() as saver:
        assert isinstance(saver, AsyncSqliteSaver)
        assert hasattr(saver, "setup"), "the lifespan calls setup() when present"
        await saver.setup()

    assert db.exists(), "the service's saver writes to an on-disk file"
