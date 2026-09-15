"""Shared fixtures for the PR 3 node tests.

The two generation nodes each resolve their model through one module-level
helper (`_reply_model` / `_decision_chain`). Tests patch that helper with a
recording fake, so nothing here touches the network and every test can assert
both *what* the model was sent and *whether it was called at all* — the latter
matters for the two short-circuit guards.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy.context import build_context_packet
from agents.xbuddy.enums import DecisionAction, SectionID, SectionStatus
from agents.xbuddy.models import SectionDecision, XBuddyData
from agents.xbuddy.state_factory import build_initial_state


@pytest.fixture(autouse=True)
def no_unmocked_model_calls(monkeypatch):
    """A fake API key is not isolation: block any forgotten model seam locally.

    Assert at teardown too, since production nodes intentionally catch failures.
    Tests that exercise model construction may replace this with their own fake.
    """
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Unmocked model call: supply a deterministic test double")

    monkeypatch.setattr("core.llm.get_model", blocked)
    yield
    assert not attempts, "A node attempted an unmocked model call (blocked locally)"


class RecordingModel:
    """Stands in for the chat model in generate_reply."""

    def __init__(self, reply: str = "What kind of role are you targeting next?"):
        self.reply = reply
        self.calls: list[list[Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1][0].content

    async def ainvoke(self, messages, config=None):
        self.calls.append(list(messages))
        return AIMessage(content=self.reply)


class RecordingChain:
    """Stands in for the structured-output chain in generate_decision.

    Returns the include_raw shape: {"raw", "parsed", "parsing_error"}.
    """

    def __init__(
        self,
        decision: SectionDecision | None = None,
        parsing_error: Exception | None = None,
        raises: Exception | None = None,
    ):
        self.decision = decision
        self.parsing_error = parsing_error
        self.raises = raises
        self.calls: list[list[Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1][0].content

    async def ainvoke(self, messages, config=None):
        self.calls.append(list(messages))
        if self.raises is not None:
            raise self.raises
        return {
            "raw": AIMessage(content=""),
            "parsed": self.decision,
            "parsing_error": self.parsing_error,
        }


def _make_decision(**overrides) -> SectionDecision:
    """A valid SectionDecision with every required field supplied.

    The real schema has no defaults (strict json_schema requires all 7 fields in
    `required`), so tests must pass them all — this keeps that explicit.
    """
    values: dict[str, Any] = {
        "action": DecisionAction.STAY,
        "modify_target": None,
        "is_satisfied": None,
        "user_satisfaction_feedback": None,
        "should_save_content": False,
        "presented_summary": False,
        "decision_reason": "target_roles still empty",
    }
    values.update(overrides)
    return SectionDecision(**values)


def _make_state(
    *,
    section: SectionID = SectionID.CAREER_GOAL,
    messages: list | None = None,
    user_data: XBuddyData | None = None,
    with_packet: bool = True,
    **overrides,
) -> dict:
    """A realistic post-router state, as generate_reply/decision would receive it."""
    state = build_initial_state(user_id=7, thread_id="t-pr3")
    state["messages"] = (
        messages if messages is not None else [HumanMessage(content="I need a new job")]
    )
    state["current_section"] = section
    state["user_data"] = user_data or XBuddyData()
    state["context_packet"] = (
        build_context_packet(
            section_id=section,
            status=SectionStatus.IN_PROGRESS,
            user_data=state["user_data"],
        )
        if with_packet
        else None
    )
    state.update(overrides)
    return state


@pytest.fixture
def make_state():
    """Factory fixture — `tests/` is not a package, so helpers arrive this way."""
    return _make_state


@pytest.fixture
def make_decision():
    """Factory for a fully-populated SectionDecision (the schema has no defaults)."""
    return _make_decision


@pytest.fixture
def reply_model(monkeypatch):
    """Patch generate_reply's model and hand the fake back to the test."""
    from agents.xbuddy.nodes import generate_reply as module

    fake = RecordingModel()
    monkeypatch.setattr(module, "_reply_model", lambda: fake)
    return fake


class PersistenceRecorder:
    """Replaces `persist_section` in memory_updater for every agent test.

    Callable with the real signature, records each attempt, and returns whatever
    the test configured. `fail_sections` fails only the named sections so retry
    ordering can be observed; `fail` fails everything.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.fail = False
        self.fail_sections: set[str] = set()

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def sections(self) -> list[str]:
        """Section values in the exact order they were attempted."""
        return [call["section_id"] for call in self.calls]

    async def __call__(self, user_id, thread_id, section_id, section, user_data):
        value = section_id.value if hasattr(section_id, "value") else str(section_id)
        self.calls.append(
            {
                "user_id": user_id,
                "thread_id": thread_id,
                "section_id": value,
                "status": section.status.value,
                "user_data": user_data,
            }
        )
        return not (self.fail or value in self.fail_sections)


class FinalOutputRecorder:
    """Replaces `persist_final_output` for every agent test.

    Mirrors the real `(persisted, reason)` contract so a test can make the durable
    write succeed, fail, or refuse without a database. `refuse_reason` reproduces the
    user-edit refusal specifically, because the node branches on it: a refusal is not
    retried, a failure is.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.fail = False
        self.refuse_reason: str | None = None

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def markdowns(self) -> list[str]:
        return [call["markdown"] for call in self.calls]

    async def __call__(self, user_id, thread_id, markdown):
        self.calls.append(
            {"user_id": user_id, "thread_id": thread_id, "markdown": markdown}
        )
        if self.refuse_reason is not None:
            return False, self.refuse_reason
        if self.fail:
            return False, "final output write failed: fake"
        return True, None


class StaleMarkRecorder:
    """Replaces `mark_final_output_stale`, same `(marked, reason)` contract."""

    def __init__(self):
        self.calls: list[dict] = []
        self.fail = False

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def __call__(self, user_id, thread_id):
        self.calls.append({"user_id": user_id, "thread_id": thread_id})
        if self.fail:
            return False, "final output stale marking failed: fake"
        return True, None


@pytest.fixture(autouse=True)
def persistence(monkeypatch):
    """Autouse guard: no agent test may reach the live Supabase project.

    `Settings` reads `.env` through `env_file=find_dotenv()`, so real credentials
    are visible during tests. Without this fixture, any test exercising
    memory_updater writes rows to the real database — which happened once during
    PR 4 Stage 5 development and left four stray rows behind.

    Covers **every** durable writer, not only sections. PR 5 added
    `persist_final_output` and `mark_final_output_stale`, and the first PR 5 test run
    reached for a real client through them precisely because this fixture patched
    only `persist_section` — the same gap, caught the same way.

    Tests that want to observe or steer persistence request `persistence`,
    `final_output_persistence`, or `stale_marker` by name; everything else gets the
    guard for free.
    """
    from agents.xbuddy.nodes import implementation as implementation_module
    from agents.xbuddy.nodes import memory_updater as memory_module

    recorder = PersistenceRecorder()
    monkeypatch.setattr(memory_module, "persist_section", recorder)
    monkeypatch.setattr(
        implementation_module, "persist_final_output", FinalOutputRecorder()
    )
    monkeypatch.setattr(
        memory_module, "mark_final_output_stale", StaleMarkRecorder(), raising=False
    )
    return recorder


@pytest.fixture
def final_output_persistence(monkeypatch):
    """The durable final-output writer, steerable and observable."""
    from agents.xbuddy.nodes import implementation as module

    recorder = FinalOutputRecorder()
    monkeypatch.setattr(module, "persist_final_output", recorder)
    return recorder


@pytest.fixture
def stale_marker(monkeypatch):
    """The durable stale-marking call, steerable and observable."""
    from agents.xbuddy.nodes import memory_updater as module

    recorder = StaleMarkRecorder()
    monkeypatch.setattr(module, "mark_final_output_stale", recorder, raising=False)
    return recorder


class RecordingExtractionChain:
    """Stands in for the structured-output chain in memory_updater.

    Returns the include_raw shape. `_extraction_chain` takes the section's model
    as an argument, so the fake records which schema it was built for — that is
    how the section-scoping test asserts the right model was selected.
    """

    def __init__(self):
        self.extracted: Any = None
        self.parsing_error: Exception | None = None
        self.raises: Exception | None = None
        self.models: list[type] = []
        self.calls: list[list[Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_model(self) -> type:
        return self.models[-1]

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1][0].content

    def build(self, extract_model):
        """Mimics `_extraction_chain(extract_model)` returning a runnable."""
        self.models.append(extract_model)
        return self

    async def ainvoke(self, messages, config=None):
        self.calls.append(list(messages))
        if self.raises is not None:
            raise self.raises
        return {
            "raw": AIMessage(content=""),
            "parsed": self.extracted,
            "parsing_error": self.parsing_error,
        }


@pytest.fixture
def extraction_chain(monkeypatch):
    """Patch memory_updater's chain builder; tests set `.extracted` / `.raises`."""
    from agents.xbuddy.nodes import memory_updater as module

    fake = RecordingExtractionChain()
    monkeypatch.setattr(module, "_extraction_chain", fake.build)
    return fake


@pytest.fixture
def decision_chain(monkeypatch):
    """Patch generate_decision's chain; tests set `.decision` / `.raises`."""
    from agents.xbuddy.nodes import generate_decision as module

    fake = RecordingChain(decision=_make_decision())
    monkeypatch.setattr(module, "_decision_chain", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def no_real_resume_store(monkeypatch):
    """Autouse guard: no test may reach the live Supabase project through Resume RAG.

    The same hazard `persistence` exists for — `.env` credentials are visible to
    tests — applied to the resume store, whose `replace` writes real rows. Every
    unit test injects a fake client; this catches the one that forgets.

    Yields the list of blocked attempts, asserted empty at teardown, because the
    interactive retrieval path swallows exceptions by design and would otherwise
    hide a forgotten injection behind an empty result.
    """
    from agents.xbuddy.resume import store as store_module

    attempts: list[str] = []

    def blocked():
        attempts.append("resume store client")
        raise AssertionError("Unmocked Supabase client in Resume RAG: inject a fake client")

    monkeypatch.setattr(store_module, "_default_client", blocked)
    yield attempts
    assert not attempts, "A test reached for the real Supabase client through the resume store"


class ResumeBackend:
    """Stands in for the resume store and retrieval as the router sees them.

    Defaults to "no resume on file", so every graph test that does not ask for
    resume behaviour runs exactly the pre-Resume-RAG path. Resume tests request
    `resume_backend` by name and set `status` / `evidence` / failure flags.
    """

    def __init__(self) -> None:
        self.status = None
        self.evidence: list = []
        self.fail_status = False
        self.fail_retrieve = False
        self.status_calls: list[tuple[int, str]] = []
        self.retrieve_calls: list[tuple[int, str, str, int]] = []

    async def fetch_status(self, user_id, thread_id):
        self.status_calls.append((user_id, thread_id))
        if self.fail_status:
            raise RuntimeError("resume status unavailable (test)")
        return self.status

    async def retrieve(self, user_id, thread_id, query, k):
        # The production helper never raises (it returns [] on any failure);
        # `fail_retrieve` checks the router does not depend on that.
        self.retrieve_calls.append((user_id, thread_id, query, k))
        if self.fail_retrieve:
            raise RuntimeError("resume retrieval unavailable (test)")
        return list(self.evidence)


@pytest.fixture(autouse=True)
def resume_backend(monkeypatch):
    """Autouse: the router's resume lookup and retrieval never leave the process."""
    from agents.xbuddy.resume import context as resume_context

    backend = ResumeBackend()
    monkeypatch.setattr(resume_context, "_fetch_status", backend.fetch_status)
    monkeypatch.setattr(resume_context, "_retrieve", backend.retrieve)
    monkeypatch.setattr(resume_context, "resume_rag_configured", lambda: True)
    return backend
