"""Issue #10 — `finished` means completion, not "the router saw a `next` directive".

The old mechanism: `router_node` set `finished = True` only inside the `NEXT` branch
of `_resolve_section`. Two consequences, and the second is the one that made it
unfixable from where it lived:

1. A fifth section completing on a `stay` turn left `finished` False.
2. The completing turn never reaches the router at all — `route_after_memory_updater`
   sends it to `implementation`, and `implementation` goes straight to `END`. So the
   node that owned the flag was structurally bypassed on exactly the turn that should
   have set it.

`finished` could therefore only flip on some *later* turn where the decision model
happened to emit `next` against an already-complete set of sections. It was a fact
about the directive, not about the conversation.

Now both `finished` and `should_generate_final_output` derive from one
`all_sections_complete` call inside `memory_updater`, which is the node that owns
DONE transitions and does run on the completing turn.

The public API is deliberately unchanged: `finished` still never crosses the
boundary. Clients read `collection_complete` and `artifact_available`.
"""

import pytest

from agents.xbuddy.enums import RouterDirective, SectionID, SectionStatus
from agents.xbuddy.models import SectionState
from agents.xbuddy.nodes.router import router_node
from agents.xbuddy.state_factory import all_sections_complete


def sections(**statuses: SectionStatus) -> dict[str, SectionState]:
    """All five sections, PENDING unless named."""
    return {
        section.value: SectionState(
            section_id=section,
            status=statuses.get(section.value, SectionStatus.PENDING),
        )
        for section in SectionID
    }


def all_done() -> dict[str, SectionState]:
    return sections(**{section.value: SectionStatus.DONE for section in SectionID})


def four_of_five() -> dict[str, SectionState]:
    done = {s.value: SectionStatus.DONE for s in SectionID}
    done[SectionID.ACTION_PLAN.value] = SectionStatus.IN_PROGRESS
    return sections(**done)


# --------------------------------------------------------------------------
# The rule itself
# --------------------------------------------------------------------------


def test_four_of_five_is_not_complete():
    assert all_sections_complete(four_of_five()) is False


def test_five_of_five_is_complete():
    assert all_sections_complete(all_done()) is True


def test_a_partial_mapping_is_not_vacuously_complete():
    """`all()` over `sections.values()` would call a one-section conversation done."""
    assert all_sections_complete({"career_goal": SectionState(
        section_id=SectionID.CAREER_GOAL, status=SectionStatus.DONE)}) is False


def test_an_empty_mapping_is_not_complete():
    assert all_sections_complete({}) is False


def test_a_checkpoint_restored_dict_entry_is_understood():
    """Restores can hand back plain dicts; the rule must not silently read them as
    incomplete, which would make a finished conversation look unfinished forever."""
    restored = {s.value: {"section_id": s.value, "status": "done"} for s in SectionID}
    assert all_sections_complete(restored) is True


def test_a_restored_dict_that_is_not_done_is_still_incomplete():
    restored = {s.value: {"section_id": s.value, "status": "done"} for s in SectionID}
    restored["action_plan"]["status"] = "in_progress"
    assert all_sections_complete(restored) is False


# --------------------------------------------------------------------------
# The directive can no longer decide completion
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "directive",
    [RouterDirective.STAY, RouterDirective.NEXT, "modify:background", "nonsense"],
)
async def test_the_router_never_writes_finished(directive):
    """Single writer. Whatever the directive, and whatever the sections say, the
    router must not be the thing that decides completion."""
    state = {
        "current_section": SectionID.ACTION_PLAN,
        "router_directive": directive,
        "section_states": all_done(),
        "finished": False,
        "messages": [],
    }
    update = await router_node(state, {})
    assert "finished" not in update, f"router wrote finished on directive {directive!r}"


@pytest.mark.asyncio
async def test_next_on_incomplete_sections_does_not_finish():
    """The inverse failure: a directive must not be able to declare completion."""
    state = {
        "current_section": SectionID.CAREER_GOAL,
        "router_directive": RouterDirective.NEXT,
        "section_states": four_of_five(),
        "finished": False,
        "messages": [],
    }
    update = await router_node(state, {})
    assert "finished" not in update
    assert update["current_section"] is SectionID.ACTION_PLAN, "routes to the unfinished one"


# --------------------------------------------------------------------------
# memory_updater is the single writer, and the two flags cannot drift
# --------------------------------------------------------------------------


def completion_fragment(section_states) -> dict:
    """Run just the completion rule the way memory_updater applies it."""
    complete = all_sections_complete(section_states)
    return {"should_generate_final_output": complete, "finished": complete}


def test_the_two_flags_are_computed_from_one_call():
    """Pinned at the source: both keys must come from `all_sections_complete`, not
    from two hand-rolled recounts that can disagree."""
    import inspect

    from agents.xbuddy.nodes import memory_updater

    source = inspect.getsource(memory_updater)
    assert source.count("all_sections_complete(sections)") == 2, "completion + demotion"
    # the old inline recount must be gone
    assert "entry.status is SectionStatus.DONE" not in source


@pytest.mark.parametrize("states,expected", [(four_of_five(), False), (all_done(), True)])
def test_flags_agree_for_every_completion_state(states, expected):
    fragment = completion_fragment(states)
    assert fragment["finished"] is expected
    assert fragment["should_generate_final_output"] is expected
    assert fragment["finished"] == fragment["should_generate_final_output"]


def test_memory_updater_writes_both_flags_together():
    """Neither may be written without the other, in either direction."""
    import inspect

    from agents.xbuddy.nodes import memory_updater

    source = inspect.getsource(memory_updater)
    assert 'update["should_generate_final_output"] = True' in source
    assert 'update["finished"] = True' in source
    assert 'fragment["should_generate_final_output"] = False' in source
    assert 'fragment["finished"] = False' in source


# --------------------------------------------------------------------------
# The public boundary is unchanged
# --------------------------------------------------------------------------


def test_public_completion_still_ignores_finished():
    """Fixing the internal semantics must not tempt anyone to publish the field."""
    from service.service import public_completion

    complete = public_completion(
        {"section_states": all_done(), "should_generate_final_output": True,
         "final_output": None, "finished": True}
    )
    assert "finished" not in complete.model_dump()
    assert complete.collection_complete is True
    assert complete.artifact_available is False


def test_public_completion_reads_the_flag_not_finished():
    """`collection_complete` tracks `should_generate_final_output`. Now that the two
    agree this is hard to observe, so it is forced apart deliberately."""
    from service.service import public_completion

    contradictory = public_completion(
        {"section_states": all_done(), "should_generate_final_output": False,
         "final_output": None, "finished": True}
    )
    assert contradictory.collection_complete is False, "must not read `finished`"


def test_finished_is_absent_from_every_public_model():
    from schema.schema import ChatHistory, CompletionState, InvokeResponse, PublicSection

    for model in (CompletionState, InvokeResponse, ChatHistory, PublicSection):
        assert "finished" not in model.model_fields


# --------------------------------------------------------------------------
# memory_updater_node, end to end — including the stale-checkpoint self-heal
# --------------------------------------------------------------------------


def satisfied_decision(make_decision):
    return make_decision(is_satisfied=True, presented_summary=True)


@pytest.mark.asyncio
async def test_the_completing_turn_sets_both_flags(make_state, make_decision, extraction_chain):
    """The fifth section reaching DONE is what flips completion — on the same turn,
    from the node that owns the transition."""
    states = sections(**{s.value: SectionStatus.DONE for s in SectionID
                         if s is not SectionID.ACTION_PLAN})
    from agents.xbuddy.nodes.memory_updater import memory_updater_node

    state = make_state(
        section=SectionID.ACTION_PLAN,
        section_states=states,
        agent_output=satisfied_decision(make_decision),
        finished=False,
        should_generate_final_output=False,
    )
    update = await memory_updater_node(state, {})

    assert update["should_generate_final_output"] is True
    assert update["finished"] is True


@pytest.mark.asyncio
async def test_a_stale_checkpoint_self_heals_without_a_migration(
    make_state, make_decision, extraction_chain
):
    """Issue #10's damage is already in production checkpoints: threads with all five
    sections DONE and `finished` stuck False.

    No migration is needed. `finished` is recomputed from the sections on the next
    turn that runs memory_updater, so a stale value cannot survive contact with the
    graph. This is the whole reason for deriving it rather than storing an
    independently-maintained flag.
    """
    from agents.xbuddy.nodes.memory_updater import memory_updater_node

    stale = make_state(
        section=SectionID.ACTION_PLAN,
        section_states=all_done(),
        agent_output=satisfied_decision(make_decision),
        finished=False,                      # the stale value
        should_generate_final_output=False,  # stale too
    )
    update = await memory_updater_node(stale, {})

    assert update["finished"] is True, "a stale False must not survive"
    assert update["should_generate_final_output"] is True


@pytest.mark.asyncio
async def test_an_incomplete_turn_does_not_claim_completion(
    make_state, make_decision, extraction_chain
):
    from agents.xbuddy.nodes.memory_updater import memory_updater_node

    state = make_state(
        section=SectionID.CAREER_GOAL,
        section_states=four_of_five(),
        agent_output=make_decision(is_satisfied=None),
        finished=False,
    )
    update = await memory_updater_node(state, {})
    assert update.get("finished") is not True
    assert update.get("should_generate_final_output") is not True


@pytest.mark.asyncio
async def test_a_failed_extraction_cannot_flip_completion(
    make_state, make_decision, extraction_chain
):
    """Degradation must never invent completion — the flags come from section state,
    not from whether the turn went well."""
    from agents.xbuddy.nodes.memory_updater import memory_updater_node

    extraction_chain.raises = RuntimeError("boom")
    state = make_state(
        section=SectionID.CAREER_GOAL,
        section_states=four_of_five(),
        agent_output=make_decision(is_satisfied=None),
        finished=False,
    )
    update = await memory_updater_node(state, {})
    assert update.get("finished") is not True
    assert update.get("should_generate_final_output") is not True
