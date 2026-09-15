"""Resume RAG inside the JobBuddy workflow: prompts, caching, and the confirmation gate.

The invariants pinned here:
- no resume -> the system prompt is byte-identical to the pre-Resume-RAG prompt;
- resume material is its own block, after KNOWN SO FAR, never inside it;
- Background proposes only candidate facts the user has not already confirmed;
- Skill Assessment gets top-3 retrieved evidence, Summary excluded, only there;
- nothing but memory_updater's extraction of the conversation writes user_data;
- the router's double pass does one status lookup and one retrieval, not two;
- every failure degrades to "no resume context" and the turn continues.

Offline: the autouse `resume_backend` fixture stands in for the store and
retrieval; fakes stand in for every model.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy.context import build_context_packet, build_system_prompt, render_known_data
from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import BackgroundExtract, ResumeContext, ResumeEvidence, XBuddyData
from agents.xbuddy.nodes.memory_updater import memory_updater_node
from agents.xbuddy.nodes.router import router_node
from agents.xbuddy.prompts import get_section_template
from agents.xbuddy.resume import context as resume_context
from agents.xbuddy.resume.context import (
    SKILL_EVIDENCE_K,
    evidence_key,
    render_resume_block,
    resolve_resume_context,
    skill_query,
)
from agents.xbuddy.resume.models import ResumeSection
from agents.xbuddy.resume.store import ResumeStatus, RetrievedChunk
from agents.xbuddy.sections.base_prompt import BASE_RULES
from agents.xbuddy.state_factory import build_initial_state, build_section_states

CANDIDATES = {
    "current_role": "Senior Backend Engineer, Northwind Logistics",
    "years_experience": 8,
    "highest_education": "BASc Computer Engineering, University of Waterloo",
    "work_history": ["Senior Backend Engineer, Northwind (2020-now)", "Backend Engineer, Brightpath (2017-2019)"],
}
BACKGROUND_HEADER = "FROM YOUR RESUME — NOT CONFIRMED"
EVIDENCE_HEADER = "RESUME EVIDENCE — NOT USER-CONFIRMED"


def status(document_id="doc-1", facts=CANDIDATES):
    return ResumeStatus(document_id=document_id, filename="cv.pdf", page_count=2, chunk_count=12,
                        embedding_model="text-embedding-3-small", content_sha256="a" * 64,
                        created_at="2026-09-11T12:00:00+00:00", candidate_facts=facts)


def chunk(i, text, section=ResumeSection.PROJECTS, similarity=0.8):
    return RetrievedChunk(id=f"c{i}", document_id="doc-1", chunk_index=i, section=section,
                          content=text, token_count=20, page=1, similarity=similarity)


EVIDENCE = [
    chunk(5, "[Projects] JobBuddy — deployed the backend to Fly.io and the Next.js frontend to Vercel."),
    chunk(2, "[Experience] Designed an event pipeline on Kafka and PostgreSQL.", ResumeSection.EXPERIENCE),
    chunk(8, "[Skills] Languages: Python, Go, SQL", ResumeSection.SKILLS, 0.6),
]


def graph_state(section, *, user_data=None, messages=None, **extra):
    state = build_initial_state(user_id=7, thread_id="thread-rag")
    sections = build_section_states()
    for done in SectionID:
        if done is section:
            break
        sections[done.value] = sections[done.value].model_copy(update={"status": SectionStatus.DONE})
    state.update({
        "current_section": section,
        "router_directive": "stay",
        "section_states": sections,
        "user_data": user_data or XBuddyData(target_roles=["AI Engineer"]),
        "messages": messages if messages is not None else [HumanMessage(content="ok", id="h1")],
    })
    state.update(extra)
    return state


async def route(state):
    update = await router_node(state, {"configurable": {}})
    return update, {**state, **update}


# --------------------------------------------------------------------------
# Prompt assembly: no resume means no change at all
# --------------------------------------------------------------------------


def pre_resume_prompt(template, user_data):
    """The system prompt exactly as composed before Resume RAG (ab41dce), kept here
    verbatim as an independent oracle. Comparing build_system_prompt with itself
    could not catch a change to it."""
    parts = [BASE_RULES.strip(), template.system_prompt_template.strip()]
    known = render_known_data(user_data) if user_data is not None else ""
    if known:
        parts.append(f"KNOWN SO FAR\n{known}")
    else:
        parts.append("KNOWN SO FAR\nNothing collected yet — this is the start of the conversation.")
    return "\n\n".join(parts)


@pytest.mark.parametrize("section", list(SectionID))
@pytest.mark.parametrize("data", [None, XBuddyData(), XBuddyData(target_roles=["AI Engineer"])])
@pytest.mark.parametrize("block", [None, ""])
def test_no_resume_block_leaves_the_prompt_byte_identical(section, data, block):
    template = get_section_template(section)
    expected = pre_resume_prompt(template, data)
    assert build_system_prompt(template, data, block) == expected
    assert build_system_prompt(template, data) == expected


def test_the_block_is_appended_after_known_so_far_and_not_inside_it():
    """Everything before the block is the pre-Resume-RAG prompt, byte for byte;
    the block is last. So KNOWN SO FAR cannot contain resume material."""
    data = XBuddyData(target_roles=["AI Engineer"])
    template = get_section_template(SectionID.BACKGROUND)
    block = render_resume_block(SectionID.BACKGROUND, ResumeContext(document_id="d", candidate_facts=CANDIDATES), data)
    prompt = build_system_prompt(template, data, block)
    assert prompt == pre_resume_prompt(template, data) + "\n\n" + block.strip()
    known_section = prompt[prompt.index("KNOWN SO FAR\n- Target role") : prompt.index(BACKGROUND_HEADER)]
    assert "Northwind" not in known_section and "Waterloo" not in known_section
    assert render_known_data(data) == "- Target role(s): AI Engineer"  # unchanged by the resume


# --------------------------------------------------------------------------
# Background candidate facts
# --------------------------------------------------------------------------


def background_block(user_data=None, facts=CANDIDATES):
    return render_resume_block(
        SectionID.BACKGROUND, ResumeContext(document_id="d", candidate_facts=facts), user_data or XBuddyData()
    )


def test_background_proposes_every_unconfirmed_candidate():
    block = background_block()
    assert block.startswith(BACKGROUND_HEADER)
    for value in ("Northwind Logistics", "8", "University of Waterloo", "Brightpath (2017-2019)"):
        assert value in block


def test_a_confirmed_field_is_never_re_proposed():
    block = background_block(XBuddyData(current_role="Staff Engineer, Acme"))
    assert "Northwind Logistics" not in block  # the resume's role is not re-proposed
    assert "Staff Engineer" not in block      # nor is the confirmed value restated
    assert "University of Waterloo" in block


def test_nothing_is_proposed_once_background_is_confirmed():
    confirmed = XBuddyData(current_role="r", years_experience=3, highest_education="e", work_history=["w"])
    assert background_block(confirmed) is None


def test_unstated_years_are_not_proposed_and_must_be_asked_for():
    block = background_block(facts={**CANDIDATES, "years_experience": None})
    assert "Years of experience" not in block
    assert "never calculate years of experience from dates" in " ".join(block.split())


def test_background_block_needs_a_resume_and_candidates():
    assert render_resume_block(SectionID.BACKGROUND, None, XBuddyData()) is None
    assert render_resume_block(SectionID.BACKGROUND, ResumeContext(document_id=None), XBuddyData()) is None
    assert render_resume_block(SectionID.BACKGROUND, ResumeContext(document_id="d"), XBuddyData()) is None


@pytest.mark.parametrize("section", [SectionID.CAREER_GOAL, SectionID.JOB_PREFERENCES, SectionID.ACTION_PLAN])
def test_other_sections_never_get_resume_material(section):
    data = XBuddyData(target_roles=["AI Engineer"])
    context = ResumeContext(
        document_id="doc-1", candidate_facts=CANDIDATES,
        evidence=[ResumeEvidence(chunk_index=5, section="projects", content="x", similarity=0.9)],
        evidence_key=evidence_key("doc-1", SectionID.SKILL_ASSESSMENT, skill_query(data)),
    )
    assert render_resume_block(section, context, data) is None


# --------------------------------------------------------------------------
# Skill Assessment evidence
# --------------------------------------------------------------------------


def skill_context(data, evidence=None, key_section=SectionID.SKILL_ASSESSMENT):
    items = evidence if evidence is not None else [
        ResumeEvidence(chunk_index=c.chunk_index, section=c.section.value, content=c.content, similarity=c.similarity)
        for c in EVIDENCE
    ]
    return ResumeContext(document_id="doc-1", evidence=items,
                         evidence_key=evidence_key("doc-1", key_section, skill_query(data)))


def test_skill_assessment_shows_the_retrieved_passages():
    data = XBuddyData(target_roles=["AI Engineer"])
    block = render_resume_block(SectionID.SKILL_ASSESSMENT, skill_context(data), data)
    assert block.startswith(EVIDENCE_HEADER)
    assert "[1] [Projects] JobBuddy — deployed the backend to Fly.io" in block
    assert "[3] [Skills] Languages: Python, Go, SQL" in block
    assert "Do not infer skills from job titles" in block


def test_a_summary_passage_can_never_appear():
    data = XBuddyData(target_roles=["AI Engineer"])
    evidence = [ResumeEvidence(chunk_index=1, section="summary", content="[Summary] everything", similarity=0.99),
                ResumeEvidence(chunk_index=5, section="projects", content="[Projects] real", similarity=0.8)]
    block = render_resume_block(SectionID.SKILL_ASSESSMENT, skill_context(data, evidence), data)
    assert "[Summary]" not in block and "[1] [Projects] real" in block


def test_evidence_for_an_old_query_is_not_shown():
    old = XBuddyData(target_roles=["Data Analyst"])
    new = XBuddyData(target_roles=["AI Engineer"])
    assert render_resume_block(SectionID.SKILL_ASSESSMENT, skill_context(old), new) is None


def test_the_query_names_the_section_purpose_and_confirmed_roles():
    assert skill_query(XBuddyData()) == "Evidence of technical skills, tools, projects, and measurable achievements"
    assert skill_query(XBuddyData(target_roles=["AI Engineer", "ML Engineer"])).endswith(
        "relevant to AI Engineer, ML Engineer"
    )


# --------------------------------------------------------------------------
# Resolution and caching
# --------------------------------------------------------------------------


async def resolve(backend, section, *, messages, cached=None, data=None):
    return await resolve_resume_context(
        user_id=7, thread_id="thread-rag", section=section, user_data=data or XBuddyData(target_roles=["AI Engineer"]),
        messages=messages, cached=cached, fetch_status=backend.fetch_status, retrieve=backend.retrieve,
    )


H1 = [HumanMessage(content="one", id="h1")]
H2 = [*H1, AIMessage(content="reply"), HumanMessage(content="two", id="h2")]


@pytest.mark.asyncio
async def test_sections_without_resume_use_do_no_io(resume_backend):
    for section in (SectionID.CAREER_GOAL, SectionID.JOB_PREFERENCES, SectionID.ACTION_PLAN):
        result = await resolve(resume_backend, section, messages=H1)
        assert (result.context, result.changed) == (None, False)
    assert resume_backend.status_calls == [] and resume_backend.retrieve_calls == []


@pytest.mark.asyncio
async def test_the_feature_is_off_without_supabase_credentials(resume_backend, monkeypatch):
    monkeypatch.setattr(resume_context, "resume_rag_configured", lambda: False)
    result = await resolve(resume_backend, SectionID.BACKGROUND, messages=H1)
    assert result.context is None and resume_backend.status_calls == []


@pytest.mark.asyncio
async def test_no_resume_is_remembered_for_the_rest_of_the_message(resume_backend):
    first = await resolve(resume_backend, SectionID.BACKGROUND, messages=H1)
    assert first.changed and first.context.document_id is None
    again = await resolve(resume_backend, SectionID.BACKGROUND, messages=H1, cached=first.context)
    assert not again.changed and len(resume_backend.status_calls) == 1


@pytest.mark.asyncio
async def test_a_new_user_message_looks_again(resume_backend):
    first = await resolve(resume_backend, SectionID.BACKGROUND, messages=H1)
    resume_backend.status = status()  # uploaded between messages
    second = await resolve(resume_backend, SectionID.BACKGROUND, messages=H2, cached=first.context)
    assert second.context.document_id == "doc-1" and len(resume_backend.status_calls) == 2


@pytest.mark.asyncio
async def test_background_uses_candidates_and_never_retrieves(resume_backend):
    resume_backend.status = status()
    result = await resolve(resume_backend, SectionID.BACKGROUND, messages=H1)
    assert result.context.candidate_facts == CANDIDATES
    assert resume_backend.retrieve_calls == []


@pytest.mark.asyncio
async def test_skill_assessment_retrieves_top_three_once(resume_backend):
    resume_backend.status, resume_backend.evidence = status(), EVIDENCE
    first = await resolve(resume_backend, SectionID.SKILL_ASSESSMENT, messages=H1)
    (user, thread, query, k), = resume_backend.retrieve_calls
    assert (user, thread, k) == (7, "thread-rag", SKILL_EVIDENCE_K) and SKILL_EVIDENCE_K == 3
    assert query.endswith("relevant to AI Engineer")
    assert [e.chunk_index for e in first.context.evidence] == [5, 2, 8]

    # Same document, section, and query on a later message: status re-read, no re-retrieval.
    second = await resolve(resume_backend, SectionID.SKILL_ASSESSMENT, messages=H2, cached=first.context)
    assert len(resume_backend.retrieve_calls) == 1 and second.context.evidence == first.context.evidence


@pytest.mark.asyncio
async def test_a_replaced_resume_invalidates_the_cache(resume_backend):
    resume_backend.status, resume_backend.evidence = status("doc-1"), EVIDENCE
    first = await resolve(resume_backend, SectionID.SKILL_ASSESSMENT, messages=H1)
    resume_backend.status = status("doc-2", facts=None)
    second = await resolve(resume_backend, SectionID.SKILL_ASSESSMENT, messages=H2, cached=first.context)
    assert second.context.document_id == "doc-2"
    assert second.context.evidence_key.startswith("doc-2:")
    assert len(resume_backend.retrieve_calls) == 2
    assert second.context.candidate_facts is None  # nothing survives from the old document


@pytest.mark.asyncio
async def test_a_changed_target_role_retrieves_again(resume_backend):
    resume_backend.status, resume_backend.evidence = status(), EVIDENCE
    first = await resolve(resume_backend, SectionID.SKILL_ASSESSMENT, messages=H1)
    await resolve(resume_backend, SectionID.SKILL_ASSESSMENT, messages=H1, cached=first.context,
                  data=XBuddyData(target_roles=["Platform Engineer"]))
    assert len(resume_backend.retrieve_calls) == 2
    assert resume_backend.retrieve_calls[1][2].endswith("relevant to Platform Engineer")


@pytest.mark.asyncio
async def test_an_empty_retrieval_is_not_cached(resume_backend):
    """An empty result is almost always a failure; caching it would disable
    evidence for the rest of the section."""
    resume_backend.status, resume_backend.evidence = status(), []
    first = await resolve(resume_backend, SectionID.SKILL_ASSESSMENT, messages=H1)
    assert first.context.evidence_key is None
    resume_backend.evidence = EVIDENCE
    second = await resolve(resume_backend, SectionID.SKILL_ASSESSMENT, messages=H2, cached=first.context)
    assert second.context.evidence and len(resume_backend.retrieve_calls) == 2


@pytest.mark.asyncio
async def test_a_failed_status_lookup_degrades_and_leaves_state_alone(resume_backend):
    resume_backend.fail_status = True
    result = await resolve(resume_backend, SectionID.BACKGROUND, messages=H1)
    assert (result.context, result.changed) == (None, False)


@pytest.mark.asyncio
async def test_a_raising_retrieval_degrades_to_no_evidence_and_retries(resume_backend):
    resume_backend.status, resume_backend.evidence, resume_backend.fail_retrieve = status(), EVIDENCE, True
    first = await resolve(resume_backend, SectionID.SKILL_ASSESSMENT, messages=H1)
    assert first.context.document_id == "doc-1"
    assert first.context.evidence == [] and first.context.evidence_key is None
    assert render_resume_block(SectionID.SKILL_ASSESSMENT, first.context, XBuddyData(target_roles=["AI Engineer"])) is None
    resume_backend.fail_retrieve = False
    second = await resolve(resume_backend, SectionID.SKILL_ASSESSMENT, messages=H2, cached=first.context)
    assert second.context.evidence and len(resume_backend.retrieve_calls) == 2


@pytest.mark.asyncio
async def test_an_unreadable_cached_value_is_treated_as_absent(resume_backend):
    result = await resolve(resume_backend, SectionID.BACKGROUND, messages=H1, cached={"evidence": "garbage"})
    assert result.context.document_id is None and len(resume_backend.status_calls) == 1


# --------------------------------------------------------------------------
# Through the real router
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("section", [SectionID.BACKGROUND, SectionID.SKILL_ASSESSMENT])
async def test_without_a_resume_the_router_builds_the_old_prompt(section):
    state = graph_state(section)
    update, _ = await route(state)
    expected = build_context_packet(section_id=section, status=SectionStatus.IN_PROGRESS,
                                    user_data=state["user_data"]).system_prompt
    assert update["context_packet"].system_prompt == expected
    assert expected == pre_resume_prompt(get_section_template(section), state["user_data"])


@pytest.mark.asyncio
async def test_background_through_the_router(resume_backend):
    resume_backend.status = status()
    update, _ = await route(graph_state(SectionID.BACKGROUND))
    prompt = update["context_packet"].system_prompt
    assert BACKGROUND_HEADER in prompt and "Northwind Logistics" in prompt
    assert "user_data" not in update  # the router never records candidates


@pytest.mark.asyncio
async def test_skill_assessment_through_the_router(resume_backend):
    resume_backend.status, resume_backend.evidence = status(), EVIDENCE
    update, _ = await route(graph_state(SectionID.SKILL_ASSESSMENT))
    prompt = update["context_packet"].system_prompt
    assert EVIDENCE_HEADER in prompt and "deployed the backend to Fly.io" in prompt
    assert "user_data" not in update  # retrieved skills are not confirmed skills


@pytest.mark.asyncio
async def test_the_routers_second_pass_repeats_no_work(resume_backend):
    resume_backend.status, resume_backend.evidence = status(), EVIDENCE
    _, after_first = await route(graph_state(SectionID.SKILL_ASSESSMENT))
    second, _ = await route(after_first)
    assert len(resume_backend.status_calls) == 1
    assert len(resume_backend.retrieve_calls) == 1
    assert EVIDENCE_HEADER in second["context_packet"].system_prompt


@pytest.mark.asyncio
async def test_moving_on_from_skill_assessment_drops_the_evidence(resume_backend):
    resume_backend.status, resume_backend.evidence = status(), EVIDENCE
    _, after = await route(graph_state(SectionID.SKILL_ASSESSMENT))
    after["current_section"] = SectionID.ACTION_PLAN
    update, _ = await route(after)
    assert EVIDENCE_HEADER not in update["context_packet"].system_prompt


@pytest.mark.asyncio
async def test_failures_never_break_the_turn(resume_backend):
    resume_backend.fail_status = True
    update, _ = await route(graph_state(SectionID.SKILL_ASSESSMENT))
    assert EVIDENCE_HEADER not in update["context_packet"].system_prompt
    assert update["current_section"] is SectionID.SKILL_ASSESSMENT

    resume_backend.fail_status, resume_backend.status, resume_backend.evidence = False, status(), []
    update, _ = await route(graph_state(SectionID.SKILL_ASSESSMENT, messages=[HumanMessage(content="x", id="h9")]))
    assert EVIDENCE_HEADER not in update["context_packet"].system_prompt


# --------------------------------------------------------------------------
# The confirmation gate: only memory_updater's extraction writes user_data
# --------------------------------------------------------------------------


def background_turn(*, user_reply, user_data=None):
    data = user_data or XBuddyData(target_roles=["AI Engineer"])
    block = render_resume_block(SectionID.BACKGROUND, ResumeContext(document_id="d", candidate_facts=CANDIDATES), data)
    proposal = ("From your resume: Senior Backend Engineer at Northwind Logistics, 8 years, "
                "BASc from Waterloo. Is that right?")
    return graph_state(
        SectionID.BACKGROUND, user_data=data,
        messages=[AIMessage(content=proposal, id="a1"), HumanMessage(content=user_reply, id="h2")],
        context_packet=build_context_packet(section_id=SectionID.BACKGROUND, status=SectionStatus.IN_PROGRESS,
                                            user_data=data, resume_block=block),
    )


def extract(**fields):
    return BackgroundExtract(**{"current_role": None, "years_experience": None,
                                "highest_education": None, "work_history": None, **fields})


@pytest.mark.asyncio
async def test_confirmation_is_recorded_by_memory_updater(extraction_chain):
    extraction_chain.extracted = extract(current_role=CANDIDATES["current_role"], years_experience=8,
                                         highest_education=CANDIDATES["highest_education"],
                                         work_history=CANDIDATES["work_history"])
    update = await memory_updater_node(background_turn(user_reply="Yes, that's all right."), {"configurable": {}})
    data = update["user_data"]
    assert data.current_role == CANDIDATES["current_role"] and data.years_experience == 8


@pytest.mark.asyncio
async def test_a_correction_beats_the_resume(extraction_chain):
    extraction_chain.extracted = extract(current_role="Staff Engineer, Acme")
    update = await memory_updater_node(
        background_turn(user_reply="Mostly — but I'm a Staff Engineer at Acme now."), {"configurable": {}}
    )
    assert update["user_data"].current_role == "Staff Engineer, Acme"
    # And the next turn does not propose the resume's role again.
    block = background_block(update["user_data"])
    assert "Northwind Logistics" not in block.split("HOW TO USE THEM")[0]


@pytest.mark.asyncio
async def test_nothing_is_recorded_when_extraction_finds_no_agreement(extraction_chain):
    extraction_chain.extracted = extract()
    update = await memory_updater_node(background_turn(user_reply="Hmm, let me think."), {"configurable": {}})
    assert "user_data" not in update or update["user_data"] == XBuddyData(target_roles=["AI Engineer"])


@pytest.mark.asyncio
async def test_the_extraction_model_never_sees_the_resume_block(extraction_chain):
    """Extraction reads the conversation and EXTRACTION_RULES only. Candidate facts
    reach it solely as the assistant's proposal in the transcript — which is what
    lets the existing "not agreed -> null" rule gate them."""
    extraction_chain.extracted = extract()
    await memory_updater_node(background_turn(user_reply="yes"), {"configurable": {}})
    system = extraction_chain.last_system_prompt
    assert BACKGROUND_HEADER not in system and "University of Waterloo" not in system


@pytest.mark.asyncio
async def test_the_reply_model_sees_the_block(reply_model, resume_backend):
    from agents.xbuddy.nodes.generate_reply import generate_reply_node

    resume_backend.status = status()
    _, state = await route(graph_state(SectionID.BACKGROUND))
    reply_update = await generate_reply_node(state, {"configurable": {}})
    assert BACKGROUND_HEADER in reply_model.last_system_prompt
    assert "user_data" not in reply_update
