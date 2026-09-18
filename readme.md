# JobBuddy

JobBuddy is a deployed AI career-planning application. It guides job seekers through a structured conversation, can use an optional resume as evidence, and turns confirmed user information into an actionable, durable career plan.

- [Live app](https://jobbuddy-plum.vercel.app)
- [Live API docs](https://jobbuddy-api.fly.dev/docs)
- [Video demo](https://www.loom.com/share/7dff808a896740649913255b4d05db42)

## Problem and intended user

### The problem

Job searching can be overwhelming, especially for people who want to make a career change but are unsure how to turn their background, skills, and goals into a clear plan. Career advice is often scattered across resumes, job postings, online resources, and generic recommendations. Users can struggle to connect evidence of their experience to a target role, understand their reported gaps, and decide what to do next.

### The intended user

JobBuddy is designed for job seekers who need more structure and personalized guidance during career planning. This includes recent graduates, professionals changing career paths, and candidates with relevant experience who are unsure how to position themselves for a new role. The ideal user has a general direction in mind but needs help organizing their experience, relating self-reported gaps to a target role, and turning that information into an actionable strategy.

## What JobBuddy does

JobBuddy guides users through five sections: Career Goal, Background, Job Preferences, Skill Assessment, and Action Plan. An optional PDF resume can be uploaded from the Welcome screen; its status also appears in the sidebar. The backend extracts candidate background facts and retrieves relevant resume evidence to propose strengths, especially during Skill Assessment. These are suggestions for the user to confirm or correct, not facts silently added to confirmed state.

The Action Plan has a proposal, revision, and separate confirmation step. Once the required sections are complete, JobBuddy synthesizes the confirmed information into a personalized Final Plan with priorities, next steps, and explicit unknowns. Conversation history, progress, resume status, and the Final Plan remain available after refresh or service restart. The interface streams responses and progress in real time, with a responsive drawer on mobile.

The result is a grounded planning aid. JobBuddy does not guarantee employment, independently verify every qualification, or make an automated hiring decision. The user remains responsible for reviewing the plan and deciding what to do.

## Architecture

JobBuddy uses a layered architecture that separates the user interface, server-side request handling, application API, agent orchestration, and persistence.

```text
Browser
   |
   v
Next.js UI and server routes
   |
   v
FastAPI API
   |
   v
LangGraph workflow
   +-----------------------> LLM calls
   +-----------------------> PostgreSQL checkpoints
   +-----------------------> Supabase application records and final artifact
   +-----------------------> Resume retrieval via Supabase pgvector
```

The Next.js frontend provides the five-section experience. The browser calls Next.js server routes, which attach server-only configuration and forward requests to FastAPI. The browser does not call FastAPI or Supabase directly, and the backend bearer token is never included in browser assets.

FastAPI is the main application API. It validates requests, applies authentication and request protection, loads the relevant conversation, and invokes the LangGraph workflow. The backend exposes synchronous invocation, Server-Sent Events (SSE) streaming, conversation history, public completion state, and final-output retrieval. The SSE path streams user-visible tokens, messages, and public progress events while suppressing internal decision, extraction, and synthesis output.

LangGraph manages the multi-step workflow. Its state records the active section, collected user data, completion and confirmation state, pending Action Plan proposals, and the next operation. Explicit graph state and deterministic routes control section transitions; the language model handles conversational generation, structured decisions, extraction, and grounded synthesis where those capabilities are useful. An assistant message is not itself proof that a section is complete: application state and a distinct user confirmation determine what can advance.

PostgreSQL checkpointing preserves LangGraph's operational execution state, including the conversation history and workflow data required to resume an interrupted thread. The checkpoint is the primary source for a live thread.

Supabase stores durable, product-facing section records, confirmed domain information, the rendered final artifact, and the indexed resume and its embeddings. These application records are separate from LangGraph checkpoints and are read through the backend API. This gives the product a stable data model without exposing LangGraph's internal checkpoint representation to application reads. The frontend restores transcript and section progress through backend read paths and checks resume status separately.

For an optional resume, the backend extracts text from a text-based PDF. One path extracts candidate current role, experience, education, and work history; another makes section-aware chunks, embeds them with OpenAI, and stores them in Supabase pgvector. During relevant sections, semantic retrieval supplies evidence for candidate strengths. Resume evidence is not confirmed user truth: only the user's confirmation or correction can put those facts and strengths into confirmed workflow state.

When all five sections are complete, the workflow enters a dedicated final-artifact path. It gathers confirmed structured information, including explicit unknowns. The model produces a structured final-output object; deterministic application code derives fields that should not drift, preserves confirmed strengths and Action Plan text, and renders the result as Markdown. The artifact is persisted and later retrieved through the backend's final-output read path, keeping its availability and content consistent for the frontend. The UI displays it as a read-only product artifact rather than inserting it into the chat transcript.

## Technical tradeoffs

### Separating LangGraph checkpoints from application persistence

I separated workflow checkpointing from application persistence because the two stores serve different responsibilities. LangGraph uses PostgreSQL checkpoints for operational state: conversation history, the active section, collected state, and routing data needed to resume a thread correctly. Supabase stores product-facing records such as structured section content and the rendered final artifact. Keeping these responsibilities separate prevents user-facing reads from depending on LangGraph's internal checkpoint format.

The cost is a second persistence layer and the possibility of temporary divergence. A live LangGraph checkpoint is allowed to advance when a Supabase write fails. The failure is recorded in checkpointed retry state so the divergence is explicit and recoverable rather than silently discarded. This does not claim transactional consistency across both stores.

During development, the persistence trigger became deterministic and state-derived instead of trusting the model's `should_save_content` flag. A redundant upsert costs latency and introduces another failure surface, while skipping a real dirty write risks silent data loss. JobBuddy therefore writes when section status or extracted user data actually changes and retries failed writes explicitly. A single database representation would have fewer moving parts, but it would couple product reads to the orchestration framework and make future workflow changes harder.

### Resume retrieval and confirmation

The resume has two jobs: structured extraction proposes Background facts, while semantic retrieval supplies evidence for Skill Assessment. Neither path bypasses user confirmation. The agent can suggest a strength from a retrieved work-history entry, but only the user's reply can establish it as confirmed state. This boundary costs an extra conversational step; it avoids treating a model interpretation or resume wording as an authoritative fact.

Production retrieval uses OpenAI `text-embedding-3-small` (1,536 dimensions), cosine similarity, section-aware chunks, and the top three matches. Summary-style chunks are excluded from retrieval because they can match broadly without supplying specific evidence. In a [labeled retrieval evaluation](evals/resume_retrieval/README.md) of 32 queries and 37 evidence items, this setup reached Recall@1 0.58, Recall@3 0.86, Recall@5 0.95, and MRR 0.76. The experiments also compared BM25 and fixed-size windows. Section-aware chunks and similarly sized fixed windows had comparable ranking quality; section-aware chunks were retained for attribution and section filtering, while top-three retrieval balanced evidence coverage against prompt size. These are retrieval metrics, not a claim that the generated career advice is equally accurate.

### Structured synthesis followed by deterministic rendering

For the final career plan, I separated content synthesis from presentation. The model produces a structured plan rather than the complete user-facing document. Application code validates that result and renders the final Markdown deterministically. The model owns reasoning-heavy content such as positioning, priorities, rationale, and recommendations; application code controls the document structure, confirmed Action Plan steps, derived unknowns, and final format.

This design requires a structured schema plus validation and rendering code, and it gives the model less freedom over presentation. Direct Markdown generation would be simpler initially, but it would make formatting, missing fields, grounding, and output consistency harder to control. It could also rewrite Action Plan steps the user had already confirmed.

The structured path preserves confirmed Action Plan text instead of asking the model to rewrite it. Missing information is derived or reported deterministically. The result is easier to validate, persist, and retrieve safely. Content fingerprints also prevent regeneration from overwriting downstream user edits; when source memory changes, the system can deliberately invalidate or regenerate the artifact instead of silently serving stale content.

### Production lessons and validation

Production testing exposed issues that unit-level generation alone would not reveal. SSE event/carry buffering across the Next.js proxy was corrected so streamed responses and progress reach the browser. Routing now advances a completed section to the next unfinished one even when a model returns `stay`. An edited Action Plan becomes a new proposal and cannot confirm itself in the same turn. Final Plan grounding checks distinguish total professional experience from tenure in a specific role, preserve confirmed strengths, and handle industry flexibility and unknowns consistently.

Profiling also found eager model-provider SDK imports on the startup path. Loading only the provider in use reduced measured application startup from approximately 13 seconds to 2 seconds. Production validation exercised both no-resume and resume-grounded journeys against the deployed Fly.io API and Vercel frontend, including all five sections, Action Plan revision and confirmation, Final Plan grounding, refresh persistence, mobile layout, streaming, security boundaries, and runtime errors.

The stack is Python, FastAPI, LangGraph/LangChain, OpenAI and optional LangSmith tracing for the agent; PostgreSQL checkpoints and Supabase/pgvector for data; Next.js, React, and TypeScript for the interface. The application uses SSE for streaming, pytest and retrieval/regression evaluations for testing, and Docker, Fly.io, and Vercel for deployment.

## Limitations and future improvements

JobBuddy is deliberately scoped as a career-planning assistant rather than a complete job-search platform. It does not independently verify every qualification claim against external evidence, guarantee that identified gaps are complete, rank candidates for employers, or make hiring decisions. Its recommendations depend primarily on information supplied during the guided conversation and optional resume, so incomplete, outdated, or inaccurate inputs can reduce the quality of the plan. Resume extraction requires a text layer; scanned PDFs are not OCR-processed. JobBuddy also does not incorporate live labor-market data, analyze job postings at scale, or continuously update the plan as external requirements change.

The structured five-section workflow is easier to reason about and supports a consistent final artifact, but users with unusual career paths may need to revisit earlier assumptions or compare multiple target roles. The system also depends on multiple backend components. A failed application-level write can temporarily leave live workflow state ahead of durable product records until a retry succeeds.

The next priorities are broader evidence and better plan iteration. Selected job descriptions and live labor-market data could ground recommendations beyond the user's inputs and optional resume. A revision/history workflow could compare multiple target roles and regenerate affected plan sections while protecting confirmed steps and user edits. Authenticated user identity, richer observability, stronger automated evaluation of plan quality and grounding, and scaled retrieval for larger document collections would improve reliability as the product grows.

## Getting started

### Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) for Python dependencies
- Node.js and npm for the frontend
- Credentials for at least one supported language-model provider

### Backend

```bash
uv sync
cp .env.example .env
uv run python src/run_service.py
```

The backend runs at `http://localhost:8080`. Configure `.env` with one supported model-provider credential, such as `OPENAI_API_KEY`. SQLite checkpointing works locally with `DATABASE_TYPE=sqlite`; PostgreSQL checkpointing uses `DATABASE_TYPE=postgres` and `POSTGRES_URI`. Full durable application persistence requires `SUPABASE_URL` and `SUPABASE_SECRET_KEY` (the legacy `SUPABASE_SERVICE_ROLE_KEY` is still supported). Resume indexing and retrieval also require OpenAI embeddings and the Supabase resume/pgvector migrations. LangSmith variables are optional. A production deployment must set `MODE=production` and `AUTH_SECRET`.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

For local development, set `NEXT_PUBLIC_API_ENV=local` and point `JOBBUDDY_API_URL_LOCAL` at the local backend. Set `JOBBUDDY_API_TOKEN` only when the backend uses `AUTH_SECRET`. The frontend runs at `http://localhost:3000`.

Do not commit `.env` or `.env.local`; the example files contain variable names and placeholders only.
