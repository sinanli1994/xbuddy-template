# JobBuddy

JobBuddy is an AI-powered career-planning assistant that turns a guided conversation into a grounded, personalized job-search plan. It preserves progress across sessions and produces a durable final artifact based on the user's own inputs.

- [Live app](https://jobbuddy-plum.vercel.app)
- [Live API](https://jobbuddy-api.fly.dev)
- [Video demo](https://www.loom.com/share/7dff808a896740649913255b4d05db42)

## Problem and intended user

### The problem

Job searching can be overwhelming, especially for people who want to make a career change but are unsure how to turn their background, skills, and goals into a clear plan. Career advice is often scattered across resumes, job postings, online resources, and generic recommendations. Users can struggle to understand which roles fit them, how their reported gaps relate to a target role, and what they should do next.

### The intended user

JobBuddy is designed for job seekers who need more structure and personalized guidance during career planning. This includes recent graduates, professionals changing career paths, and candidates with relevant experience who are unsure how to position themselves for a new role. The ideal user has a general direction in mind but needs help organizing their experience, relating self-reported gaps to a target role, and turning that information into an actionable strategy.

## What JobBuddy does

JobBuddy guides users through a five-section conversation covering Career Goal, Background, Job Preferences, Skill Assessment, and Action Plan. It preserves conversation history and workflow progress so a user can leave and resume without starting over. After the user completes the five sections, JobBuddy synthesizes the confirmed information into a personalized career plan with priorities, next steps, and explicit unknowns.

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
LangGraph workflow --------> PostgreSQL checkpoints
   |
   +-----------------------> Supabase application records
```

The Next.js frontend provides the five-section experience. The browser calls Next.js server routes, which attach server-only configuration and forward requests to FastAPI. The browser does not call FastAPI or Supabase directly, and the backend bearer token is never included in browser assets.

FastAPI is the main application API. It validates requests, applies authentication and request protection, loads the relevant conversation, and invokes the LangGraph workflow. The backend exposes synchronous invocation, Server-Sent Events (SSE) streaming, conversation history, public completion state, and final-output retrieval. The SSE path streams user-visible tokens, messages, and public progress events while suppressing internal decision, extraction, and synthesis output.

LangGraph manages the multi-step workflow. Its state records the active section, collected user data, completion and confirmation state, pending Action Plan proposals, and the next operation. Explicit graph state and deterministic routes control section transitions; the language model handles conversational generation, structured decisions, extraction, and grounded synthesis where those capabilities are useful.

PostgreSQL checkpointing preserves LangGraph's operational execution state, including the conversation history and workflow data required to resume an interrupted thread. The checkpoint is the primary source for a live thread.

Supabase stores durable, product-facing section records and the rendered final artifact. These application records are separate from LangGraph checkpoints and are read through the backend API. This gives the product a stable data model without exposing LangGraph's internal checkpoint representation to application reads.

When all five sections are complete, the workflow enters a dedicated final-artifact path. It gathers the structured information collected during the conversation, including explicit unknowns. The model produces a structured final-output object; deterministic application code derives fields that should not drift, preserves confirmed Action Plan text, and renders the result as Markdown. The artifact is persisted and later retrieved through the backend's final-output read path, keeping its availability and content consistent for the frontend. The UI displays it as a read-only product artifact rather than inserting it into the chat transcript.

## Technical tradeoffs

### Separating LangGraph checkpoints from application persistence

I separated workflow checkpointing from application persistence because the two stores serve different responsibilities. LangGraph uses PostgreSQL checkpoints for operational state: conversation history, the active section, collected state, and routing data needed to resume a thread correctly. Supabase stores product-facing records such as structured section content and the rendered final artifact. Keeping these responsibilities separate prevents user-facing reads from depending on LangGraph's internal checkpoint format.

The cost is a second persistence layer and the possibility of temporary divergence. A live LangGraph checkpoint is allowed to advance when a Supabase write fails. The failure is recorded in checkpointed retry state so the divergence is explicit and recoverable rather than silently discarded. This does not claim transactional consistency across both stores.

During development, the persistence trigger became deterministic and state-derived instead of trusting the model's `should_save_content` flag. A redundant upsert costs latency and introduces another failure surface, while skipping a real dirty write risks silent data loss. JobBuddy therefore writes when section status or extracted user data actually changes and retries failed writes explicitly. A single database representation would have fewer moving parts, but it would couple product reads to the orchestration framework and make future workflow changes harder.

### Structured synthesis followed by deterministic rendering

For the final career plan, I separated content synthesis from presentation. The model produces a structured plan rather than the complete user-facing document. Application code validates that result and renders the final Markdown deterministically. The model owns reasoning-heavy content such as positioning, priorities, rationale, and recommendations; application code controls the document structure, confirmed Action Plan steps, derived unknowns, and final format.

This design requires a structured schema plus validation and rendering code, and it gives the model less freedom over presentation. Direct Markdown generation would be simpler initially, but it would make formatting, missing fields, grounding, and output consistency harder to control. It could also rewrite Action Plan steps the user had already confirmed.

The structured path preserves confirmed Action Plan text instead of asking the model to rewrite it. Missing information is derived or reported deterministically. The result is easier to validate, persist, and retrieve safely. Content fingerprints also prevent regeneration from overwriting downstream user edits; when source memory changes, the system can deliberately invalidate or regenerate the artifact instead of silently serving stale content.

## Limitations and future improvements

JobBuddy is deliberately scoped as a career-planning assistant rather than a complete job-search platform. It does not independently verify every qualification claim against external evidence, guarantee that identified gaps are complete, rank candidates for employers, or make hiring decisions. Its recommendations depend primarily on information supplied during the guided conversation, so incomplete, outdated, or inaccurate inputs can reduce the quality of the plan. It also does not incorporate live labor-market data, analyze job postings at scale, or continuously update the plan as external requirements change.

The structured five-section workflow is easier to reason about and supports a consistent final artifact, but users with unusual career paths may need to revisit earlier assumptions or compare multiple target roles. The system also depends on multiple backend components. A failed application-level write can temporarily leave live workflow state ahead of durable product records until a retry succeeds.

The next priorities are stronger grounding and better plan revision. JobBuddy could accept resumes and selected job descriptions as supporting evidence while clearly separating user-confirmed facts from model-derived suggestions. A revision workflow could let users update individual assumptions, compare roles, and regenerate only affected plan sections while preserving confirmed Action Plan steps and protecting user edits. Richer observability, automated recovery for failed durable writes, and evaluation datasets focused on plan quality and grounding would make the system easier to operate and improve.

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

The backend runs at `http://localhost:8080`. Configure `.env` with one supported model-provider credential, such as `OPENAI_API_KEY`. SQLite checkpointing works locally with `DATABASE_TYPE=sqlite`; PostgreSQL checkpointing uses `DATABASE_TYPE=postgres` and `POSTGRES_URI`. Full durable application persistence also requires `SUPABASE_URL` and `SUPABASE_SECRET_KEY` (the legacy `SUPABASE_SERVICE_ROLE_KEY` is still supported). LangSmith variables are optional. A production deployment must set `MODE=production` and `AUTH_SECRET`.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

For local development, set `NEXT_PUBLIC_API_ENV=local` and point `JOBBUDDY_API_URL_LOCAL` at the local backend. Set `JOBBUDDY_API_TOKEN` only when the backend uses `AUTH_SECRET`. The frontend runs at `http://localhost:3000`.

Do not commit `.env` or `.env.local`; the example files contain variable names and placeholders only.
