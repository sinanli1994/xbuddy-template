# PR 6 — Deploying the JobBuddy API to Fly.io

## Architecture

```
Browser
  └─> Vercel / Next.js frontend
        └─> server-side proxy          (the bearer token lives here, never in the browser)
              └─> Fly.io FastAPI
                    ├─> OpenAI
                    └─> Supabase
```

Supabase is reached over **two independent paths**, with **two different credentials**.
Conflating them is the easiest mistake to make here, so they are spelled out:

| | Data API | Direct PostgreSQL |
|---|---|---|
| Used for | domain tables — `section_states`, `final-outputs` | LangGraph `AsyncPostgresSaver` / `AsyncPostgresStore` |
| Reached via | PostgREST over HTTPS | the Postgres wire protocol on TCP 5432 |
| Credential | `SUPABASE_SECRET_KEY` | the password embedded in `POSTGRES_URI` |
| Configured by | `SUPABASE_URL` + `SUPABASE_SECRET_KEY` | `DATABASE_TYPE=postgres` + `POSTGRES_URI` |
| Code path | `integrations/supabase/supabase_client.py` | `memory/postgres.py` |

`SUPABASE_SECRET_KEY` is an API key that PostgREST validates; it has **no meaning** to
the Postgres server. The `POSTGRES_URI` password is a database role password; it has
**no meaning** to PostgREST. Setting one and expecting the other to work fails in a way
that looks like a permissions bug and is not.

Use the **Session Pooler on port 5432** for `POSTGRES_URI`. Stage 5A established why:
LangGraph's saver runs in pipeline mode, psycopg's `prepare_threshold` defaults to `5`,
and the pool sets `autocommit` plus four `keepalives_*` options — all session-level
features that a transaction-mode pooler (6543, PgBouncer) does not support.

## Environment matrix

### Non-secret — `[env]` in `fly.toml`, committed

| Variable | Value | Why it is here |
|---|---|---|
| `MODE` | `production` | Turns on the production posture. With it set, the service **refuses to start** unless `AUTH_SECRET` is present, and CORS defaults to allowing no browser origin. |
| `DATABASE_TYPE` | `postgres` | Selects the Postgres checkpointer. There is no fallback to SQLite — a misconfigured deployment fails loudly rather than silently becoming ephemeral. |
| `PORT` | `8080` | Matches `internal_port`. `run_service.py` reads `PORT` and falls back to `settings.PORT`. |
| `CORS_ALLOW_ORIGINS` | `""` | No browser origin is allowed; the frontend proxies server-side. See the note below — this line documents intent, it is not the mechanism. |
| `RATE_LIMIT_EXPENSIVE` | optional, default `10/minute` | Per-IP limit on `/invoke` and `/stream`. |
| `LANGSMITH_PROJECT` | optional | Trace project name. Not credential-bearing. |

> The `LANGSMITH_*` variables are **not** declared as `Settings` fields — `Settings`
> uses `extra="ignore"`, so they are dropped there and read straight from the
> environment by the LangSmith SDK. Setting them as Fly env/secrets still works; they
> just do not appear in `Settings`, so do not go looking for them there.

> **`CORS_ALLOW_ORIGINS = ""` is inert by mechanism.** `Settings` sets
> `env_ignore_empty=True`, so an empty environment value is read as *unset* and never
> reaches the field. The empty-in-production result comes from the mode-aware default in
> `Settings.cors_allow_origins`, not from this line. The line is kept because it states
> the intent at the point someone will look, and because setting a *real* origin here
> does take effect. This was a genuine trap: before PR 6 Stage 5B the field defaulted to
> `http://localhost:3000`, so a production deployment would have advertised the local dev
> server as an allowed origin with an apparently-contradicting config line sitting right
> next to it.

### Secret — `fly secrets set`, never committed

| Variable | Notes |
|---|---|
| `AUTH_SECRET` | Shared bearer token. Every route except `/health` requires it. The frontend proxy holds it server-side; it must never reach the browser. |
| `OPENAI_API_KEY` | Billing-bearing. |
| `SUPABASE_SECRET_KEY` | Bypasses RLS. Server-side only, never shipped to a browser. |
| `POSTGRES_URI` | Embeds the database role password. A **different credential** from `SUPABASE_SECRET_KEY`. |
| `LANGSMITH_API_KEY` | Optional; only if tracing is on. |
| `SUPABASE_URL` | **Not credential-bearing** — it is the public project URL, of the form `https://<ref>.supabase.co`, and the frontend already exposes it. It is set as a secret here only because Fly gives no benefit to splitting it out, and because the project reference is mildly identifying. Treating it as non-secret would not be a vulnerability. |

### The secrets command

Placeholders only — never paste real values into a shell that records history, and
never into a file that git tracks:

```bash
fly secrets set \
  AUTH_SECRET='<long-random-string>' \
  OPENAI_API_KEY='sk-...' \
  SUPABASE_URL='https://<project-ref>.supabase.co' \
  SUPABASE_SECRET_KEY='sb_secret_...' \
  POSTGRES_URI='postgresql://postgres.<project-ref>:<url-encoded-password>@aws-0-<region>.pooler.supabase.com:5432/postgres?sslmode=require' \
  --app jobbuddy-api

# optional
fly secrets set LANGSMITH_API_KEY='lsv2_...' --app jobbuddy-api
```

Generate `AUTH_SECRET` with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.

If the database password contains `@`, `/`, `:` or `#`, it must be **percent-encoded**
inside the URI or it will corrupt the authority section. `p@ss/word` becomes
`p%40ss%2Fword`.

Setting secrets restarts the machine, which is expected.

## First-deploy verification sequence

Run in order. Stop at the first failure — later steps assume the earlier ones passed.

```bash
# 1. Secrets exist. Names and digests only; values are never displayed.
fly secrets list --app jobbuddy-api

# 2. Deploy.
fly deploy --app jobbuddy-api

# 3. The machine booted rather than crash-looping.
fly status --app jobbuddy-api
fly logs --app jobbuddy-api            # expect no secret values, no SQLite fallback

# 4. /health is reachable and unauthenticated.
curl -sS https://jobbuddy-api.fly.dev/health

# 5. Protected routes reject an unauthenticated caller.
curl -sS -o /dev/null -w '%{http_code}\n' \
  -X POST https://jobbuddy-api.fly.dev/history \
  -H 'Content-Type: application/json' -d '{"thread_id":"x","user_id":1}'
#   expect 401

# 6. Postgres is genuinely reachable from Fly — the first authoritative test of
#    credentials, TLS and database role, none of which local validation could reach.
fly ssh console -C "/app/.venv/bin/python /app/scripts/fly_connectivity_probe.py"

# 7. Durability across a real restart.
fly ssh console -C "/app/.venv/bin/python /app/scripts/fly_durability_probe.py write"
#    -> note the thread id it prints
fly machine restart <machine-id>
fly ssh console -C "/app/.venv/bin/python /app/scripts/fly_durability_probe.py read --thread <id>"
fly ssh console -C "/app/.venv/bin/python /app/scripts/fly_durability_probe.py cleanup --thread <id>"

# 8. One authenticated end-to-end turn. THIS COSTS A REAL OPENAI CALL.
curl -sS -X POST https://jobbuddy-api.fly.dev/invoke \
  -H "Authorization: Bearer $AUTH_SECRET" -H 'Content-Type: application/json' \
  -d '{"message":"I want to move into SRE","thread_id":"smoke-1","user_id":1}'
```

Steps 1–7 cost nothing. **Step 8 is the only one that spends money**, and it is
deliberately last so that everything cheap has already been proven.

### What the durability proof does and does not show

The restart between the `write` and `read` phases is the whole proof. Reconnecting a
pool inside one process shows only that reconnection works — the state could have been
in memory the entire time. A restarted machine is a new process with empty memory, so
state that reappears came from the database.

Both phases print the machine's uptime. **If the uptime reported by `read` is not lower
than the one reported by `write`, the machine never restarted and the run proves
nothing.** The probes also fake every model call, so the proof costs nothing and does
not depend on an OpenAI key being valid.

## Local validation limitation (resolved on Fly)

> Local Windows validation could not reach the Supabase Session Pooler on TCP 5432 from
> two client networks, while TCP 443 to the same host was reachable. Therefore
> authentication/SSL/database-role behaviour was not exercised locally. Fly runtime
> verification is the next authoritative connectivity proof.

To be precise about what this does and does not imply: the TCP handshake never
completed, so **nothing downstream of it was ever tested**. There is no evidence the
credentials, TLS configuration or database role are wrong, and no evidence they are
right. They are simply untested locally.

**Fly answered it.** The connectivity probe, run inside the deployed machine, reached
Supabase over the Session Pooler on 5432 and passed every check — so credentials, TLS
and the database role are now proven from the network position that actually matters.
The local block is a property of the developer's network, not of the configuration.

## Operational notes

**One machine, and it stays up.** `auto_stop_machines = "off"` with
`min_machines_running = 1`. Scale-to-zero would drop in-flight SSE streams when the
machine suspends and pay the full LangChain/LangGraph import cost on the next cold
start.

**Multiple instances are not proven safe.** slowapi keeps its rate-limit counters in
process memory, so N machines would mean N independent limits and an effective limit of
N × `RATE_LIMIT_EXPENSIVE`. Raising `min_machines_running` needs a shared counter store
first.

**No volume.** All persistence is in Supabase Postgres. A Fly volume would only create
machine-local state that a redeploy silently loses.

**Memory is a judgment call.** `1gb` was chosen for a service that imports LangChain,
LangGraph and several provider SDKs; it was not measured, because a reliable RSS figure
could not be obtained on the Windows development machine. If the machine OOMs on first
boot it will show in `fly logs` as an out-of-memory kill, and the fix is a one-line
change to `memory` in `fly.toml`.

---

# Live verification results

Everything below was executed against the running deployment. Where a claim is proven
offline rather than live, it says so.

## Deployment

| | |
|---|---|
| App | `jobbuddy-api` |
| Host | https://jobbuddy-api.fly.dev |
| Region | `yyz` (Toronto) — close to the Supabase project in AWS `ca-central-1` |
| Machines | one, `auto_stop_machines = "off"`, `min_machines_running = 1` |
| Volume | none — all persistence is in Supabase Postgres |

## Postgres connectivity — proven live from Fly

Run inside the machine:
`fly ssh console -C "/app/.venv/bin/python /app/scripts/fly_connectivity_probe.py"`

| Check | Result |
|---|---|
| Direct psycopg connection + `SELECT 1` | PASS |
| Server version | PostgreSQL 17.6 |
| `pg_manager.setup()` | PASS |
| `AsyncPostgresSaver` | PASS |
| `AsyncPostgresStore` | PASS |
| Saver tables | 4/4 |
| Store tables | 2/2 |
| Vector tables | **n/a — correct.** No index is configured, so `AsyncPostgresStore.setup()` never runs its VECTOR_MIGRATIONS |
| Pool close | clean |

## Restart durability — proven live

Two phases with a real `fly machine restart` between them, model calls faked (zero cost):

- checkpoint written on Fly
- machine restarted — **pid and uptime both changed**, so it was genuinely a new process
- the same thread restored its state, and the unique sentinel matched, proving it is the
  same conversation rather than a fresh one
- identifiers survived
- a different thread saw nothing
- cleanup left 0 probe rows

A single process closing and reopening a pool would have proven only reconnection. The
restart is what distinguishes process memory from database durability.

## Live API smoke

| Endpoint | Result | Evidence |
|---|---|---|
| `GET /health` | PASS | 200, unauthenticated by design — Fly's health check sends no token |
| `GET /info` | PASS | 200 with token, **401 without**; `endpoints: []`; cross-checked against the live `/openapi.json`, every advertised path is registered |
| Deleted legacy routes | PASS | `/sync_section`, `/refine_section` → 404, absent from both `/info` and the live route table |
| `POST /history` (owner) | PASS | 200; ids echo; `['human','ai']` chronological; exactly three top-level keys; zero graph internals |
| `POST /history` (wrong user) | PASS | 404 `{"detail":"Thread not found"}` — no content, no existence hint |
| `POST /invoke` | PASS | 200; five canonical sections; `collection_complete=false`, `artifact_available=false`; active-section metadata present |
| `POST /stream` | PASS | full SSE contract, below |

Paid model calls in the final smoke run: **2** (one `/invoke`, one `/stream`). Each new
thread shows exactly one human + one AI message, so neither was retried or duplicated.

## SSE event contract — observed live

```
metadata → token × 38 → message → section → completion → [DONE]
```

- exactly **one** `completion` event, and it precedes `[DONE]`
- `[DONE]` terminates the stream
- zero `error` events
- the `section` event was emitted, not silently dropped
- no `internal_extraction`, `internal_decision`, `internal_synthesis`, `skip_stream` or
  `do_not_stream` tags reached the wire
- no structured extraction/decision/synthesis JSON leaked
- the `completion` payload matched `/invoke`'s projection exactly — same booleans, same
  five sections, same three-field shape

## Enum/string boundary

A live `/invoke` initially returned 500 with `'str' object has no attribute 'value'`.
Checkpoint restores do not guarantee the typed form, and the domain layer already knew
this — `router_node` coerces with `SectionID(...)`, and `coerce_section_state` exists
because "deserialized checkpoints can hand back plain dicts". Only the service boundary
assumed it.

Fixed with one shared pair of helpers, `enum_value` and `section_status`, used by
`/invoke`, `/stream` and `public_completion` so the three cannot diverge. `enum_value`
**raises** on anything that is not an `Enum` or `str`: a `str(value)` fallback would turn
a real bug into a plausible-looking string and ship it to the frontend.

`/stream` carried the identical bug, where it was worse — wrapped in
`except Exception: logger.error(...)`, it dropped the `section` event silently. Both are
fixed and both verified live.

## LangSmith

Traces reach project **`jobbuddy-production`**. Both smoke threads were correlated by
`thread_id` — 31 runs each, **zero errored**.

## Auth, rate limiting, CORS

- Shared bearer token on every route except `/health`, compared with
  `secrets.compare_digest`.
- `MODE=production` **refuses to start** without `AUTH_SECRET` — verified by running the
  container's exact command locally: exit 1, no server started.
- Rate limiting on `/invoke` and `/stream` only, keyed on `Fly-Client-IP`.
  `X-Forwarded-For` is never consulted, because any client can send it.
- `/history` is deliberately not rate-limited — a cheap read with no model call.
- CORS production default is deny-all; `allow_credentials=False`, since no cookie or
  browser-credential flow exists.

## Known limitations

**The bearer token is demo-level auth, not identity.** One shared secret for all callers.
It proves a request came from something holding the token; it does not establish *who*.
`/history` scopes reads by `user_id`, but a caller holding the token may pass any
`user_id`. Real per-user identity needs Supabase Auth or JWTs — deliberately out of scope.

**Rate limiting is process-local.** slowapi keeps counters in memory, so the limit is
per-machine. Correct at one machine; scaling to N would give an effective limit of
N × `RATE_LIMIT_EXPENSIVE`.

**The live smoke covered a first turn only.** Multi-turn progression, sections reaching
`done`, `collection_complete` flipping true, and final-output synthesis were *not*
replayed live — they are covered offline by the 603 agent tests. This is a narrower claim
than "the agent works end to end".

**`finished` now means completion (Issue #10, fixed in this PR).** It used to be
written by the router, and only from inside the `next` branch — so a thread whose fifth
section completed on a `stay` turn sat at `finished=False` indefinitely. Worse, the
completing turn routes `memory_updater -> implementation -> END` and never reaches the
router at all, so the node that owned the flag was bypassed on exactly the turn that
should have set it.

Both `finished` and `should_generate_final_output` now derive from a single
`all_sections_complete` call in `memory_updater`, the node that owns DONE transitions.
Stale checkpoints self-heal on the next turn, so no migration was needed.

**The public API is unchanged.** `finished` is still not exposed — clients read
`collection_complete` and `artifact_available`, which describe what a client actually
needs without coupling it to internal graph state.

**The LangGraph store is assigned but never read.** `agent.store` is set in the lifespan
and no JobBuddy code consumes it. Its tables exist because `setup()` creates them, not
because the product depends on them.

**The realtime worker is inert.** `/realtime/subscribe` imports
`integrations.supabase.realtime_worker`, which does not exist in the repo.
`USE_SUPABASE_REALTIME` defaults to `False`, so the path is never taken — but the route
is registered and would fail if called. Deferred, not fixed.

**`SUPABASE_DB_URL` is dead.** Declared on `Settings` and read by nothing. Use
`POSTGRES_URI`, or the five discrete `POSTGRES_*` fields.

**Frontend proxy and RLS work is not part of this PR.** The browser is expected to reach
this API through the Next.js server-side proxy; wiring that, and replacing anon-key
access with backend-mediated reads, remains outstanding.

**Smoke threads remain in Postgres**: `smoke-7e01a4e18f21`/999100,
`smoke-probe-2`/999101, `smoke-58336ea7765f`/999200, `smoke-33bd8a5520b9`/999201. There
is no cleanup tooling designed for ordinary smoke threads; leaving them is harmless.
