import inspect
import json

from core.logging_config import get_logger, setup_logging

# Setup logging configuration
setup_logging()
import time
import warnings
from collections.abc import AsyncGenerator
import secrets
from contextlib import asynccontextmanager
from enum import Enum
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.callback import CallbackHandler  # type: ignore[import-untyped]
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient
from starlette.middleware.base import BaseHTTPMiddleware

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info
from agents.xbuddy import initialize_xbuddy_state
from agents.xbuddy.enums import SectionID
from agents.xbuddy.prompts import SECTION_TEMPLATES
from core import settings
from core.settings import DatabaseType
# Removed: # DentApp (removed) integration
from memory import initialize_database, initialize_store, pg_manager
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    CompletionState,
    InvokeResponse,
    PublicSection,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming frontend requests with detailed information."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID for tracking
        request_id = str(uuid4())[:8]
        start_time = time.time()
        
        # Log request details
        logger.info(f"=== FRONTEND_REQUEST_START: {request_id} ===")
        logger.info(f"FRONTEND_REQUEST: {request.method} {request.url}")
        logger.info(f"FRONTEND_REQUEST: Client IP: {request.client.host if request.client else 'unknown'}")
        logger.info(f"FRONTEND_REQUEST: User-Agent: {request.headers.get('user-agent', 'unknown')}")
        logger.info(f"FRONTEND_REQUEST: Content-Type: {request.headers.get('content-type', 'unknown')}")
        
        # Log all headers (excluding sensitive ones)
        sensitive_headers = {'authorization', 'cookie', 'x-api-key'}
        headers_to_log = {
            k: v for k, v in request.headers.items() 
            if k.lower() not in sensitive_headers
        }
        logger.info(f"FRONTEND_REQUEST: Headers: {headers_to_log}")
        
        # Log query parameters
        if request.query_params:
            logger.info(f"FRONTEND_REQUEST: Query params: {dict(request.query_params)}")
        
        # Log request body for POST/PUT requests, but skip for streaming endpoints
        is_streaming_endpoint = "/stream" in str(request.url.path)
        if request.method in ["POST", "PUT", "PATCH"] and not is_streaming_endpoint:
            try:
                # Use the safer approach for non-streaming endpoints
                body = await request.body()
                if body:
                    # Try to parse as JSON for better logging
                    try:
                        body_json = json.loads(body.decode('utf-8'))
                        # Mask sensitive fields
                        masked_body = mask_sensitive_fields(body_json)
                        logger.info(f"FRONTEND_REQUEST: Body (JSON): {json.dumps(masked_body, ensure_ascii=False)}")
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Log as raw text if not JSON
                        body_str = body.decode('utf-8', errors='replace')[:1000]  # Limit size
                        logger.info(f"FRONTEND_REQUEST: Body (raw): {body_str}")
                else:
                    logger.info("FRONTEND_REQUEST: Body: (empty)")
                    
            except Exception as e:
                logger.warning(f"FRONTEND_REQUEST: Could not read body: {e}")
        elif is_streaming_endpoint:
            # For streaming endpoints, just log that we're skipping body logging
            logger.info("FRONTEND_REQUEST: Body: (skipped for streaming endpoint)")
        
        # Process the request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response details
            logger.info(f"=== FRONTEND_RESPONSE_END: {request_id} ===")
            logger.info(f"FRONTEND_RESPONSE: Status: {response.status_code}")
            logger.info(f"FRONTEND_RESPONSE: Processing time: {process_time:.3f}s")
            logger.info(f"FRONTEND_RESPONSE: Content-Type: {response.headers.get('content-type', 'unknown')}")
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"=== FRONTEND_REQUEST_ERROR: {request_id} ===")
            logger.error(f"FRONTEND_ERROR: {str(e)}")
            logger.error(f"FRONTEND_ERROR: Processing time: {process_time:.3f}s")
            raise


# Display-only position badge for the progress sidebar, which renders it as
# "#1".."#5". NOT a database id: section_states.section_id is TEXT and stores the
# SectionID string value, and save_section_state() takes that string. Never
# persist this and never import it outside this module. The payload key is called
# "database_id" only because the existing frontend contract already calls it that.
_SECTION_DISPLAY_POSITION: dict[str, int] = {
    "career_goal": 1,
    "background": 2,
    "job_preferences": 3,
    "skill_assessment": 4,
    "action_plan": 5,
}


def mask_sensitive_fields(data: dict | list | Any) -> dict | list | Any:
    """Mask sensitive fields in request data for logging."""
    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in ['password', 'token', 'secret', 'key', 'auth']):
                masked[key] = "***MASKED***"
            elif isinstance(value, (dict, list)):
                masked[key] = mask_sensitive_fields(value)
            else:
                masked[key] = value
        return masked
    elif isinstance(data, list):
        return [mask_sensitive_fields(item) for item in data]
    else:
        return data


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        # Local development convenience. A deployed process cannot reach this
        # branch: Settings refuses to construct when MODE is production and
        # AUTH_SECRET is unset, so the service fails to start rather than serving
        # an unprotected API.
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    # Constant-time comparison: `!=` on secrets leaks length and prefix through
    # timing. The value is never logged — RequestLoggingMiddleware masks the
    # Authorization header, and no error message echoes the credential.
    if not secrets.compare_digest(http_auth.credentials, auth_secret):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer and store
    based on settings.
    """
    # Initialize Realtime worker if enabled
    realtime_worker = None
    if settings.USE_SUPABASE_REALTIME:
        try:
            from integrations.supabase.realtime_worker import RealtimeWorker
            realtime_worker = RealtimeWorker()
            await realtime_worker.start()
            app.state.realtime_worker = realtime_worker
            logger.info("✅ Realtime worker started")
        except Exception as e:
            logger.error(f"Failed to start Realtime worker: {e}", exc_info=True)
            # Continue without Realtime worker
    
    try:
        if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
            # Initialize PostgreSQL connection pool
            await pg_manager.setup()
            saver = pg_manager.get_saver()
            store = pg_manager.get_store()
            
            # Configure all agents
            agents = get_all_agent_info()
            for a in agents:
                agent = get_agent(a.key)
                agent.checkpointer = saver
                agent.store = store
            
            logger.info("Application startup complete with PostgreSQL connection pool")
            yield
            
            # Clean up connection pool
            await pg_manager.cleanup()
        else:
            # SQLite or MongoDB - use original logic
            async with initialize_database() as saver, initialize_store() as store:
                # Set up both components
                if hasattr(saver, "setup"):  # ignore: union-attr
                    await saver.setup()
                # Only setup store for Postgres as InMemoryStore doesn't need setup
                if hasattr(store, "setup"):  # ignore: union-attr
                    await store.setup()

                # Configure agents with both memory components
                agents = get_all_agent_info()
                for a in agents:
                    agent = get_agent(a.key)
                    # Set checkpointer for thread-scoped memory (conversation history)
                    agent.checkpointer = saver
                    # Set store for long-term memory (cross-conversation knowledge)
                    agent.store = store
                yield
    except Exception as e:
        logger.error(f"Error during database/store initialization: {e}")
        raise
    finally:
        # Stop Realtime worker on shutdown
        if realtime_worker:
            await realtime_worker.stop()
            logger.info("✅ Realtime worker stopped")


def client_ip_key(request: Request) -> str:
    """The rate-limit bucket for one request. Pure, so it is directly testable.

    Order is deliberate:

    1. **`Fly-Client-IP`** — set by Fly's edge and not forwardable by a caller, so
       it is the one proxy header this service trusts.
    2. **`request.client.host`** — the direct peer, used locally and in tests.

    `X-Forwarded-For` is **never** consulted. Any client can send it, so trusting
    it would let a caller mint a fresh bucket per request and bypass the limit
    entirely — worse than having no limit, because it would look like one.

    Caveat this cannot fix: when the browser talks to a Vercel server-side proxy
    which then calls this API, Fly sees the *proxy* as the client, so all users
    behind it share one bucket. That makes this cost and abuse protection for a
    demo, not a per-user quota. Real quotas need an authenticated identity and
    shared state, both deliberately out of scope for PR 6.
    """
    fly_client_ip = request.headers.get("Fly-Client-IP")
    if fly_client_ip and fly_client_ip.strip():
        return fly_client_ip.strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


# Applied only to the two expensive LLM entrypoints. /history is a cheap
# checkpoint read and is deliberately unthrottled in PR 6.
limiter = Limiter(key_func=client_ip_key)

app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS is an explicit, environment-driven allowlist. `["*"]` is gone: combined with
# allow_credentials it is also invalid per the CORS spec, and it let any page on the
# internet call this API from a browser.
#
# allow_credentials is False. Nothing here uses cookies or browser credentials —
# authentication is a bearer token the frontend attaches server-side, and the
# frontend reaches this API through its own Next.js route handlers rather than from
# the browser. Leaving it True would also forbid ever using a wildcard origin, and
# would imply a credential flow that does not exist.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

router = APIRouter(dependencies=[Depends(verify_bearer)])


@router.get("/info")
async def info() -> ServiceMetadata:
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=[],  # Models are server-managed, not user-selectable
        default_agent=DEFAULT_AGENT,
        default_model=None,  # Model selection is server-internal
        endpoints=[
            # Only routes that are actually registered. The handlers for
            # /sync_section and /refine_section were deleted in PR 6, but their
            # EndpointInfo entries stayed behind, so /info advertised an API that
            # answers 404 to anyone who believed it.
        ]
    )


def enum_value(value: Any) -> str:
    """Normalize an enum-like checkpoint value to its public string form.

        Enum -> .value
        str  -> unchanged

    A checkpoint restore does not guarantee which one you get. The domain layer already
    assumes both — `router_node` does `SectionID(state.get("current_section"))`, and
    `coerce_section_state` exists precisely because "deserialized checkpoints can hand
    back plain dicts". The service boundary assumed the typed form, and a live thread
    returned 500 with `'str' object has no attribute 'value'`.

    Anything else raises. A `str(value)` fallback would turn a real bug into a
    plausible-looking string and ship it to the frontend, which is how this class of
    error stays hidden.
    """
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, str):
        return value
    raise TypeError(
        f"expected an Enum or str at the public boundary, got {type(value).__name__}"
    )


def section_status(entry: Any) -> str:
    """The public status string for one checkpoint-restored section entry.

    The entry may be a `SectionState` or the plain dict it deserializes to, and its
    `status` may be a `SectionStatus` or a bare string. A missing section reads as
    `pending`, matching `public_completion`'s contract that the array is always five
    long.
    """
    if entry is None:
        return "pending"
    status = getattr(entry, "status", None)
    if status is None and isinstance(entry, dict):
        status = entry.get("status")
    if status is None:
        return "pending"
    return enum_value(status)


def public_completion(state_values: dict[str, Any]) -> CompletionState:
    """Project graph state onto the narrow public completion contract.

    The single source for both `/invoke` and the SSE `completion` event, so the two
    surfaces cannot drift.

    Three fields, each derived from exactly one internal signal:

    * `collection_complete` <- `should_generate_final_output`, which memory_updater
      computes from section statuses alone. Since Issue #10, `finished` is derived
      from that same rule and would give the same answer — but it stays internal:
      it is a routing concept, and clients should not be coupled to graph state.
    * `artifact_available` <- `final_output is not None`. The same signal
      implementation_node uses as its once-only guard.
    * `sections` <- all five in canonical `SectionID` order, projected to
      `{id, name, status}` only.

    Nothing else crosses the boundary: no `finished`, no `user_data`, no draft
    content, no satisfaction flags, no `database_id`. A section absent from state is
    reported as `pending` rather than omitted, so the array is always five long and
    a client can render progress without knowing the schema.
    """
    raw_sections = state_values.get("section_states") or {}

    sections: list[PublicSection] = []
    for section_id in SectionID:
        entry = raw_sections.get(section_id.value)
        status_value = section_status(entry)

        template = SECTION_TEMPLATES.get(section_id.value)
        sections.append(
            PublicSection(
                id=section_id.value,
                name=template.name if template else section_id.value,
                status=str(status_value),
            )
        )

    return CompletionState(
        collection_complete=bool(state_values.get("should_generate_final_output", False)),
        artifact_available=state_values.get("final_output") is not None,
        sections=sections,
    )


async def _handle_input(user_input: UserInput, agent: AgentGraph, agent_id: str) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id = uuid4()
    thread_id = user_input.thread_id
    user_id = user_input.user_id

    callbacks = []
    if settings.LANGFUSE_TRACING:
        langfuse_handler = CallbackHandler()
        callbacks.append(langfuse_handler)

    initial_state = None
    if not thread_id:
        # This is a new conversation, so we need to initialize a new state
        if agent_id == "xbuddy":
            initial_state = await initialize_xbuddy_state(user_id=user_id)
        else:
            raise ValueError(f"Unknown agent: {agent_id}")

        # Get the generated thread_id from initial_state
        # For dict-like states, use .get(), for Pydantic models use direct access
        logger.info(f"DEBUG: initial_state type = {type(initial_state)}")
        logger.info(f"DEBUG: hasattr(initial_state, 'user_id') = {hasattr(initial_state, 'user_id')}")

        if hasattr(initial_state, 'user_id'):
            user_id = initial_state.user_id
            thread_id = initial_state.thread_id
            logger.info(f"DEBUG: Got user_id={user_id}, thread_id={thread_id} from Pydantic model")
        else:
            user_id = initial_state.get("user_id")
            thread_id = initial_state.get("thread_id")
            logger.info(f"DEBUG: Got user_id={user_id}, thread_id={thread_id} from dict")

        logger.info(f"Initialized new thread with ID: {thread_id}")
    else:
        # This is an existing conversation, so we load the state
        logger.info(f"Loading existing thread with ID: {thread_id}")
    
    # Build the configurable dict
    configurable = {
        "thread_id": thread_id,
        "user_id": user_id,
    }
    
    # Pass thread_id to agent
    configurable["thread_id"] = thread_id
    
    # Add user's custom agent config if provided
    if user_input.agent_config:
        if overlap := configurable.keys() & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)
    
    # Create a configuration for the thread
    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
        callbacks=callbacks,
    )

    # Check for interrupts that need to be resumed
    # Extract current section for HumanMessage metadata
    current_section = None
    if not user_input.thread_id:
        # New thread - freshly initialized state, no interrupts to resume
        # Avoid calling aget_state which might trigger unexpected graph execution
        interrupted_tasks = []
        # Get current_section from initial_state that was just created
        if initial_state:
            if hasattr(initial_state, 'current_section'):
                current_section = initial_state.current_section
            else:
                current_section = initial_state.get("current_section")
    else:
        # Existing thread - check if there are interrupts to resume
        state = await agent.aget_state(config=config)
        interrupted_tasks = [
            task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
        ]
        # Extract current section from state for HumanMessage metadata
        current_section = state.values.get("current_section") if state.values else None

    input: Command | dict[str, Any]
    if interrupted_tasks:
        input = Command(resume=user_input.message)
    else:
        # Create HumanMessage with section metadata in additional_kwargs
        additional_kwargs = {}
        if current_section:
            # Get the section string ID (e.g., "interview", "icp", "pain")
            section_id_str = current_section.value if hasattr(current_section, 'value') else str(current_section)
            additional_kwargs.update({
                "section_id": section_id_str,
                "agent_name": agent_id,
            })

        input = {"messages": [HumanMessage(content=user_input.message, additional_kwargs=additional_kwargs)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id


@router.post("/{agent_id}/invoke")
@router.post("/invoke")
@limiter.limit(settings.RATE_LIMIT_EXPENSIVE)
async def invoke(
    request: Request, user_input: UserInput, agent_id: str = DEFAULT_AGENT
) -> InvokeResponse:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.
    """
    # Log detailed invoke request
    logger.info(f"=== INVOKE_REQUEST: agent_id={agent_id} ===")
    logger.info(f"INVOKE_REQUEST: user_id={user_input.user_id}")
    logger.info(f"INVOKE_REQUEST: thread_id={user_input.thread_id}")
    # Model selection is handled by server configuration
    logger.info(f"INVOKE_REQUEST: message_length={len(user_input.message) if user_input.message else 0}")
    logger.info(f"INVOKE_REQUEST: agent_config={user_input.agent_config}")
    
    # NOTE: Currently this only returns the last message or interrupt.
    # In the case of an agent outputting multiple AIMessages (such as the background step
    # in interrupt-agent, or a tool step in research-assistant), it's omitted. Arguably,
    # you'd want to include it. You could update the API to return a list of ChatMessages
    # in that case.
    
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent, agent_id)
    
    logger.info(f"INVOKE_REQUEST: run_id={run_id}")
    logger.info(f"INVOKE_REQUEST: config_thread_id={kwargs['config']['configurable']['thread_id']}")
    
    # Subscribe to Realtime for this thread if enabled
    if settings.USE_SUPABASE_REALTIME:
        try:
            realtime_worker = app.state.realtime_worker
            thread_id = kwargs['config']['configurable']['thread_id']
            user_id = user_input.user_id or 1
            await realtime_worker.subscribe_to_thread(
                user_id=user_id,
                thread_id=thread_id,
                agent_id=agent_id
            )
        except AttributeError:
            # Realtime worker not initialized
            pass
        except Exception as e:
            logger.warning(f"Failed to subscribe to Realtime for thread {thread_id}: {e}")

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)

        # Get the latest state to include section data
        state = await agent.aget_state(config=kwargs["config"])
        if "current_section" in state.values:
            current_section_enum = state.values["current_section"]
            current_section_id = enum_value(current_section_enum)
            section_state = state.values.get("section_states", {}).get(current_section_id)
            # Choose the right section templates based on agent_id
            if agent_id == "xbuddy":
                section_templates = SECTION_TEMPLATES
            else:
                raise ValueError(f"Unknown agent: {agent_id}")

            section_template = section_templates.get(current_section_id)

            section_data = {
                "database_id": _SECTION_DISPLAY_POSITION.get(current_section_id),
                "name": section_template.name if section_template else "Unknown Section",
                "status": section_status(section_state),
            }
            output.custom_data["section"] = section_data

        # `state` was already read above for the active-section metadata; the
        # same snapshot is the authoritative source for the completion fields.
        completion = public_completion(state.values or {})
        invoke_response = InvokeResponse(
            output=output,
            thread_id=kwargs["config"]["configurable"]["thread_id"],
            user_id=kwargs["config"]["configurable"]["user_id"],
            collection_complete=completion.collection_complete,
            artifact_available=completion.artifact_available,
            sections=completion.sections,
        )
        
        # Log successful response
        logger.info(f"=== INVOKE_RESPONSE_SUCCESS: run_id={run_id} ===")
        logger.info(f"INVOKE_RESPONSE: output_type={output.type}")
        logger.info(f"INVOKE_RESPONSE: content_length={len(output.content) if output.content else 0}")
        logger.info(f"INVOKE_RESPONSE: thread_id={invoke_response.thread_id}")
        logger.info(f"INVOKE_RESPONSE: user_id={invoke_response.user_id}")
        logger.info(f"INVOKE_RESPONSE: has_custom_data={bool(output.custom_data)}")
        
        return invoke_response
    except Exception as e:
        logger.error(f"=== INVOKE_ERROR: run_id={run_id} ===")
        logger.error(f"INVOKE_ERROR: {str(e)}")
        logger.error(f"INVOKE_ERROR: agent_id={agent_id}")
        logger.error(f"INVOKE_ERROR: user_id={user_input.user_id}")
        logger.error(f"INVOKE_ERROR: thread_id={user_input.thread_id}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT, request: Request | None = None
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    # Log stream request summary
    thread_id_display = user_input.thread_id if user_input.thread_id else 'new'
    logger.info(f"[STREAM] Start: agent={agent_id}, user={user_input.user_id}, thread_id={thread_id_display}")
    
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent, agent_id)
    
    logger.debug(f"Stream run_id: {run_id}")
    
    # Subscribe to Realtime for this thread if enabled
    if settings.USE_SUPABASE_REALTIME and request and hasattr(request.app.state, 'realtime_worker'):
        realtime_worker = request.app.state.realtime_worker
        thread_id = kwargs['config']['configurable']['thread_id']
        user_id = user_input.user_id or 1
        try:
            await realtime_worker.subscribe_to_thread(
                user_id=user_id,
                thread_id=thread_id,
                agent_id=agent_id
            )
        except Exception as e:
            logger.warning(f"Failed to subscribe to Realtime for thread {thread_id}: {e}")

    # Get the current thread's message history length to filter out historical messages
    try:
        current_state = await agent.aget_state(config=kwargs["config"])
        initial_message_count = len(current_state.values.get("messages", []))
        logger.debug(f"Initial message count: {initial_message_count}")
    except Exception as e:
        logger.debug(f"Could not get initial message count: {e}")
        initial_message_count = 0

    sent_message_count = 0  # Track the number of messages sent to prevent duplicates

    # Distinguishes a clean finish from the error path; the completion event is
    # emitted only on the former.
    stream_completed_cleanly = False
    try:
        # Send metadata as the first event in the stream
        thread_id = kwargs["config"]["configurable"]["thread_id"]
        user_id = kwargs["config"]["configurable"]["user_id"]
        
        yield f"data: {json.dumps({'type': 'metadata', 'content': {'thread_id': thread_id, 'user_id': user_id, 'run_id': str(run_id)}})}\n\n"
        
        # Process streamed events from the graph and yield messages over the SSE stream.
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages", "custom"]
        ):
            # Log stream events efficiently
            if isinstance(stream_event, tuple):
                stream_mode, _ = stream_event
                logger.stream_event("received", {"mode": stream_mode})
            else:
                logger.stream_event("received", {"mode": "unknown"})

            if not isinstance(stream_event, tuple):
                continue
            stream_mode, event = stream_event
            new_messages = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    # A simple approach to handle agent interrupts.
                    # In a more sophisticated implementation, we could add
                    # some structured ChatMessage type to return the interrupt value.
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue
                    updates = updates or {}
                    
                    # STREAM_FIX: Only send NEW messages (not historical ones)
                    # Use initial_message_count to filter out messages that were already in the thread
                    update_messages = updates.get("messages", [])
                    
                    # Only add messages that are new (beyond the initial count)
                    if len(update_messages) > 0:
                        # If we have an initial count, only take messages after that position
                        if initial_message_count > 0 and len(update_messages) > initial_message_count:
                            new_messages.extend(update_messages[initial_message_count:])
                            logger.debug(f"Sending {len(update_messages[initial_message_count:])} new messages")
                        # If no initial count or this is the first batch, check against sent_message_count
                        elif len(update_messages) > sent_message_count:
                            new_messages.extend(update_messages[sent_message_count:])
                            sent_message_count = len(update_messages)
                            logger.debug(f"Sending {len(new_messages)} new messages")

            if stream_mode == "custom":
                new_messages = [event]

            # LangGraph streaming may emit tuples: (field_name, field_value)
            # e.g. ('content', <str>), ('tool_calls', [ToolCall,...]), ('additional_kwargs', {...}), etc.
            # We accumulate only supported fields into `parts` and skip unsupported metadata.
            # More info at: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    # Skip function calling related tuples - these are internal operations, not user messages
                    if key in ['tool_calls', 'additional_kwargs', 'invalid_tool_calls']:
                        logger.debug(f"Skipping function call tuple: {key}")
                        continue
                    # Store parts in temporary dict
                    logger.debug(f"Processing tuple: {key}")
                    current_message[key] = value
                else:
                    # Add complete message if we have one in progress
                    if current_message:
                        # Only process messages that contain content (user-facing messages)
                        if 'content' in current_message:
                            processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            # Add any remaining message parts
            if current_message:
                # Only process messages that contain content (user-facing messages)
                if 'content' in current_message:
                    processed_messages.append(_create_ai_message(current_message))

            for message in processed_messages:
                # TEMP DEBUG: print the raw message structure to diagnose content KeyError
                logger.info(f"🪵 RAW_MESSAGE: {repr(message)}")
                
                try:
                    # FIX: Skip processing for internal, content-less messages from structured_output calls
                    if isinstance(message, AIMessage) and not message.content:
                        if message.tool_calls or message.invalid_tool_calls:
                            logger.info(f"🚫 SKIPPING internal tool_call message: {repr(message)}")
                            continue
                    
                    logger.info(f"🔧 CONVERTING message type: {type(message).__name__}")
                    chat_message = langchain_to_chat_message(message)
                    logger.info(f"✅ CONVERSION SUCCESS for {type(message).__name__}")
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"❌ CONVERSION FAILED: {e}")
                    logger.error(f"❌ FAILED MESSAGE TYPE: {type(message).__name__}")
                    logger.error(f"❌ FAILED MESSAGE CONTENT: {repr(message)}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                # Skip messages with internal tags
                tags = metadata.get("tags", [])
                if any(tag in tags for tag in ["skip_stream", "internal_extraction", "do_not_stream", "internal_decision", "internal_synthesis"]):
                    logger.debug(f"Skipping message with internal tags: {tags}")
                    continue
                # Only process AIMessageChunk for token streaming
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # Empty content in the context of OpenAI usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content.
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
        # Reached only when the LangGraph stream finished without raising. The
        # finally block emits a completion event only in that case; an errored
        # stream keeps its existing error -> [DONE] shape with nothing synthetic
        # appended.
        stream_completed_cleanly = True
    except Exception as e:
        import traceback
        logger.error(f"[STREAM ERROR] {str(e)} (run_id={run_id}, agent={agent_id})")
        logger.error(f"[STREAM ERROR TRACEBACK]\n{traceback.format_exc()}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        # Always send section data at the end of the stream
        try:
            state = await agent.aget_state(config=kwargs["config"])
            if "current_section" in state.values:
                current_section_enum = state.values["current_section"]
                current_section_id = enum_value(current_section_enum)
                section_state = state.values.get("section_states", {}).get(current_section_id)
                
                # JobBuddy is the only agent this service serves; the removed
                # branches referenced six undefined *_TEMPLATES names.
                section_templates = SECTION_TEMPLATES

                section_template = section_templates.get(current_section_id)

                section_data = {
                    "database_id": _SECTION_DISPLAY_POSITION.get(current_section_id),
                    "name": section_template.name if section_template else "Unknown Section",
                    "status": section_status(section_state),
                }
                yield f"data: {json.dumps({'type': 'section', 'content': section_data})}\n\n"
        except Exception as e:
            logger.error(f"Error getting section data: {e}")

        # Log stream completion
        logger.info(f"[STREAM] Complete: agent={agent_id}, thread={kwargs['config']['configurable']['thread_id'][:8]}...")
        
        # One terminal structured event, immediately before [DONE], and only on a
        # clean finish. Read through the async state API so it reflects everything
        # the turn committed — including an artifact implementation_node wrote in
        # this same turn, which the pre-stream snapshot could not see.
        if stream_completed_cleanly:
            try:
                final_state = await agent.aget_state(config=kwargs["config"])
                completion = public_completion(final_state.values or {})
                yield f"data: {json.dumps({'type': 'completion', 'content': completion.model_dump()})}\n\n"
            except Exception as e:  # noqa: BLE001 - a completion read failure must not
                # turn a successful turn into a failed one; the client still gets its
                # messages and [DONE].
                logger.error(f"Error building completion event: {e}")

        yield "data: [DONE]\n\n"


def _create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    # Ensure content field is always present (AIMessage requires it)
    if 'content' not in filtered:
        filtered['content'] = ""
    return AIMessage(**filtered)


def _sse_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
@limiter.limit(settings.RATE_LIMIT_EXPENSIVE)
async def stream(
    request: Request, user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id, request),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    # Log feedback request
    logger.info(f"=== FEEDBACK_REQUEST: run_id={feedback.run_id} ===")
    logger.info(f"FEEDBACK_REQUEST: key={feedback.key}")
    logger.info(f"FEEDBACK_REQUEST: score={feedback.score}")
    logger.info(f"FEEDBACK_REQUEST: has_kwargs={bool(feedback.kwargs)}")
    
    try:
        client = LangsmithClient()
        kwargs = feedback.kwargs or {}
        client.create_feedback(
            run_id=feedback.run_id,
            key=feedback.key,
            score=feedback.score,
            **kwargs,
        )
        
        logger.info(f"=== FEEDBACK_SUCCESS: run_id={feedback.run_id} ===")
        return FeedbackResponse()
    except Exception as e:
        logger.error(f"=== FEEDBACK_ERROR: run_id={feedback.run_id} ===")
        logger.error(f"FEEDBACK_ERROR: {str(e)}")
        raise


async def load_chat_history(agent_id: str, thread_id: str, user_id: int) -> ChatHistory:
    """Read one thread's transcript from the checkpointer, scoped to its owner.

    Shared by every `/history` route so the default-agent and explicit-agent forms
    cannot diverge; `agent_id` is a parameter rather than a hardcoded DEFAULT_AGENT.

    **Async on purpose.** The previous implementation called the synchronous
    `agent.get_state`, which is the wrong API for the savers this service actually
    configures — `AsyncSqliteSaver` locally and `AsyncPostgresSaver` in production.
    `aget_state` is the one path that works against both.

    **Scoping, and why 404 rather than 403.** A thread whose stored `user_id`
    differs from the requested one is reported as not found. 403 would confirm the
    thread exists while refusing it, which tells an unauthorised caller strictly
    more than 404 does. The cost is usability: a client that sends the wrong
    `user_id` for its own thread sees "not found" rather than "wrong user", which is
    harder to debug.

    Honest limitation for this demo: authorization is a **single shared bearer
    token**, so every caller holding it is equally trusted and `user_id` is
    self-asserted rather than proven. This check prevents accidental cross-thread
    reads and gives per-user auth somewhere to plug in later; it is not an identity
    boundary. Note also that a nonexistent thread returns 200 with an empty list
    while a non-owned one returns 404, so a token holder can still distinguish the
    two. Closing that gap needs real identity, which is deliberately out of scope
    here.
    """
    try:
        agent: AgentGraph = get_agent(agent_id)
    except KeyError as exc:
        logger.warning(f"HISTORY_UNKNOWN_AGENT: agent_id={agent_id}")
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent_id}") from exc

    try:
        # Same thread configuration convention as /invoke and /stream.
        state_snapshot = await agent.aget_state(
            config=RunnableConfig(configurable={"thread_id": thread_id, "user_id": user_id})
        )
    except Exception as e:
        logger.error(f"=== HISTORY_ERROR: thread_id={thread_id} ===")
        logger.error(f"HISTORY_ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected error") from e

    values = state_snapshot.values or {}
    if not values:
        # Preserves the previous behaviour for an unknown or brand-new thread.
        logger.warning(f"HISTORY_WARNING: No state found for thread_id={thread_id}")
        return ChatHistory(thread_id=thread_id, user_id=user_id, messages=[])

    # Deny by default: a checkpoint without a stored user_id cannot be shown to
    # anyone, because ownership cannot be established. initialize_node always
    # writes it, so this only fires on a checkpoint written by something else.
    owner_id = values.get("user_id")
    if owner_id != user_id:
        logger.warning(
            f"HISTORY_SCOPE_MISMATCH: thread_id={thread_id} requested_by={user_id}"
        )
        raise HTTPException(status_code=404, detail="Thread not found")

    messages: list[AnyMessage] = values.get("messages", [])
    chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]

    logger.info(f"=== HISTORY_SUCCESS: thread_id={thread_id} ===")
    logger.info(f"HISTORY_SUCCESS: message_count={len(chat_messages)}")

    return ChatHistory(thread_id=thread_id, user_id=user_id, messages=chat_messages)


@router.post("/{agent_id}/history")
@router.post("/history")
async def history(input: ChatHistoryInput, agent_id: str = DEFAULT_AGENT) -> ChatHistory:
    """Get the conversation transcript for one thread.

    REST, not streaming: a history read has no incremental value, and the client
    wants the whole transcript at once.

    Two route forms, matching /invoke and /stream: the bare path uses the default
    agent, the prefixed path names one explicitly.
    """
    logger.info(
        f"=== HISTORY_REQUEST: agent_id={agent_id} thread_id={input.thread_id} "
        f"user_id={input.user_id} ==="
    )
    return await load_chat_history(agent_id, input.thread_id, input.user_id)


@router.get("/check_agent_state/{agent_id}")
async def check_agent_state(
    agent_id: str,
    user_id: int,
    thread_id: str,
    section_id: str | None = None,
):
    """
    Check if agent has read the latest database updates.
    
    This endpoint compares:
    1. Current section content in Supabase database
    2. Current section content in LangGraph agent state
    
    Returns a comparison showing if they match.
    
    Args:
        agent_id: Agent identifier (e.g., "xbuddy")
        user_id: User identifier
        thread_id: Thread/conversation identifier
        section_id: Optional section ID to check (e.g., "mission", "idea")
                    If not provided, checks all sections
    
    Returns:
        Comparison result showing database vs agent state
    """
    logger.info(f"=== CHECK_AGENT_STATE: agent_id={agent_id}, thread_id={thread_id} ===")
    
    try:
        # Get agent
        agent: AgentGraph = get_agent(agent_id)
        
        # Get agent state
        config = RunnableConfig(
            configurable={
                "thread_id": thread_id,
                "user_id": user_id
            }
        )
        
        state_snapshot = await agent.aget_state(config)
        if not state_snapshot or not state_snapshot.values:
            return {
                "success": False,
                "message": "No agent state found for this thread",
                "thread_id": thread_id,
            }
        
        agent_state = state_snapshot.values
        agent_section_states = agent_state.get("section_states", {})
        
        # Get database state from Supabase
        from integrations.supabase.supabase_client import SupabaseClient
        client = SupabaseClient()
        
        # Build query
        query = client.client.table("section_states")\
            .select("*")\
            .eq("user_id", user_id)\
            .eq("thread_id", thread_id)\
            .eq("agent_id", agent_id)
        
        if section_id:
            query = query.eq("section_id", section_id)
        
        result = query.execute()
        db_section_states = result.data if result.data else []
        
        # Compare database vs agent state
        comparison = []
        
        if section_id:
            # Check specific section
            db_state = next((s for s in db_section_states if s["section_id"] == section_id), None)
            agent_state_data = agent_section_states.get(section_id)
            
            if db_state:
                db_content = db_state.get("content", {})
                db_plain_text = db_state.get("plain_text", "")
                db_updated = db_state.get("updated_at")
                
                if agent_state_data:
                    # agent_state_data is a SectionState Pydantic model, not a dict
                    agent_content = agent_state_data.content
                    if agent_content:
                        agent_plain_text = agent_content.plain_text or ""
                        # Extract text from TiptapDocument if needed
                        if hasattr(agent_content, 'content') and agent_content.content:
                            # Try to extract text from Tiptap structure
                            def get_text_from_tiptap(node):
                                if hasattr(node, 'content') and node.content:
                                    if hasattr(node, 'text'):
                                        return node.text
                                    return "".join(get_text_from_tiptap(c) for c in node.content)
                                return ""
                            agent_plain_text = "".join(get_text_from_tiptap(p) for p in agent_content.content.content) if agent_content.content.content else agent_plain_text
                    else:
                        agent_plain_text = ""
                    agent_status = agent_state_data.status.value if hasattr(agent_state_data.status, 'value') else str(agent_state_data.status)
                else:
                    agent_content = None
                    agent_plain_text = ""
                    agent_status = "not_in_state"
                
                # Extract text for comparison
                def extract_text(data):
                    if isinstance(data, str):
                        return data
                    if isinstance(data, dict):
                        if "plain_text" in data:
                            return data["plain_text"]
                        if "text" in data:
                            return data["text"]
                        # Try to extract from Tiptap structure
                        if "content" in data and isinstance(data["content"], list):
                            def get_text(node):
                                if isinstance(node, dict):
                                    if "text" in node:
                                        return node["text"]
                                    if "content" in node:
                                        return "".join(get_text(c) for c in node["content"])
                                return ""
                            return "".join(get_text(n) for n in data["content"])
                    return ""
                
                db_text = extract_text(db_content) or db_plain_text
                agent_text = agent_plain_text or extract_text(agent_content)
                
                comparison.append({
                    "section_id": section_id,
                    "database": {
                        "has_content": bool(db_content),
                        "text_preview": db_text[:200] if db_text else None,
                        "text_length": len(db_text),
                        "updated_at": db_updated,
                    },
                    "agent_state": {
                        "has_content": bool(agent_content),
                        "text_preview": agent_text[:200] if agent_text else None,
                        "text_length": len(agent_text),
                        "status": agent_status,
                    },
                    "match": db_text.strip() == agent_text.strip() if (db_text and agent_text) else False,
                    "synced": bool(agent_content) and db_text.strip() == agent_text.strip(),
                })
            else:
                comparison.append({
                    "section_id": section_id,
                    "database": {"has_content": False},
                    "agent_state": {
                        "has_content": bool(agent_state_data),
                        "status": agent_state_data.status.value if agent_state_data and hasattr(agent_state_data.status, 'value') else (str(agent_state_data.status) if agent_state_data else "not_in_state"),
                    },
                    "match": False,
                    "synced": False,
                    "message": "Section not found in database",
                })
        else:
            # Check all sections
            db_sections_by_id = {s["section_id"]: s for s in db_section_states}
            
            all_section_ids = set(list(db_sections_by_id.keys()) + list(agent_section_states.keys()))
            
            for sid in all_section_ids:
                db_state = db_sections_by_id.get(sid)
                agent_state_data = agent_section_states.get(sid)
                
                if db_state:
                    db_content = db_state.get("content", {})
                    db_plain_text = db_state.get("plain_text", "")
                    db_updated = db_state.get("updated_at")
                    
                    def extract_text(data):
                        if isinstance(data, str):
                            return data
                        if isinstance(data, dict):
                            if "plain_text" in data:
                                return data["plain_text"]
                            if "text" in data:
                                return data["text"]
                        return ""
                    
                    db_text = extract_text(db_content) or db_plain_text
                    
                    if agent_state_data:
                        # agent_state_data is a SectionState Pydantic model, not a dict
                        agent_content = agent_state_data.content
                        if agent_content:
                            agent_plain_text = agent_content.plain_text or ""
                            # Extract text from TiptapDocument if needed
                            if hasattr(agent_content, 'content') and agent_content.content:
                                def get_text_from_tiptap(node):
                                    if hasattr(node, 'content') and node.content:
                                        if hasattr(node, 'text'):
                                            return node.text
                                        return "".join(get_text_from_tiptap(c) for c in node.content)
                                    return ""
                                agent_plain_text = "".join(get_text_from_tiptap(p) for p in agent_content.content.content) if agent_content.content.content else agent_plain_text
                        else:
                            agent_plain_text = ""
                        agent_text = agent_plain_text or extract_text(agent_content) if agent_content else ""
                        agent_status = agent_state_data.status.value if hasattr(agent_state_data.status, 'value') else str(agent_state_data.status)
                    else:
                        agent_text = ""
                        agent_status = "not_in_state"
                    
                    comparison.append({
                        "section_id": sid,
                        "database": {
                            "has_content": bool(db_content),
                            "text_length": len(db_text),
                            "updated_at": db_updated,
                        },
                        "agent_state": {
                            "has_content": bool(agent_state_data),
                            "text_length": len(agent_text),
                            "status": agent_status,
                        },
                        "match": db_text.strip() == agent_text.strip() if (db_text and agent_text) else False,
                        "synced": bool(agent_state_data) and db_text.strip() == agent_text.strip(),
                    })
                else:
                    comparison.append({
                        "section_id": sid,
                        "database": {"has_content": False},
                    "agent_state": {
                        "has_content": bool(agent_state_data),
                        "status": agent_state_data.status.value if agent_state_data and hasattr(agent_state_data.status, 'value') else (str(agent_state_data.status) if agent_state_data else "not_in_state"),
                    },
                        "match": False,
                        "synced": False,
                    })
        
        # Check business_plan from business_plans table
        business_plan_comparison = {
            "database": {"has_content": False},
            "agent_state": {"has_content": False},
            "match": False,
            "synced": False,
        }
        try:
            bp_result = client.client.table("business_plans")\
                .select("*")\
                .eq("user_id", user_id)\
                .eq("thread_id", thread_id)\
                .eq("agent_id", agent_id)\
                .maybe_single()\
                .execute()
            
            db_business_plan = bp_result.data if bp_result.data else None
            agent_business_plan = agent_state.get("business_plan")
            
            # Always create comparison, even if both are None
            db_bp_content = db_business_plan.get("content") if db_business_plan else None
            db_bp_markdown = db_business_plan.get("markdown_content") if db_business_plan else None
            db_bp_text = db_bp_markdown or db_bp_content or ""
            
            agent_bp_text = agent_business_plan or ""
            
            # Normalize for comparison (strip whitespace)
            db_bp_text_normalized = db_bp_text.strip() if db_bp_text else ""
            agent_bp_text_normalized = agent_bp_text.strip() if agent_bp_text else ""
            
            business_plan_comparison = {
                "database": {
                    "has_content": bool(db_bp_text),
                    "text_length": len(db_bp_text),
                    "text_preview": db_bp_text[:500] if db_bp_text else None,
                    "updated_at": db_business_plan.get("updated_at") if db_business_plan else None,
                },
                "agent_state": {
                    "has_content": bool(agent_bp_text),
                    "text_length": len(agent_bp_text),
                    "text_preview": agent_bp_text[:500] if agent_bp_text else None,
                },
                "match": db_bp_text_normalized == agent_bp_text_normalized if (db_bp_text_normalized and agent_bp_text_normalized) else False,
                "synced": bool(agent_bp_text) and db_bp_text_normalized == agent_bp_text_normalized,
            }
        except Exception as e:
            logger.warning(f"Could not check business_plan: {e}")
            import traceback
            logger.debug(f"Business plan check error traceback: {traceback.format_exc()}")
            # Keep default comparison with has_content: False
        
        # Summary
        synced_count = sum(1 for c in comparison if c.get("synced", False))
        total_count = len(comparison)
        
        result = {
            "success": True,
            "thread_id": thread_id,
            "user_id": user_id,
            "summary": {
                "total_sections": total_count,
                "synced_sections": synced_count,
                "unsynced_sections": total_count - synced_count,
                "all_synced": synced_count == total_count and total_count > 0,
            },
            "comparison": comparison,
        }
        
        # Always include business_plan comparison
        result["business_plan"] = business_plan_comparison
        if business_plan_comparison.get("synced"):
            result["summary"]["business_plan_synced"] = True
        else:
            result["summary"]["business_plan_synced"] = False
        
        return result
        
    except Exception as e:
        logger.error(f"Error checking agent state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error checking agent state: {str(e)}")


@router.get("/get_agent_state/{agent_id}")
async def get_agent_state(
    agent_id: str,
    user_id: int,
    thread_id: str,
):
    """
    Get the complete agent state including all section texts.
    
    This endpoint returns the full LangGraph agent state, including:
    - All section states with their content
    - Current section
    - Messages
    - Any other state data
    
    Args:
        agent_id: Agent identifier (e.g., "xbuddy")
        user_id: User identifier
        thread_id: Thread/conversation identifier
    
    Returns:
        Complete agent state with all section texts
    """
    logger.info(f"=== GET_AGENT_STATE: agent_id={agent_id}, thread_id={thread_id} ===")
    
    try:
        # Get agent
        agent: AgentGraph = get_agent(agent_id)
        
        # Get agent state
        config = RunnableConfig(
            configurable={
                "thread_id": thread_id,
                "user_id": user_id
            }
        )
        
        state_snapshot = await agent.aget_state(config)
        if not state_snapshot or not state_snapshot.values:
            return {
                "success": False,
                "message": "No agent state found for this thread",
                "thread_id": thread_id,
            }
        
        agent_state = state_snapshot.values
        
        # Extract section states with readable text
        section_states = {}
        if "section_states" in agent_state:
            for section_id, section_data in agent_state["section_states"].items():
                # section_data is a SectionState Pydantic model, not a dict
                content = section_data.content if hasattr(section_data, 'content') else None
                plain_text = None
                
                # Extract plain text from content
                if content:
                    plain_text = content.plain_text if hasattr(content, 'plain_text') else None
                    if not plain_text and hasattr(content, 'content') and content.content:
                        # Try to extract from Tiptap structure
                        def extract_text(node):
                            if isinstance(node, str):
                                return node
                            if hasattr(node, 'text'):
                                return node.text
                            if hasattr(node, 'content') and node.content:
                                return "".join(extract_text(c) for c in node.content)
                            if isinstance(node, dict):
                                if "text" in node:
                                    return node["text"]
                                if "content" in node and isinstance(node["content"], list):
                                    return "".join(extract_text(c) for c in node["content"])
                            return ""
                        if hasattr(content, 'content') and content.content:
                            if hasattr(content.content, 'content'):
                                plain_text = "".join(extract_text(c) for c in content.content.content)
                            else:
                                plain_text = extract_text(content.content)
                
                # section_data is a SectionState Pydantic model
                status_value = section_data.status.value if hasattr(section_data.status, 'value') else str(section_data.status)
                satisfaction_status = section_data.satisfaction_status if hasattr(section_data, 'satisfaction_status') else None
                
                section_states[section_id] = {
                    "section_id": section_id,
                    "status": status_value,
                    "satisfaction_status": satisfaction_status,
                    "plain_text": plain_text or "",
                    "content_preview": str(content)[:500] if content else None,
                    "has_content": bool(content),
                }
        
        # Extract messages (last few for context)
        messages = []
        if "messages" in agent_state:
            for msg in agent_state["messages"][-5:]:  # Last 5 messages
                if hasattr(msg, "content"):
                    content = msg.content
                elif isinstance(msg, dict):
                    content = msg.get("content", "")
                else:
                    content = str(msg)
                
                messages.append({
                    "type": type(msg).__name__,
                    "content": content[:200] if isinstance(content, str) else str(content)[:200],
                })
        
        return {
            "success": True,
            "thread_id": thread_id,
            "user_id": user_id,
            "current_section": str(agent_state.get("current_section", "unknown")),
            "section_states": section_states,
            "messages_count": len(agent_state.get("messages", [])),
            "recent_messages": messages,
            "state_keys": list(agent_state.keys()),
        }
        
    except Exception as e:
        logger.error(f"Error getting agent state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting agent state: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""

    health_status = {"status": "ok"}

    if settings.LANGFUSE_TRACING:
        try:
            langfuse = Langfuse()
            health_status["langfuse"] = "connected" if langfuse.auth_check() else "disconnected"
        except Exception as e:
            logger.error(f"Langfuse connection error: {e}")
            health_status["langfuse"] = "disconnected"

    return health_status


@router.post("/realtime/subscribe")
async def subscribe_to_realtime(
    request: Request,
    user_id: int | None = None,
    thread_id: str | None = None,
    agent_id: str = "xbuddy",
):
    """
    Manually subscribe to Realtime events for a specific thread.
    
    This endpoint is useful when:
    - User refreshes the page and needs to re-establish subscription
    - User opens BusinessPlanEditor and wants to ensure subscription is active
    - Frontend wants to explicitly establish subscription before editing
    
    Args:
        user_id: User identifier
        thread_id: Thread/conversation identifier
        agent_id: Agent identifier (default: "xbuddy")
        request: FastAPI Request object (for accessing app.state)
    
    Returns:
        Success status and subscription info
    """
    # Try to get parameters from request body if not provided as query params
    if user_id is None or thread_id is None:
        try:
            body = await request.json()
            user_id = user_id or body.get("user_id")
            thread_id = thread_id or body.get("thread_id")
            agent_id = body.get("agent_id", agent_id)
        except:
            pass
    
    # Validate required parameters
    if not user_id or not thread_id:
        raise HTTPException(
            status_code=422,
            detail="Missing required parameters: user_id and thread_id are required"
        )
    
    if not settings.USE_SUPABASE_REALTIME:
        return {
            "success": False,
            "message": "Realtime is disabled. Set USE_SUPABASE_REALTIME=true to enable."
        }
    
    if not request or not hasattr(request.app.state, 'realtime_worker'):
        return {
            "success": False,
            "message": "Realtime worker not initialized"
        }
    
    try:
        realtime_worker = request.app.state.realtime_worker
        
        # Check if already subscribed
        if thread_id in realtime_worker.subscriptions:
            logger.info(f"Already subscribed to thread {thread_id}")
            return {
                "success": True,
                "message": "Already subscribed",
                "thread_id": thread_id,
                "subscription_count": realtime_worker.get_subscription_count()
            }
        
        # Subscribe
        await realtime_worker.subscribe_to_thread(
            user_id=user_id,
            thread_id=thread_id,
            agent_id=agent_id
        )
        
        logger.info(f"✅ Manual subscription established for thread {thread_id}")
        return {
            "success": True,
            "message": "Subscription established",
            "thread_id": thread_id,
            "subscription_count": realtime_worker.get_subscription_count()
        }
    except Exception as e:
        logger.error(f"❌ Failed to establish subscription: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Failed to establish subscription: {str(e)}"
        }


app.include_router(router)
