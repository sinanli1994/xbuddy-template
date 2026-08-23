from typing import Any, Literal, NotRequired

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from core.models import AllModelEnum


class AgentInfo(BaseModel):
    """Info about an available agent."""

    key: str = Field(
        description="Agent key.",
        examples=["research-assistant"],
    )
    description: str = Field(
        description="Description of the agent.",
        examples=["A research assistant for generating research papers."],
    )


class EndpointInfo(BaseModel):
    """Info about an available API endpoint."""

    path: str = Field(
        description="API endpoint path pattern.",
        examples=["/sync_section/{agent_id}/{section_id}"],
    )
    method: str = Field(
        description="HTTP method.",
        examples=["POST", "GET"],
    )
    description: str = Field(
        description="Description of what this endpoint does.",
        examples=["Sync LangGraph state with manually edited section content from database"],
    )
    parameters: dict[str, str] = Field(
        description="Parameters and their descriptions.",
        examples=[{
            "agent_id": "Agent identifier",
            "section_id": "Section identifier",
            "user_id": "User identifier (query param)",
            "thread_id": "Thread identifier (query param)"
        }],
    )
    example: str | None = Field(
        description="Example URL for this endpoint.",
        default=None,
        examples=["/sync_section/value-canvas/icp?user_id=12&thread_id=abc-123"],
    )


class ServiceMetadata(BaseModel):
    """Metadata about the service including available agents and models."""

    agents: list[AgentInfo] = Field(
        description="List of available agents.",
    )
    models: list[AllModelEnum] = Field(
        description="List of available LLMs.",
    )
    default_agent: str = Field(
        description="Default agent used when none is specified.",
        examples=["research-assistant"],
    )
    default_model: AllModelEnum | None = Field(
        description="Default model (server-managed, not user-selectable).",
        default=None,
    )
    endpoints: list[EndpointInfo] = Field(
        description="List of available API endpoints.",
        default=[],
    )


class UserInput(BaseModel):
    """Basic user input for the agent."""

    message: str = Field(
        description="User input to the agent.",
        examples=["What is the weather in Tokyo?"],
    )
    thread_id: str | None = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: int | None = Field(
        description="User ID to persist and continue a conversation across multiple threads.",
        default=None,
        examples=[1, 123],
    )
    agent_config: dict[str, Any] = Field(
        description="Additional configuration to pass through to the agent",
        default={},
        examples=[{"spicy_level": 0.8}],
    )


class StreamInput(UserInput):
    """User input for streaming the agent's response."""

    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client.",
        default=True,
    )


class ToolCall(TypedDict):
    """Represents a request to call a tool."""

    name: str
    """The name of the tool to be called."""
    args: dict[str, Any]
    """The arguments to the tool call."""
    id: str | None
    """An identifier associated with the tool call."""
    type: NotRequired[Literal["tool_call"]]


class ChatMessage(BaseModel):
    """Message in a chat."""

    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool", "custom"],
    )
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: list[ToolCall] = Field(
        description="Tool calls in the message.",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="Tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="Run ID of the message.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    response_metadata: dict[str, Any] = Field(
        description="Response metadata. For example: response headers, logprobs, token counts.",
        default={},
    )
    custom_data: dict[str, Any] = Field(
        description="Custom message data.",
        default={},
    )

    def pretty_repr(self) -> str:
        """Get a pretty representation of the message."""
        base_title = self.type.title() + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        print(self.pretty_repr())


class Feedback(BaseModel):  # type: ignore[no-redef]
    """Feedback for a run, to record to LangSmith."""

    run_id: str = Field(
        description="Run ID to record feedback for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="Feedback key.",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="Feedback score.",
        examples=[0.8],
    )
    kwargs: dict[str, Any] = Field(
        description="Additional feedback kwargs, passed to LangSmith.",
        default={},
        examples=[{"comment": "In-line human feedback"}],
    )


class FeedbackResponse(BaseModel):
    status: Literal["success"] = "success"


class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history.

    Both identifiers are required. `thread_id` selects the checkpoint; `user_id`
    scopes the read, so a caller cannot page through other people's threads by
    guessing ids. This is a deliberate breaking change to the previous
    thread-id-only request: the old shape had no scoping at all.
    """

    thread_id: str = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: int = Field(
        description="User the thread must belong to. Required; the read is scoped to it.",
        examples=[1, 123],
    )


class ChatHistory(BaseModel):
    """Conversation messages for one thread.

    Messages only. No `user_data`, no `section_states`, no `final_output`, no
    router directives, no persistence flags, no `finished` — a history reader needs
    the transcript, not the graph. Progress and completion are the job of
    `/invoke`'s `CompletionState` projection.

    `thread_id` and `user_id` are echoed so a client can correlate a response with
    the request that produced it.
    """

    thread_id: str = Field(
        description="The thread these messages belong to.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: int = Field(
        description="The user the thread belongs to.",
        examples=[1, 123],
    )
    messages: list[ChatMessage] = Field(
        description="The conversation, oldest first.",
    )


class PublicSection(BaseModel):
    """One section, as the outside world sees it.

    Exactly three fields. No `database_id` (a FounderBuddy UI position), no draft
    content, no satisfaction flags — a public client needs to render progress, not
    inspect graph state.
    """

    id: str = Field(
        description="Canonical section identifier.",
        examples=["career_goal", "action_plan"],
    )
    name: str = Field(
        description="Human-readable section name.",
        examples=["Career Goal", "Action Plan"],
    )
    status: str = Field(
        description="One of pending, in_progress, done.",
        examples=["pending", "in_progress", "done"],
    )


class CompletionState(BaseModel):
    """The narrow public completion projection.

    Two independent booleans, because they answer different questions and can
    legitimately disagree — a completed conversation whose synthesis failed is
    `collection_complete=True, artifact_available=False`.

    `finished` is deliberately **not** exposed, even though Issue #10 fixed its
    semantics — it is now derived from the same section-completion rule as
    `should_generate_final_output`, so it is no longer wrong, just internal.

    Two reasons to keep it inside. It is graph state whose meaning is owned by the
    agent, so publishing it would couple every client to a routing concept they have
    no use for; and `collection_complete` plus `artifact_available` already answer the
    two questions a client actually asks — is the interview over, and is there
    something to render.
    """

    collection_complete: bool = Field(
        description=(
            "Every one of the five sections is done. Derived from the agent's "
            "should_generate_final_output, which is computed from section statuses."
        ),
    )
    artifact_available: bool = Field(
        description="A final artifact exists for this thread.",
    )
    sections: list[PublicSection] = Field(
        description="All five sections in canonical order.",
    )


class InvokeResponse(CompletionState):
    """Response from an agent invocation.

    Inherits the completion projection, so `output`/`thread_id`/`user_id` keep
    their existing meaning and shape and the three new keys are purely additive.
    """

    output: ChatMessage = Field(
        description="The output of the agent.",
    )
    thread_id: str = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: int = Field(
        description="User ID to persist and continue a conversation across multiple threads.",
        examples=[1, 123],
    )


class RefineSectionInput(BaseModel):
    """Input for refining a section with AI."""

    user_id: int = Field(
        description="User identifier.",
        examples=[1, 123],
    )
    thread_id: str = Field(
        description="Thread/conversation identifier.",
        min_length=1,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    refinement_prompt: str = Field(
        description="User's instruction for how to refine the section content.",
        min_length=1,
        examples=["Make it more concise", "Add more details about the target audience"],
    )
