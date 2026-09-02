"""Graph builder for your XBuddy Agent.

This is the core of your LangGraph agent. It wires together all the nodes
into a StateGraph with conditional edges.

Study FounderBuddy's builder.py:
https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/graph/builder.py
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from ..models import XBuddyState
from ..nodes import (
    generate_decision_node,
    generate_reply_node,
    implementation_node,
    initialize_node,
    memory_updater_node,
    router_node,
)
from ..nodes.process_confirmation import process_confirmation_node
from .routes import (
    route_after_implementation,
    route_after_memory_updater,
    route_after_reply,
    route_turn,
)


def build_xbuddy_graph():
    """Build the XBuddy agent graph.

    Collection: router -> reply -> decision -> memory -> router -> END.
    Review: router -> process_confirmation -> memory -> router -> reply -> END.
    Final confirmation: memory -> implementation -> END (one readiness reply).
    An unagreed Action Plan uses explicit proposal mode and ends after that reply.
    """
    graph = StateGraph(XBuddyState)

    # Add nodes
    graph.add_node("initialize", initialize_node)
    graph.add_node("router", router_node)
    graph.add_node("generate_reply", generate_reply_node)
    graph.add_node("generate_decision", generate_decision_node)
    graph.add_node("memory_updater", memory_updater_node)
    graph.add_node("implementation", implementation_node)
    graph.add_node("process_confirmation", process_confirmation_node)

    # Add edges
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "router")

    graph.add_conditional_edges(
        "router",
        route_turn,
        {
            "process_confirmation": "process_confirmation",
            "generate_reply": "generate_reply",
            None: END,
        },
    )

    graph.add_edge("process_confirmation", "memory_updater")
    graph.add_conditional_edges(
        "generate_reply", route_after_reply,
        {"generate_decision": "generate_decision", None: END},
    )
    graph.add_edge("generate_decision", "memory_updater")

    graph.add_conditional_edges(
        "memory_updater",
        route_after_memory_updater,
        {
            "implementation": "implementation",
            "router": "router",
        },
    )

    graph.add_conditional_edges(
        "implementation", route_after_implementation, {"router": "router", None: END},
    )

    # Compile with memory checkpointer
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
