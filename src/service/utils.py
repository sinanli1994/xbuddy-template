from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)

from schema import ChatMessage


def _get_section_name(agent_name: str, section_id_str: str) -> str:
    """
    Get the section name for a given agent and section ID.

    Args:
        agent_name: Agent identifier (e.g., "value-canvas", "mission-pitch")
        section_id_str: Section string ID (e.g., "interview", "icp", "pain")

    Returns:
        Human-readable section name
    """
    # Map agent names to their SECTION_TEMPLATES module paths
    agent_template_map = {
        "value-canvas": "agents.value_canvas.prompts",
        "mission-pitch": "agents.mission_pitch.prompts",
        "social-pitch": "agents.social_pitch.prompts",
        "signature-pitch": "agents.signature_pitch.prompts",
        "special-report": "agents.special_report.prompts",
        "concept-pitch": "agents.concept_pitch.prompts",
    }

    module_path = agent_template_map.get(agent_name)

    if module_path:
        try:
            # Dynamic import of SECTION_TEMPLATES from the agent's prompts module
            import importlib
            module = importlib.import_module(module_path)
            section_templates = getattr(module, "SECTION_TEMPLATES", {})
            template = section_templates.get(section_id_str)
            if template:
                return template.name
        except (ImportError, AttributeError):
            pass

    # Fallback: format section_id_str nicely
    return section_id_str.replace("_", " ").title()


def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        # Safely handle dict items that may not contain a "type" key
        if not isinstance(content_item, dict):
            # Unexpected type, just cast to string
            text.append(str(content_item))
            continue

        item_type = content_item.get("type")
        if item_type == "text":
            text.append(content_item.get("text", ""))
        elif "text" in content_item:
            # Fallback: treat any dict containing text field as text content
            text.append(content_item["text"])
    return "".join(text)


def langchain_to_chat_message(message: BaseMessage) -> ChatMessage:
    """Create a ChatMessage from a LangChain message."""
    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                content=convert_message_content_to_string(message.content),
            )

            # Extract section metadata from additional_kwargs (similar to AIMessage handling)
            if message.additional_kwargs:
                section_id_str = message.additional_kwargs.get("section_id")
                agent_name = message.additional_kwargs.get("agent_name")

                if section_id_str and agent_name:
                    # section_id stays a string: section_states.section_id is TEXT
                    # and the frontend already expects a string for this agent.
                    section_name = _get_section_name(agent_name, section_id_str)

                    # Add to custom_data (saved_section is always False for human messages)
                    human_message.custom_data.update({
                        "section_id": section_id_str,
                        "section_name": section_name,
                        "agent_name": agent_name,
                        "saved_section": False,
                    })

            return human_message
        case AIMessage():
            ai_message = ChatMessage(
                type="ai",
                content=convert_message_content_to_string(message.content),
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata

            # Extract section metadata from additional_kwargs
            if message.additional_kwargs:
                section_id_str = message.additional_kwargs.get("section_id")
                agent_name = message.additional_kwargs.get("agent_name")

                if section_id_str and agent_name:
                    # section_id stays a string: section_states.section_id is TEXT
                    # and the frontend already expects a string for this agent.
                    section_name = _get_section_name(agent_name, section_id_str)

                    # Check if this message triggered a save operation
                    # This flag is set by memory_updater node after successful save
                    saved_section = message.additional_kwargs.get("triggered_save", False)

                    # Add to custom_data
                    ai_message.custom_data.update({
                        "section_id": section_id_str,
                        "section_name": section_name,
                        "agent_name": agent_name,
                        "saved_section": saved_section
                    })

            return ai_message
        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                content=convert_message_content_to_string(message.content),
                tool_call_id=message.tool_call_id,
            )
            return tool_message
        case LangchainChatMessage():
            if message.role == "custom":
                # Safely handle custom message content
                custom_data = {}
                if isinstance(message.content, list) and len(message.content) > 0:
                    custom_data = message.content[0]
                elif isinstance(message.content, dict):
                    custom_data = message.content
                
                custom_message = ChatMessage(
                    type="custom",
                    content="",
                    custom_data=custom_data,
                )
                return custom_message
            else:
                raise ValueError(f"Unsupported chat message role: {message.role}")
        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")


def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or (isinstance(content_item, dict) and content_item.get("type") != "tool_use")
    ]
