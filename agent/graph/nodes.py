"""LangGraph node functions for the agent reasoning loop.

Each function is a pure node: it receives the current :class:`AgentState`
and returns a *partial* state update (a dict). LangGraph merges the update
with the existing state using the registered reducers.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from langgraph.types import interrupt

from agent.graph.state import AgentState
from agent.guardrails import Guardrail

logger = logging.getLogger(__name__)


def make_agent_node(llm_with_tools: BaseChatModel):
    """Factory that creates the agent reasoning node.

    Separating node creation from the graph wiring allows the LLM to be
    injected (and easily swapped in tests) without the node function carrying
    any global state.

    Args:
        llm_with_tools: An LLM instance already bound to the skill tools via
            ``llm.bind_tools(tools)``.

    Returns:
        A LangGraph-compatible node function.
    """

    def agent_node(state: AgentState) -> dict:
        """Call the LLM with the current conversation history.

        Args:
            state: Current graph state carrying the message history.

        Returns:
            A partial state update ``{"messages": [ai_message]}``.
        """
        logger.debug("agent_node invoked with %d messages.", len(state["messages"]))
        response: AIMessage = llm_with_tools.invoke(state["messages"])
        logger.debug(
            "agent_node received response (tool_calls=%d).",
            len(response.tool_calls) if response.tool_calls else 0,
        )
        return {"messages": [response]}

    return agent_node


def should_continue(state: AgentState) -> str:
    """Routing function: decide whether to call a tool or finish.

    Args:
        state: Current graph state.

    Returns:
        ``"guardrails"`` if the last AI message contains tool calls, else ``"end"``.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.debug("Routing → guardrails (%d calls).", len(last_message.tool_calls))
        return "guardrails"
    logger.debug("Routing → end.")
    return "end"

def make_guardrail_node(guardrails: List[Guardrail]):
    """Factory that creates the guardrail node.

    Args:
        guardrails: List of registered guardrails.

    Returns:
        A LangGraph-compatible node function.
    """
    def guardrail_node(state: AgentState) -> dict:
        """Check the last AI message's tool calls against registered guardrails."""
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}

        for tool_call in last_message.tool_calls:
            for guardrail in guardrails:
                prompt = guardrail.check(tool_call)
                if prompt:
                    logger.info("Guardrail '%s' triggered.", guardrail.name)
                    # Use LangGraph interrupt to pause for human confirmation
                    response = interrupt(prompt)

                    # If the user says anything other than "yes", cancel all tool calls
                    # and inject a deterministic AIMessage so the LLM is never re-invoked.
                    # This prevents the LLM from hallucinating a "success" response.
                    if str(response).lower().strip() not in {"yes", "y", "true", "1"}:
                        logger.info("User declined guardrail confirmation.")
                        cancellation_tool_messages = [
                            ToolMessage(
                                tool_call_id=tc["id"],
                                content="Action cancelled by user.",
                            )
                            for tc in last_message.tool_calls
                        ]
                        # Append a hardcoded AIMessage so the graph routes to END
                        # without ever calling the LLM again (avoids hallucination).
                        cancellation_reply = AIMessage(
                            content="Action cancelled. The operation was not performed — nothing has been deleted or changed."
                        )
                        return {"messages": cancellation_tool_messages + [cancellation_reply]}

        return {}

    return guardrail_node

def should_continue_from_guardrails(state: AgentState) -> str:
    """Routing function: decide whether to proceed to tools, end, or go back to agent.

    Args:
        state: Current graph state.

    Returns:
        - ``"tools"`` – guardrails passed; last message is the original AIMessage
          that still carries tool_calls.
        - ``"end"`` – guardrail was triggered and the user declined; last message
          is the injected cancellation AIMessage (no tool_calls), so we skip the
          LLM entirely and terminate the turn.
        - ``"agent"`` – fallback (should not normally be reached).
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage):
        # Original agent AIMessage has tool_calls → safe to proceed to tools.
        # Injected cancellation AIMessage has no tool_calls → go straight to end.
        if last_message.tool_calls:
            return "tools"
        return "end"
    # Fallback: ToolMessage present without a cancellation AIMessage.
    return "agent"
