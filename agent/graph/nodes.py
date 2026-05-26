"""LangGraph node functions for the agent reasoning loop.

Each function is a pure node: it receives the current :class:`AgentState`
and returns a *partial* state update (a dict). LangGraph merges the update
with the existing state using the registered reducers.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from agent.graph.state import AgentState

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
        ``"tools"`` if the last AI message contains tool calls, else ``"end"``.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.debug("Routing → tools (%d calls).", len(last_message.tool_calls))
        return "tools"
    logger.debug("Routing → end.")
    return "end"
