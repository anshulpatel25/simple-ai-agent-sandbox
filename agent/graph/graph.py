"""LangGraph graph construction.

The graph implements the standard ReAct pattern:

    START → agent → (guardrails → tools? → agent)* → END

``build_graph`` is a pure factory – it accepts the LLM and tools as arguments
so it can be unit-tested independently of Docker and LM Studio.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.graph.nodes import (
    make_agent_node,
    should_continue,
    make_guardrail_node,
    should_continue_from_guardrails
)
from agent.graph.state import AgentState
from agent.guardrails import Guardrail

logger = logging.getLogger(__name__)

# Node name constants – avoids magic strings scattered throughout.
NODE_AGENT = "agent"
NODE_TOOLS = "tools"
NODE_GUARDRAILS = "guardrails"


def build_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    guardrails: list[Guardrail],
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> StateGraph:
    """Compile and return the agent's LangGraph :class:`StateGraph`.

    Args:
        llm: The base chat model. It will be bound to ``tools`` inside this
             function, so callers should pass the *unbound* model.
        tools: List of LangChain tools the agent may invoke.
        guardrails: List of guardrails to apply.
        checkpointer: Optional checkpointer for short-term memory.

    Returns:
        A compiled LangGraph application ready to ``.invoke()``.
    """
    llm_with_tools = llm.bind_tools(tools)
    agent_node = make_agent_node(llm_with_tools)
    tool_node = ToolNode(tools)
    guardrail_node = make_guardrail_node(guardrails)

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node(NODE_AGENT, agent_node)
    graph.add_node(NODE_TOOLS, tool_node)
    graph.add_node(NODE_GUARDRAILS, guardrail_node)

    # Wire edges
    graph.add_edge(START, NODE_AGENT)
    graph.add_conditional_edges(
        NODE_AGENT,
        should_continue,
        {
            NODE_GUARDRAILS: NODE_GUARDRAILS,
            "end": END,
        },
    )
    graph.add_conditional_edges(
        NODE_GUARDRAILS,
        should_continue_from_guardrails,
        {
            "tools": NODE_TOOLS,
            "agent": NODE_AGENT,
            "end": END,
        }
    )
    graph.add_edge(NODE_TOOLS, NODE_AGENT)  # loop back after tool execution

    compiled = graph.compile(checkpointer=checkpointer)
    logger.debug("Graph compiled with %d tool(s) and %d guardrail(s).", len(tools), len(guardrails))
    return compiled
