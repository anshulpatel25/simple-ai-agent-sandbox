"""LangGraph graph construction.

The graph implements the standard ReAct pattern:

    START → agent → (tools → agent)* → END

``build_graph`` is a pure factory – it accepts the LLM and tools as arguments
so it can be unit-tested independently of Docker and LM Studio.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.graph.nodes import make_agent_node, should_continue
from agent.graph.state import AgentState

logger = logging.getLogger(__name__)

# Node name constants – avoids magic strings scattered throughout.
NODE_AGENT = "agent"
NODE_TOOLS = "tools"


def build_graph(llm: BaseChatModel, tools: list[BaseTool]) -> StateGraph:
    """Compile and return the agent's LangGraph :class:`StateGraph`.

    Args:
        llm: The base chat model. It will be bound to ``tools`` inside this
             function, so callers should pass the *unbound* model.
        tools: List of LangChain tools the agent may invoke.

    Returns:
        A compiled LangGraph application ready to ``.invoke()``.
    """
    llm_with_tools = llm.bind_tools(tools)
    agent_node = make_agent_node(llm_with_tools)
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node(NODE_AGENT, agent_node)
    graph.add_node(NODE_TOOLS, tool_node)

    # Wire edges
    graph.add_edge(START, NODE_AGENT)
    graph.add_conditional_edges(
        NODE_AGENT,
        should_continue,
        {
            "tools": NODE_TOOLS,
            "end": END,
        },
    )
    graph.add_edge(NODE_TOOLS, NODE_AGENT)  # loop back after tool execution

    compiled = graph.compile()
    logger.debug("Graph compiled with %d tool(s).", len(tools))
    return compiled
