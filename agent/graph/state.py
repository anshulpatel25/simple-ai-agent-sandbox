"""LangGraph agent state definition."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """The only state the agent graph carries between nodes.

    ``messages`` uses the :func:`~langgraph.graph.message.add_messages`
    reducer so appending a new message never loses history.
    """

    messages: Annotated[list[AnyMessage], add_messages]
