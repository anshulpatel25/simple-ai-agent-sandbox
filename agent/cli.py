"""CLI entry point for the containerized AI agent.

Each invocation of this module:
1. Reads configuration from ``config.py`` / ``.env``.
2. Spins up a fresh Docker container.
3. Loads the Bash skill (which reads ``skills/bash/skill.md``).
4. Builds the LangGraph ReAct agent.
5. Runs an interactive REPL loop.
6. Destroys the container on exit – no matter what.
"""

from __future__ import annotations

import logging
import sys

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from agent.config import settings
from agent.container.manager import DockerContainerManager
from agent.graph.graph import build_graph
from agent.skills.base import SkillRegistry
from agent.skills.bash_skill import BashSkill

# ------------------------------------------------------------------ logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

console = Console()

# System prompt injected at the start of every session.
_SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to a sandboxed Ubuntu shell. "
    "Use the run_bash_command tool whenever the user asks you to perform any "
    "system or file operation. "
    "Always show the raw command output to the user. "
    "Be concise and accurate."
)


def _build_llm() -> ChatOpenAI:
    """Construct the LM Studio-backed LLM from settings."""
    return ChatOpenAI(
        base_url=settings.lm_studio_base_url,
        api_key=settings.lm_studio_api_key,  # type: ignore[arg-type]
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )


def _print_welcome(container_id: str) -> None:
    console.print(
        Panel.fit(
            Text.assemble(
                ("🤖  Simple AI Agent\n", "bold cyan"),
                ("Container: ", "dim"),
                (container_id, "green"),
                ("\nModel:     ", "dim"),
                (settings.llm_model, "green"),
                ("\nImage:     ", "dim"),
                (settings.docker_image, "green"),
                ("\n\nType ", "dim"),
                ("exit", "bold yellow"),
                (" or press ", "dim"),
                ("Ctrl-C", "bold yellow"),
                (" to quit.", "dim"),
            ),
            title="[bold]Session Started[/bold]",
            border_style="cyan",
        )
    )


def _print_response(content: str) -> None:
    """Render the assistant's reply as Markdown."""
    console.print(Rule(style="dim"))
    console.print(Markdown(content))
    console.print()


def _run_repl(agent_graph, initial_messages: list) -> None:
    """Interactive REPL loop.

    Args:
        agent_graph: Compiled LangGraph application.
        initial_messages: Pre-seeded messages (e.g. the system prompt).
    """
    conversation: list = list(initial_messages)

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Shutting down…[/dim]")
            break

        if user_input.strip().lower() in {"exit", "quit", "bye"}:
            console.print("[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        conversation.append(HumanMessage(content=user_input))

        with console.status("[bold cyan]Thinking…[/bold cyan]", spinner="dots"):
            try:
                result = agent_graph.invoke({"messages": conversation})
            except Exception as exc:
                console.print(f"[bold red]Agent error:[/bold red] {exc}")
                continue

        # Update conversation with the full returned history
        conversation = result["messages"]

        # Find and display the last AI message
        ai_messages = [
            m for m in conversation if hasattr(m, "content") and m.type == "ai"
        ]
        if ai_messages:
            last_reply = ai_messages[-1].content
            _print_response(str(last_reply))


def main() -> None:
    """Application entry point – orchestrates container, skills, graph, REPL."""
    with DockerContainerManager(image=settings.docker_image) as container_manager:
        _print_welcome(container_manager.container_id or "unknown")

        # --- Build the skill registry
        registry = SkillRegistry()
        registry.register(BashSkill(container_manager))
        tools = registry.get_tools()

        # --- Build the LLM + graph
        llm = _build_llm()
        agent_graph = build_graph(llm, tools)

        # --- Seed the system prompt as the first message
        from langchain_core.messages import SystemMessage
        initial_messages = [SystemMessage(content=_SYSTEM_PROMPT)]

        # --- Start the interactive loop
        _run_repl(agent_graph, initial_messages)

    console.print("[dim]Container cleaned up. Bye![/dim]")


if __name__ == "__main__":
    main()
