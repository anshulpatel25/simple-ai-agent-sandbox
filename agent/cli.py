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

from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
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
from agent.guardrails.base import GuardrailRegistry
from agent.guardrails.deletion import FileDeletionGuardrail

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
                ("\nRuntime:   ", "dim"),
                (settings.container_runtime.value, "green"),
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


def _run_repl(agent_graph, thread_id: str) -> None:
    """Interactive REPL loop.

    Args:
        agent_graph: Compiled LangGraph application.
        thread_id: Unique identifier for the conversation thread.
    """
    config = {"configurable": {"thread_id": thread_id}}

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

        graph_input = {"messages": [HumanMessage(content=user_input)]}

        while True:
            with console.status("[bold cyan]Thinking…[/bold cyan]", spinner="dots"):
                try:
                    # We only need to pass the new message; LangGraph + MemorySaver
                    # handles the history.
                    result = agent_graph.invoke(
                        graph_input,
                        config=config,
                    )
                except Exception as exc:
                    console.print(f"[bold red]Agent error:[/bold red] {exc}")
                    break

            # Check for interrupts
            state = agent_graph.get_state(config)
            if state.tasks and any(task.interrupts for task in state.tasks):
                # Handle the first interrupt for simplicity
                for task in state.tasks:
                    if task.interrupts:
                        interrupt_value = task.interrupts[0].value
                        # Show confirmation prompt to user
                        console.print(f"[bold yellow]Guardrail Triggered:[/bold yellow] {interrupt_value}")
                        confirmation = Prompt.ask("[bold cyan]Proceed? (yes/no)[/bold cyan]", default="no")
                        # Resume with Command
                        graph_input = Command(resume=confirmation)
                        break
                continue # Re-invoke with Command
            else:
                # No more interrupts, show the response and break inner loop
                conversation = result["messages"]
                ai_messages = [m for m in conversation if isinstance(m, AIMessage)]
                if ai_messages:
                    last_reply = ai_messages[-1].content
                    if last_reply:
                        _print_response(str(last_reply))
                break


def main() -> None:
    """Application entry point – orchestrates container, skills, graph, REPL."""
    with DockerContainerManager(
        image=settings.docker_image,
        runtime=settings.container_runtime.value,
    ) as container_manager:
        _print_welcome(container_manager.container_id or "unknown")

        # --- Build the skill registry
        registry = SkillRegistry()
        registry.register(BashSkill(container_manager))
        tools = registry.get_tools()

        # --- Build the guardrail registry
        guardrail_registry = GuardrailRegistry()
        guardrail_registry.register(FileDeletionGuardrail())
        guardrails = guardrail_registry.get_guardrails()

        # --- Build the LLM + graph
        llm = _build_llm()
        memory = MemorySaver()
        agent_graph = build_graph(llm, tools, guardrails, checkpointer=memory)

        # --- Seed the system prompt and generate session ID
        thread_id = str(uuid4())
        agent_graph.update_state(
            {"configurable": {"thread_id": thread_id}},
            {"messages": [SystemMessage(content=_SYSTEM_PROMPT)]},
        )

        # --- Start the interactive loop
        _run_repl(agent_graph, thread_id)

    console.print("[dim]Container cleaned up. Bye![/dim]")


if __name__ == "__main__":
    main()
