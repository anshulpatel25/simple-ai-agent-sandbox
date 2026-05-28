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


def _extract_token_usage(message: AIMessage) -> tuple[int, int]:
    """Extract input and output tokens from an AIMessage.

    Args:
        message: The AI message to extract usage from.

    Returns:
        A tuple of (input_tokens, output_tokens).
    """
    input_tokens = 0
    output_tokens = 0

    # LangChain 0.3+ usage_metadata
    usage_metadata = getattr(message, "usage_metadata", None)
    if usage_metadata:
        input_tokens = usage_metadata.get("input_tokens", 0)
        output_tokens = usage_metadata.get("output_tokens", 0)
    # Fallback to response_metadata (often used by older providers or specific integrations)
    elif "token_usage" in message.response_metadata:
        usage = message.response_metadata["token_usage"]
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

    return input_tokens, output_tokens


def _print_token_usage(input_tokens: int, output_tokens: int) -> None:
    """Render token usage using Rich."""
    if input_tokens == 0 and output_tokens == 0:
        return

    total_tokens = input_tokens + output_tokens
    usage_text = Text.assemble(
        ("Tokens: ", "dim"),
        (str(input_tokens), "cyan"),
        (" input, ", "dim"),
        (str(output_tokens), "cyan"),
        (" output ", "dim"),
        (f"({total_tokens} total)", "dim italic"),
    )
    console.print(usage_text, justify="right")


def _run_repl(agent_graph, thread_id: str) -> None:
    """Interactive REPL loop.

    Args:
        agent_graph: Compiled LangGraph application.
        thread_id: Unique identifier for the conversation thread.
    """
    if settings.deepeval_tracing:
        console.print("[dim]DeepEval tracing enabled.[/dim]")

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

        config = {"configurable": {"thread_id": thread_id}}
        if settings.deepeval_tracing:
            try:
                from deepeval.integrations.langchain.callback import CallbackHandler
                config["callbacks"] = [CallbackHandler()]
            except ImportError:
                pass

        graph_input = {"messages": [HumanMessage(content=user_input)]}
        total_input_tokens = 0
        total_output_tokens = 0

        # Initialize processed IDs with existing messages to only count new ones in this turn
        state = agent_graph.get_state(config)
        processed_message_ids: set[str] = {
            msg.id for msg in state.values.get("messages", []) if msg.id
        }

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

            # Accumulate tokens from all AI messages in this turn
            for msg in result["messages"]:
                if isinstance(msg, AIMessage) and msg.id not in processed_message_ids:
                    in_t, out_t = _extract_token_usage(msg)
                    total_input_tokens += in_t
                    total_output_tokens += out_t
                    if msg.id:
                        processed_message_ids.add(msg.id)

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
                _print_token_usage(total_input_tokens, total_output_tokens)
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
