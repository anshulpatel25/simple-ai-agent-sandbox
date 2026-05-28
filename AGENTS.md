# AGENTS.md

> Guidance for AI coding agents working on **simple-ai-agent-sandbox**.

---

## Project Overview

`simple-ai-agent-sandbox` is a CLI-based AI agent where **every session runs inside its own ephemeral Ubuntu Docker container**. The agent uses [LangGraph](https://github.com/langchain-ai/langgraph) to implement a ReAct loop with interactive guardrails and persistent short-term memory. It connects to [LM Studio](https://lmstudio.ai/) as the local LLM backend.

### Key Design Principles

1. **Ephemeral sandboxing** – each CLI session gets a fresh Docker container; the container is always destroyed on exit.
2. **Skill-based extensibility** – capabilities are added through a `Skill` ABC + `SkillRegistry`.
3. **Interactive Guardrails** – sensitive tool calls (like file deletion) trigger a human-in-the-loop confirmation using LangGraph's `interrupt`.
4. **Conversational Memory** – uses LangGraph `MemorySaver` to persist state across turns within a session.
5. **Pure factory functions** – `build_graph()` accepts dependencies (llm, tools, guardrails, checkpointer) as arguments for better testability.

---

## Repository Layout

```
simple-ai-agent-sandbox/
├── main.py                        # Entry point — calls agent.cli:main
├── pyproject.toml                 # Project metadata & dependencies (uv)
├── .env.example                   # Environment configuration template
│
├── skills/                        # Human-readable skill reference docs
│   └── bash/
│       └── skill.md               # Bash command reference; read by BashSkill
│
└── agent/                         # All application code
    ├── cli.py                     # REPL loop, wires everything together
    ├── config.py                  # Pydantic Settings (reads .env)
    ├── utils.py                   # PyInstaller-aware path resolution
    │
    ├── container/
    │   └── manager.py             # DockerContainerManager
    │
    ├── guardrails/
    │   ├── base.py                # Guardrail ABC + GuardrailRegistry
    │   └── deletion.py            # FileDeletionGuardrail implementation
    │
    ├── skills/
    │   ├── base.py                # Skill ABC + SkillRegistry
    │   └── bash_skill.py          # BashSkill
    │
    └── graph/
        ├── state.py               # AgentState TypedDict
        ├── nodes.py               # agent_node, guardrail_node, routers
        └── graph.py               # build_graph() — compiles the StateGraph
```

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| Package manager | [`uv`](https://docs.astral.sh/uv/) |
| Agentic loop | [LangGraph](https://github.com/langchain-ai/langgraph) ≥ 0.2 |
| LLM client | [LangChain OpenAI](https://github.com/langchain-ai/langchain) ≥ 0.3 |
| LLM backend | [LM Studio](https://lmstudio.ai/) (OpenAI-compatible) |
| Container runtime | [Docker SDK for Python](https://docker-py.readthedocs.io/) |
| CLI / TUI | [Rich](https://rich.readthedocs.io/) |

---

## Architecture: Data Flow

```
User Input (REPL)
      │
      ▼
agent/cli.py  ─── HumanMessage ──►  agent/graph/graph.py
                                          │
                                   StateGraph (ReAct)
                                    START
                                      │
                                 agent_node  ◄──────────────────────────┐
                                      │                                 │
                            ┌─────────┴──────────┐                      │
                      tool_calls?             no tool calls             │
                            │                      │                    │
                            ▼                    END                    │
                      guardrail_node                                    │
                            │                                           │
                ┌───────────┴───────────┐                               │
            triggered?               not triggered                      │
                │                       │                               │
                ▼                       ▼                               │
          interrupt()               ToolNode (tools) ───────────────────┘
          (Human Confirmation)          │
                │                       ▼
          ┌─────┴─────┐           DockerContainerManager.exec()
       approved    rejected             │
          │           │                 ▼
          ▼           └───────────► ToolMessage (cancelled) ────────────┘
       ToolNode
```

---

## Adding a New Skill

1. **Write reference doc**: Create `skills/<your-skill>/skill.md`.
2. **Implement `Skill`**: Create `agent/skills/<your_skill>.py`.
3. **Register**: Add `registry.register(MySkill())` in `agent/cli.py`.

## Adding a New Guardrail

1. **Implement `Guardrail`**: Create `agent/guardrails/<your_guardrail>.py` inheriting from `Guardrail`.
2. **Implement `check`**: Return a string prompt if the `tool_call` should be interrupted for confirmation.
3. **Register**: Add `guardrail_registry.register(MyGuardrail())` in `agent/cli.py`.

---

## Key Modules — What to Know

### `agent/graph/graph.py` — `build_graph()`
- Wires the `agent` node, `guardrails` node, and `tools` node.
- Uses `MemorySaver` for short-term memory persistence.

### `agent/graph/nodes.py` — `guardrail_node`
- Inspects `tool_calls` against registered guardrails.
- Uses `langgraph.types.interrupt` to pause execution for human input.

### `agent/cli.py` — REPL
- Handles graph interrupts by presenting the confirmation prompt to the user.
- Resumes execution using `langgraph.types.Command(resume=...)`.
- Tracks and displays token usage per AI message.

---

## Code Conventions

- **Python 3.12+** syntax.
- **Type hints** on all function signatures.
- **Rich** for all CLI output (panels, rules, colored text).
- **Factory patterns** for graph and node creation to ensure testability.
- **Resource paths**: Always use `agent.utils.get_resource_path` for bundled files (to support PyInstaller).

---

## Common Pitfalls

| Pitfall | Fix |
|---|---|
| Interrupt not handled | Ensure the REPL in `cli.py` checks `state.tasks` for interrupts and uses `Command(resume=...)`. |
| Skill files not found | Use `get_resource_path` to resolve paths like `skills/bash/skill.md`. |
| Token usage missing | Model must support `usage_metadata` or `token_usage` in `response_metadata`. |
| Container leaking | Use `DockerContainerManager` as a context manager. |
