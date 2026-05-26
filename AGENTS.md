# AGENTS.md

> Guidance for AI coding agents (e.g. Copilot, Cursor, Antigravity) working on **simple-ai-agent-sandbox**.

---

## Project Overview

`simple-ai-agent-sandbox` is a CLI-based AI agent where **every session runs inside its own ephemeral Ubuntu Docker container**. The agent uses [LangGraph](https://github.com/langchain-ai/langgraph) to implement a ReAct (Reason + Act) loop and connects to [LM Studio](https://lmstudio.ai/) as the local LLM backend via an OpenAI-compatible API.

### Key Design Principles

1. **Ephemeral sandboxing** – each CLI session gets a fresh Docker container; the container is always destroyed on exit (even on crash or Ctrl-C).
2. **Skill-based extensibility** – the agent's capabilities are added through a `Skill` ABC + `SkillRegistry`. No existing files need changing when adding a new skill.
3. **Pure factory functions** – `build_graph()` and `make_agent_node()` accept dependencies as arguments, keeping them unit-testable without Docker or LM Studio.
4. **Single settings singleton** – `agent/config.py` provides one `settings` object read from `.env`; import it everywhere instead of constructing `Settings()` again.

---

## Repository Layout

```
simple-ai-agent-sandbox/
├── main.py                        # Entry point — calls agent.cli:main
├── pyproject.toml                 # Project metadata & dependencies (uv / hatchling)
├── .envrc / .envrc.example        # direnv env-var activation (uses uv venv)
├── .python-version                # Pinned Python version (3.12)
│
├── skills/                        # Human-readable skill reference docs
│   └── bash/
│       └── skill.md               # Bash command reference; read by BashSkill at runtime
│
└── agent/                         # All application code lives here
    ├── __init__.py
    ├── cli.py                     # REPL loop, wires together container + skills + graph
    ├── config.py                  # Pydantic Settings (reads .env); exposes `settings`
    │
    ├── container/
    │   └── manager.py             # DockerContainerManager — create / exec / destroy
    │
    ├── skills/
    │   ├── base.py                # Skill ABC + SkillRegistry
    │   └── bash_skill.py          # BashSkill — runs shell commands in the container
    │
    └── graph/
        ├── state.py               # AgentState TypedDict
        ├── nodes.py               # make_agent_node(), should_continue()
        └── graph.py               # build_graph() — compiles the LangGraph StateGraph
```

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| Package manager | [`uv`](https://docs.astral.sh/uv/) |
| Agentic loop | [LangGraph](https://github.com/langchain-ai/langgraph) ≥ 0.2 |
| LLM client | [LangChain OpenAI](https://github.com/langchain-ai/langchain) ≥ 0.2 |
| LLM backend | [LM Studio](https://lmstudio.ai/) (OpenAI-compatible, `http://localhost:1234/v1`) |
| Container runtime | [Docker SDK for Python](https://docker-py.readthedocs.io/) ≥ 7.0 |
| Settings | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) ≥ 2.0 |
| CLI / TUI | [Rich](https://rich.readthedocs.io/) ≥ 13.0 |

---

## Development Setup

```bash
# 1. Install uv (if not already installed)
curl -Lsf https://astral.sh/uv/install.sh | sh

# 2. Install all dependencies into the managed .venv
uv sync

# 3. Copy and edit the environment config
cp .envrc.example .envrc
# Set LLM_MODEL to the exact model identifier shown in LM Studio

# 4. Start LM Studio → load a model → enable "Start Server"

# 5. Run the agent
uv run python main.py
# or via the script entry-point:
uv run agent
```

### Runtime Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | Pinned in `.python-version` |
| Docker daemon | Must be running; user must have Docker socket access |
| LM Studio | Running with a model loaded and local server enabled |

---

## Architecture: Data Flow

```
User Input (REPL)
      │
      ▼
agent/cli.py  ─── SystemMessage + HumanMessage ──►  agent/graph/graph.py
                                                            │
                                                     StateGraph (ReAct)
                                                      START
                                                        │
                                                   agent_node  ◄──────────────┐
                                                        │                      │
                                              ┌─────────┴──────────┐          │
                                        tool_calls?             no tool calls  │
                                              │                      │         │
                                              ▼                    END         │
                                         ToolNode                              │
                                        (run_bash_command)                     │
                                              │                                │
                                    DockerContainerManager.exec()              │
                                              │                                │
                                         ToolMessage ─────────────────────────┘
```

---

## Adding a New Skill

Adding a skill requires **only two new files** and one line of registration:

### 1. Write a skill reference doc

Create `skills/<your-skill>/skill.md` describing what the skill does. The agent reads this at runtime to understand its capabilities.

### 2. Implement the `Skill` subclass

Create `agent/skills/<your_skill>.py`:

```python
from pathlib import Path
from langchain_core.tools import BaseTool, tool
from agent.skills.base import Skill

class MySkill(Skill):
    @property
    def name(self) -> str:
        return "my_skill"  # must be unique across registered skills

    def as_langchain_tool(self) -> BaseTool:
        @tool
        def my_tool(input: str) -> str:
            """Describe what this tool does (the LLM reads this)."""
            ...
        return my_tool
```

### 3. Register in `agent/cli.py`

```python
registry.register(MySkill())
```

No other files need to change. The `SkillRegistry` raises `ValueError` on duplicate names, so keep skill names unique.

---

## Key Modules — What to Know

### `agent/config.py` — `Settings`

- Single `settings` singleton sourced from `.env` (via `pydantic-settings`).
- **Always import `settings`** from this module; do not instantiate `Settings()` elsewhere to avoid repeated disk reads.
- All fields have sensible defaults so the agent runs out-of-the-box.

### `agent/container/manager.py` — `DockerContainerManager`

- Manages one container per session.
- **Always use as a context manager** (`with DockerContainerManager(...) as mgr`) — `__exit__` calls `destroy()` unconditionally.
- `exec(command: str) -> ExecResult` runs any bash command. Check `result.succeeded` (exit code 0) before trusting `result.stdout`.

### `agent/skills/base.py` — `Skill` + `SkillRegistry`

- `Skill` is an abstract base class; subclasses must implement `name` (str property) and `as_langchain_tool()`.
- `SkillRegistry.register()` raises `ValueError` on duplicate skill names.
- `SkillRegistry.get_tools()` returns a flat `list[BaseTool]` passed directly to the LangGraph `ToolNode`.

### `agent/graph/graph.py` — `build_graph()`

- Pure factory: accepts `llm` and `tools` as arguments — no global state.
- Implements the standard ReAct pattern: `START → agent → (tools → agent)* → END`.
- The conditional edge is decided by `should_continue()` in `nodes.py`.

### `agent/graph/nodes.py` — `make_agent_node()` / `should_continue()`

- `make_agent_node(llm_with_tools)` is a factory to avoid global LLM state in the node closure.
- `should_continue` inspects the last `AIMessage`; routes to `"tools"` if `tool_calls` is non-empty, otherwise `"end"`.

### `agent/cli.py` — `main()`

- Entry point that wires everything together.
- `_SYSTEM_PROMPT` is injected as the first `SystemMessage` in every session.
- The REPL appends `HumanMessage` objects to the conversation list and updates it from the full returned state after each invocation.

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio OpenAI-compatible API base URL |
| `LM_STUDIO_API_KEY` | `lm-studio` | Dummy API key (LM Studio ignores it) |
| `LLM_MODEL` | `local-model` | Model identifier as shown in LM Studio |
| `LLM_TEMPERATURE` | `0.0` | LLM sampling temperature |
| `DOCKER_IMAGE` | `ubuntu:latest` | Docker image used for agent session containers |
| `CONTAINER_TIMEOUT` | `0` | Idle timeout in seconds (0 = disabled) |
| `SKILLS_BASE_DIR` | `skills` | Root directory for skill markdown definitions |

---

## Code Conventions

- **Python 3.12+** with `from __future__ import annotations` at the top of every module.
- **Type hints everywhere** — use `Optional[X]` (not `X | None`) for compatibility, and add `# type: ignore` sparingly and with a comment explaining why.
- **Logging over print** — use `logging.getLogger(__name__)` in every module. The root level is `WARNING`; use `logger.debug` / `logger.info` for operational messages.
- **Docstrings** follow Google style (Args / Returns / Raises sections).
- **Dataclasses** for plain data containers (e.g. `ExecResult`).
- **No circular imports** — dependency direction: `cli → graph, skills, container, config`; `graph → skills`; `skills → container`.

---

## Testing Guidance

There are no automated tests yet. When writing tests:

- **Mock `DockerContainerManager`** — never spin up a real container in unit tests. Inject a mock or subclass.
- **Mock `ChatOpenAI`** — use `MagicMock` or `langchain_core.language_models.fake.FakeListChatModel` to avoid LM Studio dependency.
- `build_graph()` and `make_agent_node()` are pure factories and are straightforward to test in isolation.
- `BashSkill` accepts a `skill_md_path` override in its constructor — use a temp file in tests.
- Run tests with: `uv run pytest`

---

## Common Pitfalls

| Pitfall | Fix |
|---|---|
| `FileNotFoundError` on `skills/bash/skill.md` | Run the agent from the **project root**; `BashSkill` resolves the path relative to CWD |
| Docker socket permission denied | Add your user to the `docker` group: `sudo usermod -aG docker $USER` (re-login required) |
| LM Studio model identifier mismatch | Copy the exact model string from LM Studio UI into `LLM_MODEL` in `.env` |
| Duplicate skill name `ValueError` | Each skill registered in `cli.py` must have a unique `name` property |
| Container not cleaned up | Always use `DockerContainerManager` as a context manager — never call `create()` without a matching `destroy()` |
