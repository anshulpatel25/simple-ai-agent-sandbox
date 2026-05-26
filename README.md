# Simple AI Agent Sandbox

A CLI-based AI agent where **every session runs inside its own ephemeral Ubuntu Docker container**. The agent uses [LangGraph](https://github.com/langchain-ai/langgraph) for the agentic ReAct loop and connects to [LM Studio](https://lmstudio.ai/) as the local LLM backend.

> **Package manager:** [`uv`](https://docs.astral.sh/uv/) — fast Python package and project manager.

---

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                   Host Machine                        │
│                                                       │
│  ┌──────────────┐   docker run   ┌───────────────┐    │
│  │  cli.py      │ ─────────────► │ ubuntu:latest │    │
│  │  (REPL)      │                │ (per session) │    │
│  │              │ ◄─ stdout/err ─└───────────────┘    │
│  └──────┬───────┘                                     │
│         │                                             │
│         ▼                                             │
│  ┌──────────────┐                                     │
│  │  LangGraph   │  agent → tools → agent loop         │
│  │  ReAct Graph │                                     │
│  └──────┬───────┘                                     │
│         │                                             │
│         ▼                                             │
│  ┌──────────────┐                                     │
│  │  LM Studio   │  OpenAI-compatible local API        │
│  │  (localhost) │  http://localhost:1234/v1           │
│  └──────────────┘                                     │
└───────────────────────────────────────────────────────┘
```

## Project Structure

```
simple-ai-agent-sandbox/
├── main.py                    # Entry point
├── pyproject.toml
├── .env.example               # Config template
│
├── skills/
│   └── bash/
│       └── skill.md           # Bash command reference (agent reads this)
│
└── agent/
    ├── cli.py                 # REPL + wiring
    ├── config.py              # Pydantic Settings
    │
    ├── container/
    │   └── manager.py         # DockerContainerManager
    │
    ├── skills/
    │   ├── base.py            # Skill ABC + SkillRegistry
    │   └── bash_skill.py      # BashSkill (reads skill.md)
    │
    └── graph/
        ├── state.py           # AgentState
        ├── nodes.py           # agent_node + should_continue
        └── graph.py           # build_graph()
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | |
| uv | `curl -Lsf https://astral.sh/uv/install.sh` |
| Docker | Must be running; user must have Docker socket access |
| LM Studio | Running with a model loaded and the local server **enabled** |

---

## Setup

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd simple-ai-agent-sandbox
```

### 2. Install dependencies

```bash
uv sync
```

`uv sync` reads `pyproject.toml`, creates a virtual environment in `.venv/`, and installs all dependencies in one step.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set `LLM_MODEL` to the exact model identifier shown in LM Studio:

```env
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio
LLM_MODEL=lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF   # example
DOCKER_IMAGE=ubuntu:latest
```

### 4. Start LM Studio

- Open LM Studio → load a model → click **"Start Server"**
- Default address: `http://localhost:1234`

### 5. Run the agent

```bash
# Preferred – uv runs inside the managed venv automatically
uv run python main.py

# Or use the installed script entry-point
uv run agent
```

---

## Example Session

```
╭─ Session Started ──────────────────────────────────╮
│ 🤖  Simple AI Agent                                │
│ Container: a3f2b1c                                 │
│ Model:     lmstudio-community/Meta-Llama-3-8B      │
│ Image:     ubuntu:latest                           │
│                                                    │
│ Type exit or press Ctrl-C to quit.                 │
╰────────────────────────────────────────────────────╯

You: What OS is running inside the container?
─────────────────────────────────────────────────────
The container is running Ubuntu. Here's the output of `uname -a`:

Linux a3f2b1c 5.15.0 #1 SMP ... x86_64 x86_64 x86_64 GNU/Linux

You: Create a file called hello.txt with "Hello World" in it
─────────────────────────────────────────────────────
Done! I ran:
    echo "Hello World" > hello.txt
Let me verify: `cat hello.txt` → **Hello World**

You: exit
Goodbye!
Container a3f2b1c removed.
```

---

## Adding a New Skill

1. Create `skills/<your-skill>/skill.md` documenting what the skill can do.
2. Create `agent/skills/<your_skill>.py`:

```python
from agent.skills.base import Skill
from langchain_core.tools import BaseTool, tool

class MySkill(Skill):
    @property
    def name(self) -> str:
        return "my_skill"

    def as_langchain_tool(self) -> BaseTool:
        @tool
        def my_tool(input: str) -> str:
            """Tool description for the LLM."""
            ...
        return my_tool
```

3. Register it in `agent/cli.py`:

```python
registry.register(MySkill())
```

That's it – no other files need to change.

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio API base URL |
| `LM_STUDIO_API_KEY` | `lm-studio` | Dummy key (LM Studio ignores it) |
| `LLM_MODEL` | `local-model` | Model identifier as shown in LM Studio |
| `LLM_TEMPERATURE` | `0.0` | LLM sampling temperature |
| `DOCKER_IMAGE` | `ubuntu:latest` | Docker image for agent containers |
| `CONTAINER_TIMEOUT` | `0` | Idle timeout in seconds (0 = disabled) |
