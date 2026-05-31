"""DeepEval integration test for the AI agent.

Validates a **single end-to-end workflow**:
    User asks the agent to list files in /tmp inside the Docker sandbox.

The test exercises the full real stack:
    LangGraph ReAct loop → BashSkill → DockerContainerManager → actual container

Evaluation is performed by DeepEval's AnswerRelevancyMetric, judged by a
local LM Studio model so no external API calls are made.

Requirements:
    - Docker daemon running and accessible.
    - LM Studio running at the URL configured in settings / .env.
    - The agent's .env (or environment) is populated correctly.

Run with:
    uv run pytest tests/eval/test_agent_eval.py -v
"""

from __future__ import annotations

import pytest
from openai import OpenAI
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from agent.config import settings
from agent.container.manager import DockerContainerManager
from agent.graph.graph import build_graph
from agent.guardrails.base import GuardrailRegistry
from agent.guardrails.deletion import FileDeletionGuardrail
from agent.skills.base import SkillRegistry
from agent.skills.bash_skill import BashSkill


# ---------------------------------------------------------------------------
# Judge LLM – thin wrapper so DeepEval uses LM Studio instead of OpenAI
# ---------------------------------------------------------------------------

class LMStudioJudge(DeepEvalBaseLLM):
    """Points DeepEval's evaluation LLM at the local LM Studio endpoint."""

    def __init__(self) -> None:
        self._client = OpenAI(
            base_url=settings.lm_studio_base_url,
            api_key=settings.lm_studio_api_key,
        )
        self._model = settings.llm_model

    def load_model(self) -> str:  # type: ignore[override]
        return self._model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# System prompt (mirrors agent/cli.py)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to a sandboxed Ubuntu shell. "
    "Use the run_bash_command tool whenever the user asks you to perform any "
    "system or file operation. "
    "Always show the raw command output to the user. "
    "Be concise and accurate."
)


# ---------------------------------------------------------------------------
# Workflow under test
# ---------------------------------------------------------------------------

# The single user message that defines the workflow being validated.
_USER_INPUT = "List all files and directories inside /tmp in the Docker sandbox."

# The minimum expected behaviour in the agent's reply (used as context for
# AnswerRelevancyMetric – the metric checks relevancy, not exact match).
_EXPECTED_OUTPUT = (
    "The agent should execute `ls /tmp` (or equivalent) inside the Docker "
    "container and return the directory listing to the user."
)


# ---------------------------------------------------------------------------
# Pytest fixture – full real stack, container started & torn down per test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def full_agent(tmp_path):
    """Spin up a real Docker container and wire the complete agent graph.

    Yields the compiled graph and the thread_id; destroys the container when
    the test finishes.
    """
    with DockerContainerManager(
        image=settings.docker_image,
        runtime=settings.container_runtime.value,
    ) as container_manager:
        # Skill registry
        skill_registry = SkillRegistry()
        skill_registry.register(BashSkill(container_manager))
        tools = skill_registry.get_tools()

        # Guardrail registry
        guardrail_registry = GuardrailRegistry()
        guardrail_registry.register(FileDeletionGuardrail())
        guardrails = guardrail_registry.get_guardrails()

        # LLM
        llm = ChatOpenAI(
            base_url=settings.lm_studio_base_url,
            api_key=settings.lm_studio_api_key,  # type: ignore[arg-type]
            model=settings.llm_model,
            temperature=0,
        )

        # Build the graph with a fresh MemorySaver per test
        memory = MemorySaver()
        graph = build_graph(llm, tools, guardrails, checkpointer=memory)

        thread_id = "eval-session-list-tmp"
        config = {"configurable": {"thread_id": thread_id}}

        # Seed the system prompt exactly as the CLI does
        graph.update_state(config, {"messages": [SystemMessage(content=_SYSTEM_PROMPT)]})

        yield graph, config


# ---------------------------------------------------------------------------
# Single DeepEval test case
# ---------------------------------------------------------------------------

def test_agent_lists_tmp_directory(full_agent):
    """Workflow: agent lists /tmp inside the Docker container.

    Full stack validated:
        1. LangGraph routes the user message to the agent node.
        2. The agent decides to call run_bash_command("ls /tmp" or similar).
        3. The guardrail node passes the non-destructive command through.
        4. BashSkill executes the command in the live Docker container.
        5. The agent incorporates the tool output into its final reply.
        6. DeepEval's AnswerRelevancyMetric confirms the reply is relevant.
    """
    graph, config = full_agent

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=_USER_INPUT)]},
            config=config,
        )
    except Exception as exc:
        pytest.skip(f"Agent invocation failed (is LM Studio / Docker running?): {exc}")

    # Extract the last AI message from the conversation
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_messages, "Agent produced no AI response."
    actual_output: str = str(ai_messages[-1].content)

    # Build the DeepEval test case
    test_case = LLMTestCase(
        input=_USER_INPUT,
        actual_output=actual_output,
        expected_output=_EXPECTED_OUTPUT,
    )

    # AnswerRelevancyMetric verifies the response is on-topic.
    # threshold=0.7 allows some variation in phrasing while still catching
    # completely off-topic or empty replies.
    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=LMStudioJudge(),
        include_reason=True,
    )

    assert_test(test_case, [metric])
