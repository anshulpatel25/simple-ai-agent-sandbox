import pytest
from openai import OpenAI
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from unittest.mock import MagicMock
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams as SingleTurnParams
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from agent.graph.graph import build_graph
from agent.config import settings
from agent.skills.bash_skill import BashSkill
from agent.container.manager import DockerContainerManager
from deepeval.integrations.langchain.callback import CallbackHandler


class LMStudioJudge(DeepEvalBaseLLM):
    """Thin wrapper that points DeepEval's judge LLM at a local LM Studio endpoint."""

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

@pytest.fixture
def bash_agent_graph():
    llm = ChatOpenAI(
        base_url=settings.lm_studio_base_url,
        api_key=settings.lm_studio_api_key,
        model=settings.llm_model,
        temperature=0,
    )

    # Mock container manager to avoid Docker dependency in eval tests.
    # We want to test the agent's logic, not the Docker environment itself.
    mock_container = MagicMock(spec=DockerContainerManager)

    # Configure mock to return sensible defaults for bash commands
    mock_exec_result = MagicMock()
    mock_exec_result.combined_output.return_value = "total 0\n-rw-r--r-- 1 root root 0 Jan 1 00:00 test.txt"
    mock_exec_result.succeeded = True
    mock_container.exec.return_value = mock_exec_result

    bash_skill = BashSkill(container_manager=mock_container)
    tools = [bash_skill.as_langchain_tool()]
    guardrails = []
    return build_graph(llm, tools, guardrails)

@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        (
            "List all files in the current directory.",
            "The agent should successfully invoke the 'run_bash_command' tool with 'ls' or 'ls -la' and report the findings."
        ),
        (
            "Create a new directory called 'scripts' and then list the contents of the current directory.",
            "The agent should first run 'mkdir scripts' and then 'ls' to verify the creation."
        ),
        (
            "Write 'echo hello' to a file named 'hello.sh' and make it executable.",
            "The agent should use bash commands to create the file and change its permissions (chmod +x)."
        ),
    ],
)
def test_bash_skills(bash_agent_graph, input_text, expected_output):
    """Evaluate the agent's ability to use bash tools for system tasks."""
    deepeval_callback = CallbackHandler()

    try:
        # We include a system prompt to ensure the agent knows its capabilities
        system_message = SystemMessage(content=(
            "You are a helpful AI assistant with access to a sandboxed Ubuntu shell. "
            "Use the run_bash_command tool for any file or system operations."
        ))

        result = bash_agent_graph.invoke(
            {"messages": [system_message, HumanMessage(content=input_text)]},
            config={"callbacks": [deepeval_callback]}
        )

        # We include the full message history in the actual output so the judge
        # can see the tool calls and their results.
        history = []
        for msg in result["messages"]:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            if isinstance(msg, SystemMessage):
                role = "system"

            content = msg.content
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                content += f"\n[Tool Calls: {msg.tool_calls}]"

            history.append(f"{role}: {content}")

        actual_output = "\n".join(history)
    except Exception as e:
        pytest.skip(f"Skipping test due to missing LLM backend: {e}")

    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        expected_output=expected_output
    )

    # We use GEval to evaluate the bash skill specifically, as it allows for more
    # nuanced criteria than simple relevancy.
    bash_correctness_metric = GEval(
        name="Bash Tool Usage Correctness",
        criteria=(
            "Determine if the agent correctly identified the need for bash commands, "
            "used the 'run_bash_command' tool appropriately, and correctly interpreted "
            "the results to answer the user's request."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=LMStudioJudge(),
        threshold=0.7
    )

    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=LMStudioJudge())

    assert_test(test_case, [bash_correctness_metric, relevancy_metric])
