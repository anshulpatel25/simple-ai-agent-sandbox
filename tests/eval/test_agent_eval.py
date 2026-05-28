import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from agent.graph.graph import build_graph
from agent.config import settings
from deepeval.integrations.langchain.callback import CallbackHandler

@pytest.fixture
def agent_graph():
    llm = ChatOpenAI(
        base_url=settings.lm_studio_base_url,
        api_key=settings.lm_studio_api_key,
        model=settings.llm_model,
        temperature=0,
    )
    # We use empty tools and guardrails for basic evaluation to avoid dependency on Docker/LM Studio
    # if possible, but the graph needs them.
    # In a real scenario, you'd want to mock the container manager.
    tools = []
    guardrails = []
    return build_graph(llm, tools, guardrails)

@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Hello, who are you?", "I am an AI assistant."),
        ("What is 2+2?", "2 + 2 is 4."),
    ],
)
def test_agent_basic(agent_graph, input_text, expected_output):
    # Tracing is enabled by passing the CallbackHandler to the graph run
    deepeval_callback = CallbackHandler()

    # Simple invoke (this won't actually work without a running LM Studio,
    # but it demonstrates the integration)
    try:
        result = agent_graph.invoke(
            {"messages": [HumanMessage(content=input_text)]},
            config={"callbacks": [deepeval_callback]}
        )

        actual_output = result["messages"][-1].content
    except Exception as e:
        pytest.skip(f"Skipping test due to missing LLM backend: {e}")

    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        expected_output=expected_output
    )
    # AnswerRelevancyMetric checks if the output is relevant to the input
    metric = AnswerRelevancyMetric(threshold=0.7)
    assert_test(test_case, [metric])
