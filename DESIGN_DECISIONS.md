# Design Decision: Manual Graph vs. `create_agent`

This document summarizes the evaluation of using LangChain's `create_agent` factory versus the current manual `StateGraph` implementation.

## Recommendation: Stick with Manual Graph

After analysis, we have decided to keep the manual LangGraph implementation. While `create_agent` reduces boilerplate, it is less efficient and less safe for a sandboxed bash-based agent.

### 1. Sequential vs. Parallel Execution
- **Manual Graph:** Uses a standard `ToolNode` which executes tools sequentially. This is critical for shell commands where order matters (e.g., `mkdir` then `cd`).
- **`create_agent`:** Defaults to parallel execution using `Send`. This can cause race conditions and unpredictable behavior in a stateful filesystem container.

### 2. Centralized Guardrails
- **Manual Graph:** Our custom `guardrail_node` inspects the entire batch of `tool_calls` from the LLM. It can issue a single human-in-the-loop interrupt for the whole set and cancel all of them atomically if the user declines.
- **`create_agent`:** Guardrails would need to be implemented as middleware. Since `create_agent` processes tool calls via `Send`, the user might be prompted multiple times for a single turn, and atomic "all-or-nothing" cancellation is significantly harder to implement.

### 3. Transparency
- The manual graph explicitly defines the `agent -> guardrails -> tools -> agent` loop. This makes the logic easy to audit and modify as the agent's safety requirements evolve.

## Conclusion
The current implementation is optimized for **Safety** and **Predictability**, which are the primary goals of a sandboxed AI agent. The abstraction provided by `create_agent` is better suited for stateless API-based assistants.
