"""Guardrails Demo — validate, transform, and block at every layer.

Tool input guardrails declare parameters matching the tool's signature —
the system extracts only the params the guardrail declares:
  - Return None   → pass through unchanged
  - Return a dict → partial update of kwargs
  - Raise         → block execution

Tool output guardrails receive the typed return value (before encoding):
  - Return None   → pass through unchanged
  - Return a value → replace the result
  - Raise         → block execution

Agent guardrails receive (value: str, agent) and return str | None.
When an agent uses response_format (structured output), output guardrails
declare fields from the BaseModel — same matching as tool input guardrails.

Run:  MOTUS_LOG_LEVEL=WARNING python examples/guardrails_demo.py
"""

import json
import re

from pydantic import BaseModel

from motus.guardrails import (
    InputGuardrailTripped,
    OutputGuardrailTripped,
    ToolInputGuardrailTripped,
    ToolOutputGuardrailTripped,
    run_guardrails,
    run_structured_output_guardrails,
)
from motus.runtime import shutdown
from motus.runtime.agent_task import agent_task
from motus.tools import FunctionTool, tool, tools

# ═══════════════════════════════════════════════════════════════════════
# 1. Tool input guardrail — block dangerous arguments
# ═══════════════════════════════════════════════════════════════════════


def demo_tool_input_block():
    print("── 1. Tool Input Guardrail: Block ──")

    def reject_drop_statements(query: str):
        """Guardrail declares 'query' — system extracts it from tool kwargs."""
        if "DROP" in query.upper():
            raise ToolInputGuardrailTripped("DROP statements are forbidden")

    async def execute_sql(query: str) -> str:
        """Execute a SQL query."""
        return f"result of: {query}"

    sql_tool = FunctionTool(execute_sql, input_guardrails=[reject_drop_statements])

    # Safe query passes through
    result = sql_tool(json.dumps({"query": "SELECT * FROM users"})).af_result()
    print(f"  Safe query:      {result}")
    assert "SELECT" in result

    # Dangerous query is blocked — error returned to agent, not thrown
    result = sql_tool(json.dumps({"query": "DROP TABLE users"})).af_result()
    print(f"  Blocked:         {result}")
    assert "DROP" in result and "error" in result

    print()


# ═══════════════════════════════════════════════════════════════════════
# 2. Tool input guardrail — modify arguments (partial update)
# ═══════════════════════════════════════════════════════════════════════


def demo_tool_input_modify():
    print("── 2. Tool Input Guardrail: Modify ──")

    def redact_api_keys(token: str) -> dict:
        """Return dict to partially update kwargs."""
        return {"token": re.sub(r"sk-[A-Za-z0-9]+", "[REDACTED]", token)}

    async def call_api(url: str, token: str) -> str:
        """Call an external API."""
        return f"called {url} with {token}"

    api_tool = FunctionTool(call_api, input_guardrails=[redact_api_keys])

    result = api_tool(
        json.dumps({"url": "https://api.example.com", "token": "sk-abc123secret"})
    ).af_result()
    print(f"  Result:          {result}")
    assert "sk-abc123secret" not in result
    assert "[REDACTED]" in result

    print()


# ═══════════════════════════════════════════════════════════════════════
# 3. Tool output guardrail — redact sensitive output
# ═══════════════════════════════════════════════════════════════════════


def demo_tool_output_redact():
    print("── 3. Tool Output Guardrail: Redact ──")

    def redact_passwords(result: str) -> str:
        """Receives the typed return value (str), not JSON."""
        return re.sub(r"password=\S+", "password=***", result)

    def block_if_error(result: str):
        if "error" in result:
            raise ToolOutputGuardrailTripped("Tool returned an error")

    async def get_user(user_id: int) -> str:
        """Look up user info."""
        return f"user={user_id} password=hunter2 role=admin"

    user_tool = FunctionTool(
        get_user, output_guardrails=[redact_passwords, block_if_error]
    )

    result = user_tool(json.dumps({"user_id": 42})).af_result()
    print(f"  Result:          {result}")
    assert "hunter2" not in result
    assert "password=***" in result

    print()


# ═══════════════════════════════════════════════════════════════════════
# 4. @tool decorator with guardrails
# ═══════════════════════════════════════════════════════════════════════


def demo_decorator_guardrails():
    print("── 4. @tool Decorator Guardrails ──")

    def max_length(text: str):
        """Guardrail declares 'text' — same name as the tool param."""
        if len(text) > 100:
            raise ToolInputGuardrailTripped("Input text exceeds 100 characters")

    def redact_emails(result: str) -> str:
        """Output guardrail receives the typed str result."""
        return re.sub(r"\S+@\S+\.\S+", "[EMAIL]", result)

    @tool(input_guardrails=[max_length], output_guardrails=[redact_emails])
    async def summarize(text: str) -> str:
        """Summarize text."""
        return f"summary: {text[:30]}... contact: admin@example.com"

    ft = FunctionTool(summarize)

    # Short text passes, email is redacted in output
    result = ft(json.dumps({"text": "Hello world"})).af_result()
    print(f"  Short text:      {result}")
    assert "admin@example.com" not in result
    assert "[EMAIL]" in result

    # Long text is blocked — error returned to agent, not thrown
    result = ft(json.dumps({"text": "x" * 200})).af_result()
    print(f"  Blocked:         {result}")
    assert "error" in result

    print()


# ═══════════════════════════════════════════════════════════════════════
# 5. @tools class-level guardrails
# ═══════════════════════════════════════════════════════════════════════


def demo_class_guardrails():
    print("── 5. @tools Class-Level Guardrails ──")

    call_log = []

    def log_all_calls(path: str):
        """Class-level: logs every tool call that has a 'path' param."""
        call_log.append(f"call: {path[:50]}")

    def validate_path(path: str):
        if ".." in path:
            raise ToolInputGuardrailTripped("Path traversal not allowed")

    @tools(input_guardrails=[log_all_calls])
    class FileTools:
        def __init__(self, root: str):
            self.root = root

        async def read(self, path: str) -> str:
            """Read a file."""
            return f"contents of {self.root}/{path}"

        @tool(input_guardrails=[validate_path])
        async def write(self, path: str, content: str) -> str:
            """Write a file."""
            return f"wrote to {self.root}/{path}"

    from motus.tools import normalize_tools

    file_tools = normalize_tools(FileTools("/data"))

    # read — uses class-level [log_all_calls]
    result = file_tools["read"](json.dumps({"path": "config.json"})).af_result()
    print(f"  read:            {result}")
    assert len(call_log) == 1

    # write — uses method-level [validate_path], overrides class-level
    result = file_tools["write"](
        json.dumps({"path": "out.txt", "content": "hello"})
    ).af_result()
    print(f"  write:           {result}")
    assert len(call_log) == 1  # log_all_calls was NOT called (overridden)

    # write with path traversal — blocked, error returned to agent
    result = file_tools["write"](
        json.dumps({"path": "../etc/passwd", "content": "hack"})
    ).af_result()
    print(f"  Blocked:         {result}")
    assert "error" in result

    print()


# ═══════════════════════════════════════════════════════════════════════
# 6. Guardrail chaining — sequential pipeline
# ═══════════════════════════════════════════════════════════════════════


def demo_chaining():
    print("── 6. Guardrail Chaining ──")

    def normalize_whitespace(text: str) -> dict:
        """Collapse repeated spaces in text."""
        return {"text": " ".join(text.split())}

    def lowercase(text: str) -> dict:
        """Lowercase text."""
        return {"text": text.lower()}

    def reject_profanity(text: str):
        """Block comments containing bad words."""
        bad_words = {"damn", "crap"}
        words = set(text.split())
        if words & bad_words:
            raise ToolInputGuardrailTripped("Profanity detected")

    @tool(input_guardrails=[normalize_whitespace, lowercase, reject_profanity])
    async def post_comment(text: str) -> str:
        """Post a comment."""
        return f"posted: {text}"

    ft = FunctionTool(post_comment)

    # Input flows through the pipeline: normalize → lowercase → check
    result = ft(json.dumps({"text": "  Hello   WORLD  "})).af_result()
    print(f"  Clean input:     {result}")
    assert "hello world" in result  # normalized and lowercased

    result = ft(json.dumps({"text": "This is DAMN bad"})).af_result()
    print(f"  Blocked:         {result}")
    assert "error" in result

    print()


# ═══════════════════════════════════════════════════════════════════════
# 7. Parallel guardrails — tripwire-only mode
# ═══════════════════════════════════════════════════════════════════════


def demo_parallel():
    print("── 7. Parallel Guardrails (tripwire) ──")

    checks_run = []

    def check_length(text: str):
        checks_run.append("length")
        if len(text) > 1000:
            raise InputGuardrailTripped("Too long")

    def check_language(text: str):
        checks_run.append("language")
        # Simulate language detection
        if "禁止" in text:
            raise InputGuardrailTripped("Forbidden content")

    def try_to_modify(text: str) -> str:
        checks_run.append("modify")
        return text.upper()  # This modification is IGNORED in parallel mode

    @agent_task
    async def run_parallel_guardrails(value):
        return await run_guardrails(
            [check_length, check_language, try_to_modify],
            value,
            parallel=True,
        )

    # All three guardrails run, but modifications are discarded
    checks_run.clear()
    result = run_parallel_guardrails("Hello world").af_result()
    print(f"  Result:          '{result}'")
    print(f"  Checks run:      {checks_run}")
    assert result == "Hello world"  # NOT uppercased — parallel ignores modifications
    assert len(checks_run) == 3  # all three ran

    # Tripwire fires even in parallel
    checks_run.clear()
    try:
        run_parallel_guardrails("禁止内容").af_result()
        assert False, "Should have been blocked"
    except InputGuardrailTripped as e:
        print(f"  Blocked:         {e.message}")

    print()


# ═══════════════════════════════════════════════════════════════════════
# 8. Agent guardrails — receive the agent instance
# ═══════════════════════════════════════════════════════════════════════


def demo_agent_guardrails():
    print("── 8. Agent Guardrails (with agent instance) ──")

    received = {}

    def check_with_agent(value: str, agent):
        received["agent_name"] = getattr(agent, "name", "unknown")
        received["value"] = value
        if "homework" in value.lower():
            raise InputGuardrailTripped("No homework help!")

    def redact_output(value: str, agent) -> str:
        return re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", value)

    # Simulate what AgentBase._execute does internally
    @agent_task
    async def simulate_agent_execute(prompt, agent):
        # Input guardrails
        prompt = await run_guardrails([check_with_agent], prompt, agent=agent)
        # ... agent._run() would happen here ...
        output = f"Agent {agent.name} says: SSN is 123-45-6789"
        # Output guardrails
        output = await run_guardrails([redact_output], output, agent=agent)
        return output

    class FakeAgent:
        name = "TestBot"

    agent = FakeAgent()

    # Normal prompt — guardrail receives the agent instance
    result = simulate_agent_execute("Tell me about Python", agent).af_result()
    print(f"  Result:          {result}")
    print(f"  Agent name seen: {received['agent_name']}")
    assert received["agent_name"] == "TestBot"
    assert "[SSN]" in result  # output was redacted
    assert "123-45-6789" not in result

    # Blocked prompt
    try:
        simulate_agent_execute("Help with homework", agent).af_result()
        assert False, "Should have been blocked"
    except InputGuardrailTripped as e:
        print(f"  Blocked:         {e.message}")

    print()


# ═══════════════════════════════════════════════════════════════════════
# 9. Structured output guardrails — field-matching on BaseModel results
# ═══════════════════════════════════════════════════════════════════════


def demo_structured_output_guardrails():
    print("── 9. Structured Output Guardrails (BaseModel) ──")

    class AnalysisResult(BaseModel):
        score: float
        summary: str
        raw_data: str

    # Only cares about 'score' — other fields untouched
    def validate_score(score: float):
        if score < 0 or score > 1:
            raise OutputGuardrailTripped("Score must be between 0 and 1")

    # Redact one field via partial update
    def redact_raw(raw_data: str) -> dict:
        return {"raw_data": "[REDACTED]"}

    # Access the agent instance
    received = {}

    def log_result(summary: str, agent):
        received["agent"] = getattr(agent, "name", "unknown")
        received["summary"] = summary

    class FakeAgent:
        name = "Analyst"

    agent = FakeAgent()

    # Simulate what AgentBase._execute does for structured output
    @agent_task
    async def simulate_structured_execute(result: AnalysisResult, agent_inst):
        guardrails = [validate_score, redact_raw, log_result]
        updated = await run_structured_output_guardrails(
            guardrails, result.model_dump(), agent=agent_inst
        )
        return AnalysisResult.model_validate(updated)

    # Valid result — redacted and logged
    original = AnalysisResult(score=0.85, summary="looks good", raw_data="secret-123")
    result = simulate_structured_execute(original, agent).af_result()
    print(f"  Score:           {result.score}")
    print(f"  Summary:         {result.summary}")
    print(f"  Raw data:        {result.raw_data}")
    print(f"  Agent seen:      {received['agent']}")
    assert result.score == 0.85  # untouched
    assert result.summary == "looks good"  # untouched
    assert result.raw_data == "[REDACTED]"  # redacted
    assert received["agent"] == "Analyst"

    # Invalid score — blocked
    bad = AnalysisResult(score=1.5, summary="bad", raw_data="data")
    try:
        simulate_structured_execute(bad, agent).af_result()
        assert False, "Should have been blocked"
    except OutputGuardrailTripped as e:
        print(f"  Blocked:         {e.message}")

    print()


# ═══════════════════════════════════════════════════════════════════════
# 10. Sync and async guardrails mixed
# ═══════════════════════════════════════════════════════════════════════


def demo_sync_async_mix():
    print("── 10. Sync + Async Guardrails ──")

    def sync_trim(query: str) -> dict:
        """Sync guardrail — runs on a background thread."""
        return {"query": query.strip()}

    async def async_validate(query: str):
        """Async guardrail — runs directly on the event loop."""
        if not query:
            raise ToolInputGuardrailTripped("query is required")

    async def search(query: str) -> str:
        """Search."""
        return f"results for: {query}"

    ft = FunctionTool(search, input_guardrails=[sync_trim, async_validate])

    result = ft(json.dumps({"query": "motus framework"})).af_result()
    print(f"  Result:          {result}")
    assert "motus" in result

    result = ft(json.dumps({"query": ""})).af_result()
    print(f"  Blocked:         {result}")
    assert "error" in result

    print()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    demo_tool_input_block()
    demo_tool_input_modify()
    demo_tool_output_redact()
    demo_decorator_guardrails()
    demo_class_guardrails()
    demo_chaining()
    demo_parallel()
    demo_agent_guardrails()
    demo_structured_output_guardrails()
    demo_sync_async_mix()

    shutdown()
    print("All guardrail demos passed!")


if __name__ == "__main__":
    main()
