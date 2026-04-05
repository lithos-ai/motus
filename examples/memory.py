"""
Memory examples — CompactionMemory and session restore.

Usage:
    python examples/memory.py <demo>

Demos:
    compaction         Interactive agent with context compaction (requires API key)
    session_restore    Save/restore conversation through agent interface (mock or live)
    memory_restore     Save/restore at the memory API level (no API key needed)

Examples:
    python examples/memory.py memory_restore
    python examples/memory.py session_restore
    python examples/memory.py session_restore --live
    OPENROUTER_API_KEY=sk-... python examples/memory.py compaction
"""

import argparse
import asyncio
import json
import logging
import os
import tempfile
from typing import Optional

from dotenv import load_dotenv

from motus.agent import ReActAgent
from motus.memory import CompactionMemory, CompactionMemoryConfig
from motus.models.base import (
    BaseChatClient,
    ChatCompletion,
    ChatMessage,
    FunctionCall,
    ToolCall,
    ToolDefinition,
)

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "warning").upper(),
    format="%(name)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _separator(title: str) -> None:
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")


def _print_messages(messages: list[ChatMessage], label: str) -> None:
    print(f"\n  [{label}] {len(messages)} messages:")
    for i, msg in enumerate(messages):
        content = (msg.content or "")[:100].replace("\n", " ")
        if len(msg.content or "") > 100:
            content += "..."
        tc = ""
        if msg.tool_calls:
            names = [t.function.name for t in msg.tool_calls]
            tc = f" [calls: {', '.join(names)}]"
        print(f"    {i + 1}. [{msg.role:9s}]{tc} {content}")


# ---------------------------------------------------------------------------
# Mock client for demos that don't need real API calls
# ---------------------------------------------------------------------------


class MockChatClient(BaseChatClient):
    """Deterministic mock client that simulates tool use and text responses."""

    def __init__(self):
        self._call_count = 0

    async def create(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: Optional[list[ToolDefinition]] = None,
        include_reasoning: bool = True,
        **kwargs,
    ) -> ChatCompletion:
        self._call_count += 1
        last_msg = messages[-1]

        if last_msg.role == "tool":
            return ChatCompletion(
                id=f"mock-{self._call_count}",
                model=model,
                content=f"The result is {last_msg.content or ''}.",
                finish_reason="stop",
            )

        user_text = (last_msg.content or "").lower()

        # Route to calculator tool
        if tools and any("calculator" in (t.name or "") for t in tools):
            if any(op in user_text for op in ["*", "+", "calculate", "divide"]):
                if "1234" in user_text and "5678" in user_text:
                    expr = "1234 * 5678"
                elif "divide" in user_text and "result" in user_text:
                    expr = "7006652 / 3"
                else:
                    expr = "42 + 1"
                return ChatCompletion(
                    id=f"mock-{self._call_count}",
                    model=model,
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=f"call_{self._call_count}",
                            function=FunctionCall(
                                name="calculator",
                                arguments=json.dumps({"expression": expr}),
                            ),
                        )
                    ],
                    finish_reason="tool_calls",
                )

        if "remember" in user_text and "number" in user_text:
            return ChatCompletion(
                id=f"mock-{self._call_count}",
                model=model,
                content="The number you asked me to remember was 42.",
                finish_reason="stop",
            )
        if "multiplication" in user_text or "earlier" in user_text:
            return ChatCompletion(
                id=f"mock-{self._call_count}",
                model=model,
                content="The multiplication result from earlier was 1234 * 5678 = 7,006,652.",
                finish_reason="stop",
            )
        if "remember" in user_text and "42" in user_text:
            return ChatCompletion(
                id=f"mock-{self._call_count}",
                model=model,
                content="Got it! I'll remember the number 42 for you.",
                finish_reason="stop",
            )

        return ChatCompletion(
            id=f"mock-{self._call_count}",
            model=model,
            content="I'm here to help! What would you like to know?",
            finish_reason="stop",
        )

    async def parse(self, model, messages, response_format, **kwargs):
        raise NotImplementedError("Mock client does not support parse()")


# ---------------------------------------------------------------------------
# Simple tools
# ---------------------------------------------------------------------------


def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file '{path}' not found."


def list_directory(path: str = ".") -> str:
    """List files in a directory."""
    try:
        return "\n".join(sorted(os.listdir(path)))
    except FileNotFoundError:
        return f"Error: directory '{path}' not found."


def current_time() -> str:
    """Return the current date and time."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Demo 1: Interactive compaction
# ---------------------------------------------------------------------------


async def compaction(args):
    """Interactive agent with CompactionMemory — shows auto/manual compaction.

    Commands: /status, /compact, /context, /quit
    """
    from motus.models import OpenRouterChatClient

    model = os.getenv("MODEL", "anthropic/claude-3-5-sonnet")
    compact_model = os.getenv("COMPACT_MODEL", "anthropic/claude-opus-4.6")

    def on_compact(stats: dict) -> None:
        _separator("COMPACTION TRIGGERED")
        print(f"  Messages compacted : {stats['messages_compacted']}")
        print(f"  Summary tokens     : {stats['summary_tokens']}")
        summary = stats["summary"]
        print(
            summary[:800]
            if len(summary) <= 800
            else summary[:800] + f"\n  ... ({len(summary)} chars)"
        )

    client = OpenRouterChatClient(api_key=os.getenv("OPENROUTER_API_KEY"))
    memory = CompactionMemory(
        config=CompactionMemoryConfig(
            compact_model_name=compact_model,
            token_threshold=args.threshold,
            safety_ratio=float(os.getenv("SAFETY_RATIO", "0.75")),
        ),
        on_compact=on_compact,
    )

    if args.load:
        with open(args.load) as f:
            data = json.load(f)
        memory._messages = [ChatMessage.model_validate(m) for m in data["messages"]]
        print(f"  Loaded {len(memory._messages)} messages from {args.load}")

    agent = ReActAgent(
        client=client,
        model_name=model,
        system_prompt="You are a helpful coding assistant.",
        memory=memory,
        tools=[read_file, list_directory],
        max_steps=10,
    )

    print(f"  Model: {model}")
    if args.threshold:
        print(f"  Token threshold: {args.threshold}")
    print("  Commands: /status, /compact, /context, /quit\n")

    while True:
        try:
            user_input = input("[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/status":
            tokens = memory.estimate_working_memory_tokens()
            trace = memory.get_memory_trace()
            print(
                f"  Messages: {len(memory.messages)}  Tokens: ~{tokens}  Compactions: {trace['total_compactions']}"
            )
            continue
        if user_input == "/compact":
            summary = await memory.compact()
            print("  (nothing to compact)" if summary is None else "  Compacted.")
            continue
        if user_input == "/context":
            for i, msg in enumerate(memory.get_context()):
                content = (msg.content or "")[:150].replace("\n", "\\n")
                if len(msg.content or "") > 150:
                    content += "..."
                print(f"  [{i}] {msg.role.upper()}: {content}")
            continue

        result = await agent(user_input)
        print(f"\n[Agent]: {result}")


# ---------------------------------------------------------------------------
# Demo 2: Session restore through agent interface
# ---------------------------------------------------------------------------


async def session_restore(args):
    """Save/restore a full conversation through the ReActAgent interface.

    Phases: chat → simulate shutdown → restore → continue conversation.
    Uses a mock client by default (no API key needed). Pass --live for real LLM.
    """
    tmp_dir = tempfile.mkdtemp(prefix="motus_session_demo_")
    log_path = f"{tmp_dir}/conversation_logs"

    if args.live:
        from motus.models import OpenRouterChatClient

        client = OpenRouterChatClient(api_key=os.getenv("OPENROUTER_API_KEY"))
        model = os.getenv("MODEL", "anthropic/claude-sonnet-4")
        mode = "LIVE"
    else:
        client = MockChatClient()
        model = "mock-model"
        mode = "MOCK (no API key needed)"

    system_prompt = "You are a helpful coding assistant. Keep answers concise."
    print(f"  Mode: {mode}  Model: {model}  Logs: {log_path}")

    # Phase 1: First session
    _separator("PHASE 1: Chat with Agent")
    memory = CompactionMemory(
        config=CompactionMemoryConfig(session_id="demo-001", log_base_path=log_path),
    )
    agent = ReActAgent(
        client=client,
        model_name=model,
        system_prompt=system_prompt,
        memory=memory,
        tools=[calculator, current_time],
        max_steps=5,
    )

    for prompt in [
        "What is 1234 * 5678?",
        "Now divide that result by 3",
        "Please remember the number 42 for me.",
    ]:
        print(f"\n  [You]: {prompt}")
        result = await agent(prompt)
        print(f"  [Agent]: {result}")

    session_id = memory.session_id
    msg_count = len(agent._memory.messages)
    _print_messages(agent._memory.messages, "Session 1")

    # Phase 2: Shutdown
    _separator("PHASE 2: Simulating Shutdown")
    del agent, memory
    print(f"  Agent deleted. session_id={session_id}, {msg_count} messages saved.")

    # Phase 3: Restore
    _separator("PHASE 3: Restore from Log")
    restored_memory = CompactionMemory.restore_from_log(
        session_id=session_id,
        log_base_path=log_path,
    )
    agent2 = ReActAgent(
        client=client,
        model_name=model,
        system_prompt=system_prompt,
        memory=restored_memory,
        tools=[calculator, current_time],
        max_steps=5,
    )
    restored_count = len(agent2._memory.messages)
    print(f"  Restored {restored_count} messages (expected {msg_count})")
    assert restored_count == msg_count

    # Phase 4: Continue
    _separator("PHASE 4: Continue — Agent Remembers")
    for prompt in [
        "What was the number I asked you to remember?",
        "And what was the multiplication result from earlier?",
    ]:
        print(f"\n  [You]: {prompt}")
        result = await agent2(prompt)
        print(f"  [Agent]: {result}")

    _separator("DEMO COMPLETE")


# ---------------------------------------------------------------------------
# Demo 3: Memory-level restore (no agent, no API key)
# ---------------------------------------------------------------------------


async def memory_restore(args):
    """Save/restore at the CompactionMemory API level.

    Demonstrates the full lifecycle: create → populate → restore → verify → compact → restore.
    No API key needed.
    """
    tmp_dir = tempfile.mkdtemp(prefix="motus_memory_demo_")
    log_path = f"{tmp_dir}/conversation_logs"
    print(f"  Log path: {log_path}")

    # Phase 1: Original session
    _separator("PHASE 1: Original Session")
    memory = CompactionMemory(
        config=CompactionMemoryConfig(
            session_id="demo-mem-001", log_base_path=log_path
        ),
        model_name="gpt-4o",
        compact_fn=lambda msgs, sp: f"Summary of {len(msgs)} messages",
    )
    memory.set_system_prompt("You are a helpful Python tutor.")

    conversations = [
        ("user", "How do I read a CSV file in Python?"),
        ("assistant", "Use pandas: `pd.read_csv('data.csv')`"),
        ("user", "What if the CSV has no header row?"),
        ("assistant", "Pass `header=None` to `pd.read_csv()`."),
        ("user", "How do I filter rows where col1 > 100?"),
        ("assistant", "Use boolean indexing: `df[df['col1'] > 100]`"),
    ]
    for role, content in conversations:
        msg = (
            ChatMessage.user_message(content)
            if role == "user"
            else ChatMessage.assistant_message(content)
        )
        await memory.add_message(msg)

    session_id = memory.session_id
    _print_messages(memory.messages, "Original")

    # Phase 2: Restore
    _separator("PHASE 2: Restore")
    restored = CompactionMemory.restore_from_log(
        session_id=session_id,
        log_base_path=log_path,
        model_name="gpt-4o",
        compact_fn=lambda msgs, sp: f"Summary of {len(msgs)} messages",
    )
    _print_messages(restored.messages, "Restored")

    assert len(memory.messages) == len(restored.messages)
    for orig, rest in zip(memory.messages, restored.messages):
        assert orig.role == rest.role and orig.content == rest.content
    print(f"\n  All {len(memory.messages)} messages match!")

    # Phase 3: Continue + compact + restore
    _separator("PHASE 3: Compact + Restore")
    await restored.add_message(ChatMessage.user_message("Can I also sort by col1?"))
    await restored.add_message(
        ChatMessage.assistant_message("Yes! Use `df.sort_values('col1')`."),
    )

    summary = await restored.compact()
    print(f'  Compacted! Summary: "{summary}"')
    await restored.add_message(ChatMessage.user_message("What about groupby?"))

    final = CompactionMemory.restore_from_log(
        session_id=session_id,
        log_base_path=log_path,
        model_name="gpt-4o",
        compact_fn=lambda msgs, sp: f"Summary of {len(msgs)} messages",
    )
    _print_messages(final.messages, "Restored after compaction")
    assert final._compaction_count == 1
    assert len(final.messages) == 2  # continuation + "What about groupby?"
    print("\n  All assertions passed!")

    _separator("DEMO COMPLETE")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEMOS = {
    "compaction": compaction,
    "session_restore": session_restore,
    "memory_restore": memory_restore,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Memory examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("demo", choices=DEMOS.keys(), help="Demo to run")
    parser.add_argument(
        "--live", action="store_true", help="Use real LLM (session_restore only)"
    )
    parser.add_argument(
        "--load", metavar="FILE", help="Load history JSON (compaction only)"
    )
    parser.add_argument(
        "--threshold", type=int, help="Token threshold (compaction only)"
    )
    args = parser.parse_args()

    asyncio.run(DEMOS[args.demo](args))
