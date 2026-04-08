# =============================================================================
# Human-in-the-Loop Agent Example
# =============================================================================
#
# A file-management agent that asks for user approval before destructive
# operations and asks clarifying questions when instructions are ambiguous.
#
# Demonstrates:
#   - @tool(requires_approval=True) for tool-level approval gates
#   - ask_user_question for structured clarification from the user
#   - Both features working end-to-end through motus serve
#
# Without OPENROUTER_API_KEY, falls back to a deterministic echo agent
# so you can verify the serving infrastructure without API credits.
#
# -----------------------------------------------------------------------------
# Start the server
# -----------------------------------------------------------------------------
#
#   uv run motus serve start examples.serving.hitl_agent:agent --port 8000
#
# -----------------------------------------------------------------------------
# Verify via CLI chat
# -----------------------------------------------------------------------------
#
#   uv run motus serve chat http://localhost:8000
#
#   Try these prompts:
#
#   > delete /tmp/test.txt
#     → Agent calls delete_file → approval prompt appears → approve or reject
#
#   > list files in /tmp
#     → Agent calls list_files → no approval needed → returns immediately
#
#   > help me organize my project
#     → Agent calls ask_user_question → you pick options → agent continues
#
# -----------------------------------------------------------------------------
# Verify via curl (HTTP API)
# -----------------------------------------------------------------------------
#
#   # 1. Create session
#   curl -s -X POST http://localhost:8000/sessions | jq
#
#   # 2. Send message (triggers approval)
#   SID=<session_id from step 1>
#   curl -s -X POST http://localhost:8000/sessions/$SID/messages \
#     -H 'Content-Type: application/json' \
#     -d '{"role":"user","content":"delete /tmp/test.txt"}' | jq
#
#   # 3. Poll — should return status=interrupted with tool_approval
#   curl -s "http://localhost:8000/sessions/$SID?wait=true&timeout=10" | jq
#
#   # 4. Approve (use the interrupt_id from step 3)
#   IID=<interrupt_id from step 3>
#   curl -s -X POST http://localhost:8000/sessions/$SID/resume \
#     -H 'Content-Type: application/json' \
#     -d "{\"interrupt_id\":\"$IID\",\"value\":{\"approved\":true}}" | jq
#
#   # 5. Poll for final result
#   curl -s "http://localhost:8000/sessions/$SID?wait=true&timeout=10" | jq
#
# =============================================================================

import os

from motus.tools import tool
from motus.tools.builtins.ask_user import ask_user_question

# ── Tools ──────────────────────────────────────────────────────────


@tool(requires_approval=True)
async def delete_file(path: str) -> str:
    """Delete a file at the given path. Requires user approval."""
    # In a real agent this would call os.remove(path).
    return f"Deleted {path}"


@tool(requires_approval=True)
async def move_file(src: str, dst: str) -> str:
    """Move a file from src to dst. Requires user approval."""
    return f"Moved {src} → {dst}"


@tool
async def list_files(directory: str) -> str:
    """List files in a directory. Safe — no approval needed."""
    try:
        entries = os.listdir(directory)
        return "\n".join(entries[:50]) or "(empty)"
    except OSError as e:
        return f"Error: {e}"


@tool
async def read_file(path: str) -> str:
    """Read the first 200 lines of a file. Safe — no approval needed."""
    try:
        with open(path) as f:
            lines = f.readlines()[:200]
        return "".join(lines)
    except OSError as e:
        return f"Error: {e}"


# ── Agent ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a file management assistant. You can list, read, move, and delete files.

Rules:
- For destructive operations (delete, move), use the appropriate tool — the
  system will automatically ask the user for approval before executing.
- When the user's request is ambiguous (e.g., "organize my files" without
  specifying how), use ask_user_question to present concrete options.
- For safe reads (list_files, read_file), just proceed directly.
"""

ALL_TOOLS = [delete_file, move_file, list_files, read_file, ask_user_question]

if os.environ.get("OPENROUTER_API_KEY"):
    from motus.agent import ReActAgent
    from motus.models import OpenRouterChatClient

    agent = ReActAgent(
        client=OpenRouterChatClient(),
        model_name="anthropic/claude-sonnet-4",
        system_prompt=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
    )
else:
    # Fallback: deterministic agent that exercises HITL without an LLM.
    # Recognizes keywords in the user message and calls tools directly.
    from motus.models import ChatMessage

    async def agent(message: ChatMessage, state: list[ChatMessage]):
        content = (message.content or "").lower()

        if "delete" in content:
            from motus.serve.interrupt import interrupt

            decision = await interrupt(
                {
                    "type": "tool_approval",
                    "tool_name": "delete_file",
                    "tool_args": {"path": "/tmp/test.txt"},
                }
            )
            if decision.get("approved"):
                text = "Deleted /tmp/test.txt"
            else:
                text = "OK, cancelled the deletion."

        elif "organize" in content or "help" in content:
            from motus.serve.interrupt import interrupt

            answer = await interrupt(
                {
                    "type": "user_input",
                    "questions": [
                        {
                            "question": "How should I organize your files?",
                            "header": "Strategy",
                            "multiSelect": False,
                            "options": [
                                {
                                    "label": "By type",
                                    "description": "Group by extension (.py, .md, .json)",
                                },
                                {
                                    "label": "By date",
                                    "description": "Group by last-modified date",
                                },
                                {
                                    "label": "Flatten",
                                    "description": "Move everything into one directory",
                                },
                            ],
                        }
                    ],
                }
            )
            text = f"Got it, I'll organize by: {answer}"

        elif "list" in content:
            text = "bin\netc\ntmp\nusr\nvar"

        else:
            text = "I can list, read, move, or delete files. What would you like to do?"

        response = ChatMessage.assistant_message(content=text)
        return response, state + [message, response]
