"""Tool guardrails with a real agent — interactive demo.

A database assistant with guardrails that prevent dangerous SQL operations.
The agent receives guardrail errors as tool results and explains them to the user.

Usage:
  python examples/runtime/guardrails/tool_guardrails_agent.py
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from motus.agent import ReActAgent
from motus.guardrails import ToolInputGuardrailTripped
from motus.models.openai_client import OpenAIChatClient
from motus.runtime import shutdown
from motus.tools import tool

load_dotenv()
logging.basicConfig(level=os.getenv("MOTUS_LOG_LEVEL", "WARNING").upper())


# ── Guardrails ──────────────────────────────────────────────────


def reject_destructive_sql(query: str):
    """Block DROP, DELETE, TRUNCATE, ALTER statements."""
    dangerous = {"DROP", "DELETE", "TRUNCATE", "ALTER"}
    first_word = query.strip().split()[0].upper() if query.strip() else ""
    if first_word in dangerous:
        raise ToolInputGuardrailTripped(
            f"Destructive SQL blocked: {first_word} statements are forbidden"
        )


def reject_wildcard_select(query: str):
    """Block SELECT * (require explicit column names)."""
    normalized = " ".join(query.upper().split())
    if "SELECT *" in normalized:
        raise ToolInputGuardrailTripped(
            "SELECT * is not allowed — please specify column names explicitly"
        )


# ── Tool ────────────────────────────────────────────────────────

FAKE_DB = {
    "users": [
        {"id": 1, "name": "Alice", "email": "alice@example.com", "role": "admin"},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "role": "user"},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com", "role": "user"},
    ],
    "orders": [
        {"id": 101, "user_id": 1, "item": "Widget", "amount": 29.99},
        {"id": 102, "user_id": 2, "item": "Gadget", "amount": 49.99},
    ],
}


@tool(input_guardrails=[reject_destructive_sql, reject_wildcard_select])
async def run_sql(query: str) -> str:
    """Execute a SQL query against the database. Supports SELECT queries on users and orders tables."""
    q = query.strip().upper()
    if q.startswith("SELECT"):
        for table_name, rows in FAKE_DB.items():
            if table_name.upper() in q:
                return str(rows)
        return "No matching table found"
    return f"Query executed: {query}"


# ── Agent ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a database assistant. You help users query a database with two tables:
- users (id, name, email, role)
- orders (id, user_id, item, amount)

Use the run_sql tool to execute queries. If a query is blocked by a guardrail,
explain what happened and suggest a safe alternative."""


async def main():
    client = OpenAIChatClient(
        api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
    )
    agent = ReActAgent(
        client=client,
        model_name="gpt-4o-mini",
        name="DB Assistant",
        system_prompt=SYSTEM_PROMPT,
        tools=[run_sql],
    )

    print("Database Assistant (type 'quit' to exit)")
    print("Try: 'show me all users', 'drop the users table', 'select * from orders'")
    print()

    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip().lower() in ("quit", "exit"):
            break

        response = await agent(prompt)
        print(f"\n{response}\n")

    shutdown()


if __name__ == "__main__":
    asyncio.run(main())
