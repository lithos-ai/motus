"""LLM judge — scores a session's input/output via the model proxy.

Runs inside the agent server container, which already has OPENAI_API_KEY
and OPENAI_BASE_URL set to the platform's model proxy. Usage is billed to
the project's tenant via the existing API key auth flow.

The judge prompt structure is platform-owned:

  System prompt  (fixed)  — defines the judge role + output format.
  User prompt    (built)  — injects input/output and appends developer criteria.

Developers only supply the ``criteria`` text; everything else is handled
here so scoring is consistent across applications.
"""

import json
import logging
import os

import httpx

from .schemas import JudgeResponse

logger = logging.getLogger("motus.serve.judge")


# Fixed system prompt. Developers cannot change this — it guarantees a
# consistent judge role + output contract across applications.
_SYSTEM_PROMPT = (
    "You are an evaluator that judges whether an AI agent's response to a "
    "user's request meets the developer-supplied criteria.\n\n"
    "Respond with a single JSON object with exactly these fields:\n"
    '- "score": a number between 0.0 and 1.0\n'
    '- "passed": a boolean\n'
    '- "reason": a brief explanation string\n'
    "Do not include any other text."
)


def _build_user_message(user_input: str, agent_output: str, criteria: str) -> str:
    """User message template. Input/output injection is platform-owned;
    only ``criteria`` is developer-configurable."""
    return (
        f"User input:\n{user_input[:2000]}\n\n"
        f"Agent output:\n{agent_output[:2000]}\n\n"
        f"Criteria:\n{criteria.strip()}"
    )


async def run_llm_judge(
    model: str,
    criteria: str,
    user_input: str,
    agent_output: str,
) -> JudgeResponse | None:
    """Score a session turn using the configured LLM judge.

    Uses OPENAI_API_KEY + OPENAI_BASE_URL from the container environment.
    Returns a JudgeResponse or None if the call fails.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        logger.warning("Judge skipped: OPENAI_API_KEY or OPENAI_BASE_URL not set")
        return None

    user_content = _build_user_message(user_input, agent_output, criteria)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": 256,
                    "response_format": {"type": "json_object"},
                },
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"]

        # Some providers wrap JSON in markdown despite response_format.
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(text)
        return JudgeResponse(
            score=max(0.0, min(1.0, float(result.get("score", 0)))),
            passed=bool(result.get("passed", False)),
            reason=str(result.get("reason", "")),
        )
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Judge failed: %s", e)
        return None
