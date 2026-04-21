"""LLM judge — scores a session's input/output via the model proxy.

Runs inside the agent server container, which already has OPENAI_API_KEY
and OPENAI_BASE_URL set to the platform's model proxy. Usage is billed to
the project's tenant via the existing API key auth flow.

The judge prompt structure is platform-owned:

  System prompt  (fixed)  — defines the judge role.
  User prompt    (built)  — injects input/output and appends developer criteria.

Developers only supply the ``criteria`` text. Structured output is enforced
by passing the ``JudgeResponse`` Pydantic model as ``response_format``, so
the model proxy returns a validated JSON object matching the schema — no
manual parsing, no markdown stripping.
"""

import logging
import os

from openai import AsyncOpenAI

from .schemas import JudgeResponse

logger = logging.getLogger("motus.serve.judge")


# Fixed system prompt. Developers cannot change this — it guarantees a
# consistent judge role across applications. The response format is
# enforced via ``response_format=JudgeResponse`` below.
_SYSTEM_PROMPT = (
    "You are an evaluator that judges whether an AI agent's response to a "
    "user's request meets the developer-supplied criteria."
)


def _build_user_message(user_input: str, agent_output: str, criteria: str) -> str:
    """User message template. Input/output injection is platform-owned;
    only ``criteria`` is developer-configurable."""
    return (
        f"User input:\n{user_input}\n\n"
        f"Agent output:\n{agent_output}\n\n"
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
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
        completion = await client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format=JudgeResponse,
            max_tokens=512,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            logger.warning("Judge returned no parsed object")
            return None
        # Clamp score into [0, 1] in case the model drifts slightly.
        return JudgeResponse(
            score=max(0.0, min(1.0, parsed.score)),
            passed=parsed.passed,
            reason=parsed.reason,
        )
    except Exception as e:
        logger.warning("Judge failed: %s", e)
        return None
