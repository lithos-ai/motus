"""LLM judge — scores a session's input/output via the model proxy.

Runs inside the agent server container, which already has OPENAI_API_KEY
and OPENAI_BASE_URL set to the platform's model proxy. Usage is billed to
the project's tenant via the existing API key auth flow.
"""

import json
import logging
import os

import httpx

from .schemas import JudgeResponse

logger = logging.getLogger("motus.serve.judge")


async def run_llm_judge(
    model: str,
    prompt: str,
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

    # Simple replace (not .format) so JSON braces in the prompt aren't
    # interpreted as format placeholders.
    filled_prompt = prompt.replace("{input}", user_input[:2000]).replace(
        "{output}", agent_output[:2000]
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": filled_prompt}],
                    "max_tokens": 256,
                },
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"]

        # Parse JSON from response (handle markdown code blocks)
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
