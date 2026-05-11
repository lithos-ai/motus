from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Sequence

from motus.agent.templates import CodingAgent
from motus.tools.providers.local import LocalShell

from .client import MarsOpenAIChatClient
from .record import record_agent_run


def load_prompt(
    *,
    prompt: str | None,
    prompt_file: str | Path | None,
) -> str:
    if prompt is not None:
        return prompt
    if prompt_file is None:
        raise ValueError("either prompt or prompt_file is required")
    return Path(prompt_file).read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motus-mars-record-coding-agent",
        description="Run a Motus CodingAgent against Mars and save a MotusTracing trace.",
    )
    parser.add_argument("--model", required=True, help="Model name sent to Mars")
    parser.add_argument("--trace-path", type=Path, required=True, help="Output trace JSON")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Prompt text for the CodingAgent")
    prompt_group.add_argument("--prompt-file", type=Path, help="File containing the prompt")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MARS_BASE_URL"),
        help="Mars OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MARS_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
        help="API key for the OpenAI-compatible client",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--agent-instance-id")
    parser.add_argument("--agent-class-id", default="coding-agent")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--disable-web", action="store_true")
    parser.add_argument("--disable-subagents", action="store_true")
    return parser


async def run_recording(args: argparse.Namespace) -> tuple[str, object]:
    agent_instance_id = args.agent_instance_id or args.trace_id
    client = MarsOpenAIChatClient(
        api_key=args.api_key,
        base_url=args.base_url,
        agent_instance_id=agent_instance_id,
        agent_class_id=args.agent_class_id,
    )
    agent = CodingAgent(
        client=client,
        model_name=args.model,
        sandbox=LocalShell(cwd=str(args.project_root)),
        project_root=args.project_root,
        enable_web=not args.disable_web,
        enable_subagents=not args.disable_subagents,
        max_steps=args.max_steps,
    )
    prompt = load_prompt(prompt=args.prompt, prompt_file=args.prompt_file)
    result, trace = await record_agent_run(
        agent,
        prompt,
        args.trace_path,
        trace_id=args.trace_id,
        agent_instance_id=agent_instance_id,
        agent_class_id=args.agent_class_id,
        source={},
    )
    return str(result), trace


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result, _trace = asyncio.run(run_recording(args))
    if args.result_path:
        args.result_path.parent.mkdir(parents=True, exist_ok=True)
        args.result_path.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
