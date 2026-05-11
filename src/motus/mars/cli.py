from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .client import MarsOpenAIChatClient
from .replay import ReplaySummary, TraceReplayRunner


def collect_trace_paths(paths: Sequence[str | Path], *, limit: int | None = None) -> list[Path]:
    trace_paths: list[Path] = []
    for path_like in paths:
        path = Path(path_like)
        if path.is_dir():
            trace_paths.extend(sorted(path.rglob("*.json")))
        else:
            trace_paths.append(path)
    trace_paths = sorted(trace_paths)
    if limit is not None:
        return trace_paths[: max(0, limit)]
    return trace_paths


def _jsonl_row(row: dict[str, Any]) -> str:
    return json.dumps(row, separators=(",", ":"), sort_keys=True)


def _add_replay_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("traces", nargs="*", help="Trace JSON files or directories")
    parser.add_argument(
        "--traces",
        dest="trace_args",
        nargs="+",
        help="Trace JSON files or directories",
    )
    parser.add_argument("--model", required=True, help="Model name sent to Mars")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MARS_BASE_URL"),
        help="Mars OpenAI-compatible base URL, for example http://host:30000/v1",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MARS_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
        help="API key for the OpenAI-compatible client",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of root traces to replay concurrently",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        type=Path,
        help="Directory for summary.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of trace files after path collection and sorting",
    )


def build_parser(prog: str = "motus-mars-replay") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Replay Motus Mars trace files against an OpenAI-compatible Mars backend.",
    )
    _add_replay_args(parser)
    return parser


async def run_replay(args: argparse.Namespace) -> ReplaySummary:
    trace_inputs = [*args.traces, *(args.trace_args or [])]
    if not trace_inputs:
        raise ValueError("at least one trace file or directory is required")
    trace_paths = collect_trace_paths(trace_inputs, limit=args.limit)
    client = MarsOpenAIChatClient(api_key=args.api_key, base_url=args.base_url)
    runner = TraceReplayRunner(
        client=client,
        model=args.model,
        concurrency=args.concurrency,
    )
    return await runner.run_many(trace_paths)


def write_summary(summary: ReplaySummary, output_dir: Path | None) -> None:
    summary_json = json.dumps(asdict(summary), indent=2)
    if output_dir is None:
        print(summary_json)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(summary_json + "\n", encoding="utf-8")
    write_requests_jsonl(summary, output_dir / "requests.jsonl")
    write_errors_jsonl(summary, output_dir / "errors.jsonl")
    print(str(summary_path))


def write_requests_jsonl(summary: ReplaySummary, path: Path) -> None:
    rows = []
    for result in summary.results:
        for turn in result.turn_results:
            rows.append(
                _jsonl_row(
                    {
                        "trace_id": turn.trace_id,
                        "agent_instance_id": turn.agent_instance_id,
                        "agent_class_id": turn.agent_class_id,
                        "turn_index": turn.turn_index,
                        "output_tokens_requested": turn.output_tokens_requested,
                        "output_tokens_observed": turn.output_tokens_observed,
                        "planned_tools": turn.planned_tools,
                        "duration_ms_requested": turn.duration_ms_requested,
                        "started_at": turn.started_at,
                        "ended_at": turn.ended_at,
                        "finish_reason": turn.finish_reason,
                        "usage": turn.usage,
                    }
                )
            )
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_errors_jsonl(summary: ReplaySummary, path: Path) -> None:
    rows = [_jsonl_row(asdict(error)) for error in summary.errors]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = asyncio.run(run_replay(args))
    write_summary(summary, args.output_dir)
    return 1 if summary.failed_traces else 0


def _motus_command(args: argparse.Namespace) -> None:
    summary = asyncio.run(run_replay(args))
    write_summary(summary, args.output_dir)
    sys.exit(1 if summary.failed_traces else 0)


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser("mars-replay", help="replay Mars trace files")
    _add_replay_args(parser)
    parser.set_defaults(func=_motus_command)


if __name__ == "__main__":
    sys.exit(main())
