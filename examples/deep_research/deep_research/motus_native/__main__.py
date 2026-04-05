"""
Deep Research - Command line entry point.

Usage:
    python -m deep_research.motus_native [OPTIONS]
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime

from motus.models import OpenAIChatClient
from motus.tools import DictTools, WebSearchTool

from .config import (
    API_KEY,
    BASE_URL,
    BRAVE_API_KEY,
    MAX_REACT_ITERATIONS,
    MODEL,
)
from .researcher import deep_research_workflow

logger = logging.getLogger("DeepResearch")


async def main():
    parser = argparse.ArgumentParser(
        description="Deep research demo with ReAct-style iterative search."
    )
    parser.add_argument(
        "--question",
        default="What are promising applications of agentic AI in the next 5 years?",
        help="Research question to investigate.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_REACT_ITERATIONS,
        help=f"Maximum ReAct iterations per research task "
        f"(default: {MAX_REACT_ITERATIONS})",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Model to use (default: {MODEL})",
    )
    parser.add_argument(
        "--store-path",
        type=str,
        default=None,
        help="Path to save the final report as a markdown file",
    )
    args = parser.parse_args()

    client = OpenAIChatClient(api_key=API_KEY, base_url=BASE_URL)

    logger.info(f"Using model: {args.model}")
    logger.info(f"Max ReAct iterations: {args.max_iterations}")

    tools = DictTools({"web_search": WebSearchTool(BRAVE_API_KEY)})
    report_text = await deep_research_workflow(
        client,
        args.model,
        args.question,
        tools,
        max_iterations=args.max_iterations,
    )

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(report_text)

    # Save report to file if store_path is provided
    if args.store_path:
        store_path = args.store_path
        if os.path.isdir(store_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_report_{timestamp}.md"
            store_path = os.path.join(store_path, filename)
        os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
        # Write report
        with open(store_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info(f"Report saved to: {store_path}")


def run():
    """Entry point for the module."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
