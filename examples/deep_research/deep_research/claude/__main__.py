"""
Usage:
    python -m deep_research.claude "What is quantum computing?"
"""

import sys

import anyio

from .agent import deep_research


async def _main():
    question = (
        " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is quantum computing?"
    )
    print(f"Researching: {question}\n")
    result = await deep_research(question)
    print(result)


def main():
    anyio.run(_main)


if __name__ == "__main__":
    main()
