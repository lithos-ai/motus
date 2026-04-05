"""
Usage:
    python -m deep_research.openai_agents "What is quantum computing?"
"""

import asyncio
import sys

from .agent import deep_research


def main():
    question = (
        " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is quantum computing?"
    )
    print(f"Researching: {question}\n")
    result = asyncio.run(deep_research(question))
    print(result)


if __name__ == "__main__":
    main()
