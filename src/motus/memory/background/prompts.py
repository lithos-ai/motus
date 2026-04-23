"""System prompts for the background memory agents, loaded from markdown files."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.md"
    return path.read_text()


MEMORY_UPDATE_AGENT_PROMPT = _load_prompt("memory_update_agent")
MEMORY_SEARCH_AGENT_PROMPT = _load_prompt("memory_search_agent")
