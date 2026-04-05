"""Skills module for motus agents.

Skills are self-contained units of knowledge/instructions that agents can load
on demand. Each skill is a directory containing a SKILL.md file with YAML
frontmatter (name, description) and markdown instructions.

Usage:
    from motus.agent.skills import SkillTool

    agent = ReActAgent(
        client=client,
        model_name=model,
        system_prompt="...",
        tools=[my_tools, SkillTool("path/to/skills/")],
    )
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from motus.tools import FunctionTool

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A self-contained unit of knowledge/instructions for an agent."""

    name: str
    description: str
    instructions: str
    path: str


def load_skill(skill_dir: str | Path) -> Skill:
    """Load a skill from a directory containing SKILL.md.

    The SKILL.md file should have YAML frontmatter with 'name' and 'description'
    fields, followed by markdown instructions:

        ---
        name: my_skill
        description: What this skill does
        version: 1.0.0
        ---
        # My Skill
        Instructions here...

    Args:
        skill_dir: Path to a directory containing a SKILL.md file.

    Returns:
        A Skill object with parsed metadata and instructions.

    Raises:
        FileNotFoundError: If SKILL.md doesn't exist in the directory.
        ValueError: If SKILL.md has no valid YAML frontmatter.
    """
    skill_dir = Path(skill_dir)
    skill_md = skill_dir / "SKILL.md"

    if not skill_md.exists():
        raise FileNotFoundError(f"No SKILL.md found in {skill_dir}")

    content = skill_md.read_text(encoding="utf-8")

    # Parse YAML frontmatter
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", content, re.DOTALL)
    if not match:
        raise ValueError(f"No YAML frontmatter in {skill_md}")

    frontmatter = yaml.safe_load(match.group(1))
    if not frontmatter or not isinstance(frontmatter, dict):
        raise ValueError(f"Invalid YAML frontmatter in {skill_md}")

    instructions = match.group(2).strip()

    return Skill(
        name=frontmatter.get("name", skill_dir.name),
        description=frontmatter.get("description", ""),
        instructions=instructions,
        path=str(skill_md),
    )


def load_skills(skills_dir: str | Path) -> list[Skill]:
    """Load all skills from subdirectories of skills_dir.

    Scans for subdirectories containing SKILL.md files. Directories without
    SKILL.md are silently skipped.

    Args:
        skills_dir: Path to a directory containing skill subdirectories.

    Returns:
        List of Skill objects, sorted by name.
    """
    skills_dir = Path(skills_dir)
    skills = []

    if not skills_dir.is_dir():
        logger.warning(f"Skills directory not found: {skills_dir}")
        return skills

    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "SKILL.md").exists():
            continue
        try:
            skills.append(load_skill(entry))
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Skipping skill in {entry}: {e}")

    return skills


class SkillTool:
    """Exposes a ``load_skill`` tool backed by a skills directory.

    Usage::

        from motus.agent.skills import SkillTool

        tools = {**my_tools, **SkillTool("path/to/skills/").build_tools()}
    """

    def __init__(self, skills_dir: str | Path):
        skills = load_skills(skills_dir)
        self._skill_map = {s.name: s for s in skills}
        self._skill_listing = "\n".join(f"- {s.name}: {s.description}" for s in skills)

    def build_tools(self) -> dict[str, FunctionTool]:
        """Build a dict containing the ``load_skill`` FunctionTool."""
        skill_map = self._skill_map
        skill_listing = self._skill_listing

        async def load_skill(skill_name: str) -> str:
            skill = skill_map.get(skill_name)
            if not skill:
                available = ", ".join(skill_map.keys())
                return f"Unknown skill '{skill_name}'. Available: {available}"
            return skill.instructions

        load_skill.__doc__ = (
            "Load detailed instructions for a skill.\n\n"
            f"Available skills:\n{skill_listing}\n\n"
            "Args:\n"
            "    skill_name: Name of the skill to load.\n\n"
            "Returns:\n"
            "    The skill's instructions as markdown text."
        )

        return {"load_skill": FunctionTool(load_skill)}
