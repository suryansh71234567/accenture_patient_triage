"""
registry.py  (skills)
---------------------
SkillRegistry — lazy loader for SKILL.md procedure files.

Skills are procedural markdown documents that the agent loads ONLY when
the relevant workflow is active. They are NEVER all injected into the
system prompt simultaneously — that would bloat every turn with irrelevant
clinical procedures.

Usage
-----
registry = SkillRegistry()
registry.register("triage_assessment", Path(".../triage_assessment/SKILL.md"))
text = registry.load("triage_assessment")   # reads file on first call, cached
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent


class SkillRegistry:
    """
    Registry and lazy loader for agent skills.

    Each skill is a SKILL.md file with YAML-like frontmatter:
        ---
        name: triage_assessment
        description: One-line description for registry listing.
        ---

    The body of the file is the procedural instruction set.
    """

    def __init__(self) -> None:
        self._skills: Dict[str, Path] = {}
        self._cache: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, str]] = {}

    def register(self, skill_name: str, path: Path) -> None:
        """Register a skill by name and path to its SKILL.md."""
        if not path.exists():
            raise FileNotFoundError(f"SKILL.md not found at {path}")
        self._skills[skill_name] = path
        logger.debug("Skill registered: %s → %s", skill_name, path)

    def load(self, skill_name: str) -> Optional[str]:
        """
        Load and return the skill text for the given name.
        Returns None if the skill is not registered.
        Caches the result after the first read.
        """
        if skill_name not in self._skills:
            logger.warning("Skill %r is not registered.", skill_name)
            return None

        if skill_name not in self._cache:
            path = self._skills[skill_name]
            try:
                text = path.read_text(encoding="utf-8")
                self._cache[skill_name] = text
                self._metadata[skill_name] = _parse_frontmatter(text)
            except Exception as exc:
                logger.error("Failed to load skill %r: %s", skill_name, exc)
                return None

        return self._cache[skill_name]

    def get_description(self, skill_name: str) -> str:
        """Return the one-line description from the skill's frontmatter."""
        if skill_name not in self._metadata:
            self.load(skill_name)
        return self._metadata.get(skill_name, {}).get("description", "")

    def list_skills(self) -> List[Dict[str, str]]:
        """Return a list of {name, description} dicts for all registered skills."""
        result = []
        for name in self._skills:
            desc = self.get_description(name)
            result.append({"name": name, "description": desc})
        return result

    def is_registered(self, skill_name: str) -> bool:
        return skill_name in self._skills

    def clear_cache(self) -> None:
        """Clear the in-memory cache (useful for testing)."""
        self._cache.clear()
        self._metadata.clear()

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"SkillRegistry({list(self._skills.keys())})"


def _parse_frontmatter(text: str) -> Dict[str, str]:
    """
    Extract simple key: value pairs from YAML frontmatter (--- block).
    Returns an empty dict if no frontmatter is found.
    """
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}
    meta: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta


def build_default_registry() -> SkillRegistry:
    """
    Build a SkillRegistry pre-loaded with all standard agent skills.
    Skills are loaded from the skills/ subdirectory of this package.
    """
    registry = SkillRegistry()
    skills_dir = _DEFAULT_SKILLS_DIR

    skill_names = [
        "patient_lookup",
        "patient_update",
        "triage_assessment",
        "xgb_explanation",
        "rag_reasoning",
        "hospital_status",
        "routing",
        "human_review",
    ]

    for name in skill_names:
        skill_path = skills_dir / name / "SKILL.md"
        if skill_path.exists():
            registry.register(name, skill_path)
        else:
            logger.warning("SKILL.md not found for skill %r at %s", name, skill_path)

    return registry
