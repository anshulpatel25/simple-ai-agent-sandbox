"""Abstract base classes for skills and the skill registry.

Adding a new skill
------------------
1. Create a new module under ``agent/skills/``.
2. Subclass :class:`Skill` and implement ``name`` and ``as_langchain_tool``.
3. Register an instance with :class:`SkillRegistry`.

The rest of the system (graph, CLI) only ever talks to the registry, so new
skills are automatically picked up without touching any other file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool


class Skill(ABC):
    """Contract that every skill must fulfil."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique, human-readable skill name (e.g. ``"bash"``)."""
        ...

    @abstractmethod
    def as_langchain_tool(self) -> BaseTool:
        """Return a LangChain :class:`BaseTool` the LLM can invoke."""
        ...


class SkillRegistry:
    """Central registry that collects skills and exposes them as LangChain tools.

    The registry is intentionally simple: register skill instances, then ask
    for the tool list. It is the *only* place the graph needs to look when
    binding tools to the LLM.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill. Raises :class:`ValueError` on duplicate names."""
        if skill.name in self._skills:
            raise ValueError(
                f"A skill named '{skill.name}' is already registered. "
                "Each skill must have a unique name."
            )
        self._skills[skill.name] = skill

    def get_tools(self) -> list[BaseTool]:
        """Return all registered skills as LangChain tools."""
        return [skill.as_langchain_tool() for skill in self._skills.values()]

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        names = list(self._skills.keys())
        return f"SkillRegistry(skills={names})"
