"""Bash skill – lets the agent execute shell commands in a Docker container.

The skill reads its definition from ``skills/bash/skill.md`` at construction
time. The markdown content is embedded in the tool description so the LLM
understands what commands are available and how to use them.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from agent.container.manager import DockerContainerManager
from agent.skills.base import Skill
from agent.utils import get_resource_path

logger = logging.getLogger(__name__)

# Path to the skill definition, relative to the project root.
_SKILL_MD_PATH = get_resource_path(Path("skills") / "bash" / "skill.md")


def _load_skill_description(skill_md_path: Path) -> str:
    """Read and return the contents of skill.md.

    Args:
        skill_md_path: Path to the markdown skill definition file.

    Returns:
        The file contents as a string.

    Raises:
        FileNotFoundError: If the skill definition file is missing.
    """
    if not skill_md_path.exists():
        raise FileNotFoundError(
            f"Bash skill definition not found at '{skill_md_path}'. "
            "Ensure you are running the agent from the project root."
        )
    return skill_md_path.read_text(encoding="utf-8")


class BashSkill(Skill):
    """Skill that executes bash commands inside the session's Docker container.

    The :class:`~agent.container.manager.DockerContainerManager` is injected
    at construction time, keeping this class decoupled from Docker details.

    Args:
        container_manager: The active container manager for this session.
        skill_md_path: Override the default path to ``skill.md`` (useful in
            tests).
    """

    def __init__(
        self,
        container_manager: DockerContainerManager,
        skill_md_path: Path = _SKILL_MD_PATH,
    ) -> None:
        self._container_manager = container_manager
        self._skill_description = _load_skill_description(skill_md_path)
        logger.debug("BashSkill loaded description from '%s'.", skill_md_path)

    # ---------------------------------------------------------------- Skill
    @property
    def name(self) -> str:
        return "bash"

    def as_langchain_tool(self) -> BaseTool:
        """Build and return the LangChain tool, closing over the container."""
        container_manager = self._container_manager
        skill_description = self._skill_description

        tool_description = (
            "Execute a bash shell command inside a sandboxed Ubuntu Docker container. "
            "The container persists for the entire session so working directory, "
            "installed packages, and environment variables are preserved between calls.\n\n"
            "--- BASH SKILL REFERENCE ---\n"
            f"{skill_description}"
        )

        @tool(description=tool_description)
        def run_bash_command(command: str) -> str:
            """Run a bash command in the session's Ubuntu container.

            Args:
                command: The bash command to execute, e.g. ``"ls -la /tmp"``.

            Returns:
                Combined stdout and stderr output, or an error description.
            """
            logger.info("run_bash_command: %s", command)
            try:
                result = container_manager.exec(command)
                output = result.combined_output()
                if not result.succeeded:
                    return (
                        f"Command failed (exit code {result.exit_code}):\n{output}"
                    )
                return output
            except RuntimeError as exc:
                return f"Container error: {exc}"
            except Exception as exc:
                logger.exception("Unexpected error executing bash command.")
                return f"Unexpected error: {exc}"

        return run_bash_command  # type: ignore[return-value]
