from __future__ import annotations

import re
from typing import Optional
from langchain_core.messages import ToolCall
from agent.guardrails.base import Guardrail

class FileDeletionGuardrail(Guardrail):
    """Guardrail that triggers on file or directory deletion commands."""

    # Patterns that suggest deletion in a bash command
    _DELETION_PATTERNS = [
        r"\brm(\s+|$)",           # rm followed by space or end of line
        r"\brmdir\b",        # rmdir command
        r"\bunlink\b",       # unlink command
        r"\bapt(-get)?\s+(remove|purge)\b",  # apt/apt-get remove/purge
    ]

    @property
    def name(self) -> str:
        return "file_deletion"

    def check(self, tool_call: ToolCall) -> Optional[str]:
        """Check if the bash command contains a deletion instruction."""
        if tool_call["name"] != "run_bash_command":
            return None

        command = tool_call["args"].get("command", "")
        for pattern in self._DELETION_PATTERNS:
            if re.search(pattern, command):
                return f"The command '{command}' might delete files or directories. Are you sure you want to proceed?"

        return None
