from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.messages import ToolCall

logger = logging.getLogger(__name__)

class Guardrail(ABC):
    """Abstract base class for all guardrails."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the guardrail."""
        pass

    @abstractmethod
    def check(self, tool_call: ToolCall) -> Optional[str]:
        """Check if a tool call triggers this guardrail.

        Args:
            tool_call: The tool call to inspect.

        Returns:
            A string containing the confirmation prompt if the guardrail is
            triggered, otherwise None.
        """
        pass

class GuardrailRegistry:
    """Registry for managing and accessing guardrails."""

    def __init__(self) -> None:
        self._guardrails: dict[str, Guardrail] = {}

    def register(self, guardrail: Guardrail) -> None:
        """Register a new guardrail.

        Args:
            guardrail: The guardrail instance to register.

        Raises:
            ValueError: If a guardrail with the same name is already registered.
        """
        if guardrail.name in self._guardrails:
            raise ValueError(f"Guardrail '{guardrail.name}' is already registered.")
        self._guardrails[guardrail.name] = guardrail
        logger.debug("Registered guardrail: %s", guardrail.name)

    def get_guardrails(self) -> list[Guardrail]:
        """Return all registered guardrails."""
        return list(self._guardrails.values())
