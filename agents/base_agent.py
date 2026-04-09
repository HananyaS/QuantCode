"""Abstract base class for all pipeline agents."""
from abc import ABC


class BaseAgent(ABC):
    """Every agent receives a shared context dict, updates its field, and returns the context."""

    def run(self, context: dict) -> dict:
        """Execute agent logic and return the updated context.

        Args:
            context: Shared mutable pipeline state.

        Returns:
            Updated context dict.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.run() must be implemented")
