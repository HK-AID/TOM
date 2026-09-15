"""Base Tool abstraction for TOM."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional


class RiskLevel(str, Enum):
    """Risk levels associated with tool execution."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __str__(self) -> str:
        return self.value


class Tool(ABC):
    """Abstract base class for all TOM tools.

    Attributes:
        name: The unique identifier for the tool.
        description: A human-readable description of what the tool does.
        risk_level: The risk level associated with executing this tool.
    """

    name: str = ""
    description: str = ""
    risk_level: str = RiskLevel.LOW.value

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> None:
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if risk_level is not None:
            self.risk_level = risk_level.value if isinstance(risk_level, Enum) else str(risk_level)

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool action.

        Args:
            *args: Positional arguments for tool execution.
            **kwargs: Keyword arguments for tool execution.

        Returns:
            Result of the tool execution.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the tool metadata."""
        risk_val = self.risk_level.value if isinstance(self.risk_level, Enum) else str(self.risk_level)
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": risk_val,
        }

    def __repr__(self) -> str:
        risk_val = self.risk_level.value if isinstance(self.risk_level, Enum) else str(self.risk_level)
        return (
            f"<{self.__class__.__name__} "
            f"name={self.name!r} risk_level={risk_val!r}>"
        )
