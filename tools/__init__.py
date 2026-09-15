"""TOM Tools System Foundation.

Provides the base Tool abstraction and ToolRegistry for registering and
managing tools within TOM.
"""
from tools.base import RiskLevel, Tool
from tools.registry import (
    DuplicateToolError,
    ToolError,
    ToolNotFoundError,
    ToolRegistry,
    default_registry,
)

__all__ = [
    "Tool",
    "RiskLevel",
    "ToolRegistry",
    "default_registry",
    "ToolError",
    "DuplicateToolError",
    "ToolNotFoundError",
]
