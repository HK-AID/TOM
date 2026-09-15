"""TOM Tools System Foundation.

Provides the base Tool abstraction, ToolRegistry, and built-in tools.
"""
from tools.base import RiskLevel, Tool
from tools.registry import (
    DuplicateToolError,
    ToolError,
    ToolNotFoundError,
    ToolRegistry,
    default_registry,
)
from tools.system_info import SystemInfoTool, system_info_tool

__all__ = [
    "Tool",
    "RiskLevel",
    "ToolRegistry",
    "default_registry",
    "ToolError",
    "DuplicateToolError",
    "ToolNotFoundError",
    "SystemInfoTool",
    "system_info_tool",
]
