"""Tool Registry for TOM."""
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from tools.base import Tool


class ToolError(Exception):
    """Base exception for tool-related errors."""
    pass


class DuplicateToolError(ToolError, ValueError):
    """Raised when attempting to register a tool with an already existing name."""
    pass


class ToolNotFoundError(ToolError, KeyError):
    """Raised when a requested tool is not found in the registry."""
    pass


class ToolRegistry:
    """Registry for managing and discovering TOM tools.

    Provides registration, retrieval, discovery, and lifecycle management
    for tools adhering to the Tool abstraction.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Union[Tool, Type[Tool]]) -> Tool:
        """Register a tool instance or tool class.

        Args:
            tool: A Tool instance or a Tool class.

        Returns:
            The registered Tool instance.

        Raises:
            TypeError: If the tool is not an instance or subclass of Tool.
            ValueError: If tool name is empty, not a string, or already registered.
            DuplicateToolError: If a tool with the same name is already registered
                (subclasses ValueError).
        """
        if isinstance(tool, type) and issubclass(tool, Tool):
            tool_instance = tool()
        elif isinstance(tool, Tool):
            tool_instance = tool
        else:
            raise TypeError(
                f"Expected a Tool instance or subclass, got {type(tool).__name__}"
            )

        name = getattr(tool_instance, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name must be a non-empty string.")

        tool_name = name.strip()
        if tool_name in self._tools:
            raise DuplicateToolError(
                f"Tool with name '{tool_name}' is already registered."
            )

        self._tools[tool_name] = tool_instance
        return tool_instance

    def get(self, name: str, default: Optional[Tool] = None) -> Optional[Tool]:
        """Retrieve a registered tool by name.

        Args:
            name: Name of the tool.
            default: Value to return if tool is not found. Defaults to None.

        Returns:
            The Tool instance if found, otherwise default.
        """
        return self._tools.get(name, default)

    def has(self, name: str) -> bool:
        """Check if a tool is registered by name.

        Args:
            name: Name of the tool.

        Returns:
            True if registered, False otherwise.
        """
        return name in self._tools

    def list_tools(self) -> List[Tool]:
        """List all registered tool instances.

        Returns:
            A list of all registered Tool instances.
        """
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """List the names of all registered tools.

        Returns:
            A list of registered tool names.
        """
        return list(self._tools.keys())

    def unregister(self, name: str) -> Tool:
        """Unregister and return a tool by name.

        Args:
            name: Name of the tool to unregister.

        Returns:
            The unregistered Tool instance.

        Raises:
            ToolNotFoundError: If the tool name is not registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' is not registered.")
        return self._tools.pop(name)

    def clear(self) -> None:
        """Remove all registered tools from the registry."""
        self._tools.clear()

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __getitem__(self, name: str) -> Tool:
        tool = self.get(name)
        if tool is None:
            raise ToolNotFoundError(f"Tool '{name}' is not registered.")
        return tool

    def __repr__(self) -> str:
        names = list(self._tools.keys())
        return f"<ToolRegistry tools={names}>"


# Default global registry instance
default_registry = ToolRegistry()
