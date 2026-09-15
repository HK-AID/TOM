"""Tests for TOM Tool System Foundation."""
import unittest
from abc import ABC
from typing import Any

from tools.base import RiskLevel, Tool
from tools.registry import (
    DuplicateToolError,
    ToolError,
    ToolNotFoundError,
    ToolRegistry,
    default_registry,
)
import tools


class DummyEchoTool(Tool):
    """A dummy tool for testing that echoes input text."""

    name = "dummy_echo"
    description = "Echoes input text for testing"
    risk_level = RiskLevel.LOW

    def execute(self, text: str = "") -> str:
        return text


class DummyCalculatorTool(Tool):
    """A dummy calculator tool for testing math operations."""

    name = "dummy_calculator"
    description = "Performs simple arithmetic"
    risk_level = RiskLevel.MEDIUM

    def execute(self, a: int = 0, b: int = 0, operation: str = "add") -> int:
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        raise ValueError(f"Unknown operation: {operation}")


class TestToolBase(unittest.TestCase):
    """Test suite for Tool abstraction and RiskLevel."""

    def test_tool_is_abstract(self):
        """Verify Tool cannot be instantiated directly without implementing execute()."""
        with self.assertRaises(TypeError):
            Tool()  # type: ignore[abstract]

    def test_tool_subclass_execution(self):
        """Verify concrete Tool subclass executes properly."""
        tool = DummyEchoTool()
        self.assertEqual(tool.name, "dummy_echo")
        self.assertEqual(tool.description, "Echoes input text for testing")
        self.assertEqual(tool.risk_level, "low")
        self.assertEqual(tool.execute("hello world"), "hello world")

    def test_tool_subclass_with_kwargs(self):
        """Verify concrete Tool executes with positional and keyword arguments."""
        calc = DummyCalculatorTool()
        self.assertEqual(calc.name, "dummy_calculator")
        self.assertEqual(calc.risk_level, "medium")
        self.assertEqual(calc.execute(2, 3, operation="add"), 5)
        self.assertEqual(calc.execute(3, 4, operation="multiply"), 12)

    def test_tool_instance_initialization_override(self):
        """Verify name, description, risk_level can be set via constructor."""
        class DynamicTool(Tool):
            def execute(self, *args: Any, **kwargs: Any) -> Any:
                return "ok"

        tool = DynamicTool(
            name="custom_tool",
            description="Custom description",
            risk_level=RiskLevel.HIGH,
        )
        self.assertEqual(tool.name, "custom_tool")
        self.assertEqual(tool.description, "Custom description")
        self.assertEqual(tool.risk_level, "high")
        self.assertEqual(tool.execute(), "ok")

    def test_risk_level_values(self):
        """Verify RiskLevel enum values and string compatibility."""
        self.assertEqual(RiskLevel.LOW.value, "low")
        self.assertEqual(RiskLevel.MEDIUM.value, "medium")
        self.assertEqual(RiskLevel.HIGH.value, "high")
        self.assertEqual(RiskLevel.CRITICAL.value, "critical")
        self.assertEqual(RiskLevel.LOW, "low")
        self.assertEqual(RiskLevel.HIGH, "high")

    def test_tool_to_dict(self):
        """Verify to_dict returns correct metadata dictionary."""
        tool = DummyEchoTool()
        data = tool.to_dict()
        self.assertEqual(
            data,
            {
                "name": "dummy_echo",
                "description": "Echoes input text for testing",
                "risk_level": "low",
            },
        )

    def test_tool_repr(self):
        """Verify tool repr format."""
        tool = DummyEchoTool()
        repr_str = repr(tool)
        self.assertIn("DummyEchoTool", repr_str)
        self.assertIn("dummy_echo", repr_str)
        self.assertIn("low", repr_str)


class TestToolRegistry(unittest.TestCase):
    """Test suite for ToolRegistry."""

    def setUp(self):
        """Create a fresh registry for each test."""
        self.registry = ToolRegistry()

    def test_register_and_get(self):
        """Verify registering a tool instance and retrieving it."""
        tool = DummyEchoTool()
        registered = self.registry.register(tool)
        self.assertIs(registered, tool)
        self.assertIs(self.registry.get("dummy_echo"), tool)

    def test_register_class(self):
        """Verify registering a tool class directly instantiates and registers it."""
        registered = self.registry.register(DummyCalculatorTool)
        self.assertIsInstance(registered, DummyCalculatorTool)
        self.assertEqual(registered.name, "dummy_calculator")
        self.assertIs(self.registry.get("dummy_calculator"), registered)

    def test_has(self):
        """Verify has() checks tool existence."""
        self.assertFalse(self.registry.has("dummy_echo"))
        self.registry.register(DummyEchoTool())
        self.assertTrue(self.registry.has("dummy_echo"))
        self.assertFalse(self.registry.has("unknown_tool"))

    def test_get_nonexistent_returns_default(self):
        """Verify get() returns default value when tool is missing."""
        self.assertIsNone(self.registry.get("nonexistent"))
        dummy_default = DummyEchoTool()
        self.assertIs(self.registry.get("nonexistent", default=dummy_default), dummy_default)

    def test_list_tools(self):
        """Verify list_tools() returns all registered tool instances."""
        echo = DummyEchoTool()
        calc = DummyCalculatorTool()
        self.registry.register(echo)
        self.registry.register(calc)

        tools_list = self.registry.list_tools()
        self.assertEqual(len(tools_list), 2)
        self.assertIn(echo, tools_list)
        self.assertIn(calc, tools_list)

    def test_list_names(self):
        """Verify list_names() returns all registered tool names."""
        self.registry.register(DummyEchoTool())
        self.registry.register(DummyCalculatorTool())

        names = self.registry.list_names()
        self.assertEqual(names, ["dummy_echo", "dummy_calculator"])

    def test_reject_duplicate_tool_names(self):
        """Verify registry rejects duplicate tool names with ValueError / DuplicateToolError."""
        self.registry.register(DummyEchoTool())

        # Second instance with same name
        duplicate = DummyEchoTool()
        with self.assertRaises(DuplicateToolError) as ctx:
            self.registry.register(duplicate)

        self.assertIn("already registered", str(ctx.exception))
        # Ensure DuplicateToolError is also a ValueError
        self.assertIsInstance(ctx.exception, ValueError)

    def test_reject_invalid_tool_type(self):
        """Verify registering non-Tool objects raises TypeError."""
        with self.assertRaises(TypeError):
            self.registry.register("not_a_tool")  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            self.registry.register(123)  # type: ignore[arg-type]

    def test_reject_tool_with_empty_name(self):
        """Verify registering a tool with empty name raises ValueError."""
        class EmptyNameTool(Tool):
            name = ""
            def execute(self, **kwargs: Any) -> Any:
                return None

        with self.assertRaises(ValueError):
            self.registry.register(EmptyNameTool())

        class WhitespaceNameTool(Tool):
            name = "   "
            def execute(self, **kwargs: Any) -> Any:
                return None

        with self.assertRaises(ValueError):
            self.registry.register(WhitespaceNameTool())

    def test_unregister(self):
        """Verify unregistering a tool removes it."""
        tool = DummyEchoTool()
        self.registry.register(tool)
        self.assertTrue(self.registry.has("dummy_echo"))

        unregistered = self.registry.unregister("dummy_echo")
        self.assertIs(unregistered, tool)
        self.assertFalse(self.registry.has("dummy_echo"))
        self.assertIsNone(self.registry.get("dummy_echo"))

    def test_unregister_nonexistent_raises_key_error(self):
        """Verify unregistering a nonexistent tool raises ToolNotFoundError / KeyError."""
        with self.assertRaises(ToolNotFoundError) as ctx:
            self.registry.unregister("nonexistent")
        self.assertIsInstance(ctx.exception, KeyError)

    def test_clear(self):
        """Verify clear() removes all tools from registry."""
        self.registry.register(DummyEchoTool())
        self.registry.register(DummyCalculatorTool())
        self.assertEqual(len(self.registry), 2)

        self.registry.clear()
        self.assertEqual(len(self.registry), 0)
        self.assertEqual(self.registry.list_tools(), [])

    def test_container_protocol(self):
        """Verify len, contains, getitem, and iteration protocol on ToolRegistry."""
        echo = DummyEchoTool()
        calc = DummyCalculatorTool()
        self.registry.register(echo)
        self.registry.register(calc)

        self.assertEqual(len(self.registry), 2)
        self.assertIn("dummy_echo", self.registry)
        self.assertNotIn("unknown", self.registry)

        self.assertIs(self.registry["dummy_echo"], echo)
        with self.assertRaises(KeyError):
            _ = self.registry["missing"]

        tools_from_iter = list(self.registry)
        self.assertEqual(tools_from_iter, [echo, calc])

    def test_repr(self):
        """Verify ToolRegistry repr."""
        self.registry.register(DummyEchoTool())
        self.assertIn("dummy_echo", repr(self.registry))

    def test_default_registry_instance(self):
        """Verify default_registry is an instance of ToolRegistry."""
        self.assertIsInstance(default_registry, ToolRegistry)


class TestToolsPackageExports(unittest.TestCase):
    """Test public exports from the tools package."""

    def test_public_exports(self):
        """Verify all required symbols are exported from tools."""
        self.assertTrue(hasattr(tools, "Tool"))
        self.assertTrue(hasattr(tools, "RiskLevel"))
        self.assertTrue(hasattr(tools, "ToolRegistry"))
        self.assertTrue(hasattr(tools, "default_registry"))
        self.assertTrue(hasattr(tools, "DuplicateToolError"))
        self.assertTrue(hasattr(tools, "ToolError"))
        self.assertTrue(hasattr(tools, "ToolNotFoundError"))

        self.assertIn("Tool", tools.__all__)
        self.assertIn("RiskLevel", tools.__all__)
        self.assertIn("ToolRegistry", tools.__all__)
        self.assertIn("default_registry", tools.__all__)
        self.assertIn("DuplicateToolError", tools.__all__)


if __name__ == "__main__":
    unittest.main()
