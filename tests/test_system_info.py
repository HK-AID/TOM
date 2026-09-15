"""Tests for TOM system_info tool."""
import platform
import sys
import unittest
from unittest.mock import mock_open, patch

from tools.base import RiskLevel, Tool
from tools.registry import ToolRegistry, default_registry
from tools.system_info import (
    SystemInfoTool,
    _format_bytes,
    _get_ram_linux,
    system_info_tool,
)


class TestSystemInfoTool(unittest.TestCase):
    """Test suite for SystemInfoTool."""

    def setUp(self):
        """Create tool instance for testing."""
        self.tool = SystemInfoTool()

    def test_tool_metadata(self):
        """Verify tool name, risk_level, and description."""
        self.assertEqual(self.tool.name, "system_info")
        self.assertEqual(self.tool.risk_level, "low")
        self.assertEqual(self.tool.risk_level, RiskLevel.LOW)
        self.assertIsInstance(self.tool, Tool)
        self.assertTrue(len(self.tool.description) > 0)

    def test_execute_structure(self):
        """Verify execute() returns the required top-level and structured fields."""
        data = self.tool.execute()

        self.assertIsInstance(data, dict)
        self.assertIn("operating_system", data)
        self.assertIn("cpu", data)
        self.assertIn("ram", data)
        self.assertIn("total_ram", data)
        self.assertIn("available_ram", data)
        self.assertIn("python_version", data)
        self.assertIn("python", data)

    def test_operating_system_information(self):
        """Verify operating system details."""
        os_info = self.tool.execute()["operating_system"]

        self.assertIsInstance(os_info, dict)
        self.assertEqual(os_info["name"], platform.system())
        self.assertEqual(os_info["release"], platform.release())
        self.assertEqual(os_info["version"], platform.version())
        self.assertEqual(os_info["architecture"], platform.machine())
        self.assertEqual(os_info["platform"], platform.platform())

    def test_cpu_information(self):
        """Verify CPU details."""
        cpu_info = self.tool.execute()["cpu"]

        self.assertIsInstance(cpu_info, dict)
        self.assertGreaterEqual(cpu_info["cores"], 1)
        self.assertIsInstance(cpu_info["processor"], str)
        self.assertIsInstance(cpu_info["architecture"], str)

    def test_ram_information(self):
        """Verify RAM details and format."""
        data = self.tool.execute()
        ram_info = data["ram"]

        self.assertIsInstance(ram_info, dict)
        self.assertIn("total_bytes", ram_info)
        self.assertIn("available_bytes", ram_info)
        self.assertIn("used_bytes", ram_info)
        self.assertIn("total", ram_info)
        self.assertIn("available", ram_info)
        self.assertIn("percent_used", ram_info)

        # On Windows host, RAM should be non-zero
        if sys.platform == "win32":
            self.assertGreater(ram_info["total_bytes"], 0)
            self.assertGreater(ram_info["available_bytes"], 0)
            self.assertGreaterEqual(ram_info["total_bytes"], ram_info["available_bytes"])
            self.assertTrue(data["total_ram"].endswith("GB"))
            self.assertTrue(data["available_ram"].endswith("GB"))
            self.assertGreaterEqual(ram_info["percent_used"], 0.0)
            self.assertLessEqual(ram_info["percent_used"], 100.0)

    def test_python_information(self):
        """Verify Python version and runtime details."""
        data = self.tool.execute()

        self.assertEqual(data["python_version"], platform.python_version())
        self.assertEqual(data["python"]["version"], platform.python_version())
        self.assertEqual(data["python"]["implementation"], platform.python_implementation())
        self.assertEqual(data["python"]["executable"], sys.executable)

    def test_read_only_and_repeatable(self):
        """Verify execute() is read-only and idempotent."""
        first = self.tool.execute()
        second = self.tool.execute()

        # Invariant hardware and OS specs must remain strictly identical
        self.assertEqual(first["operating_system"], second["operating_system"])
        self.assertEqual(first["cpu"], second["cpu"])
        self.assertEqual(first["python_version"], second["python_version"])
        self.assertEqual(first["python"], second["python"])
        self.assertEqual(first["ram"]["total_bytes"], second["ram"]["total_bytes"])

    def test_execute_accepts_arbitrary_kwargs(self):
        """Verify execute accepts optional args and kwargs gracefully without error."""
        result = self.tool.execute("dummy_arg", option="value")
        self.assertIsInstance(result, dict)
        self.assertIn("operating_system", result)

    def test_no_automatic_registration_on_import(self):
        """Verify importing SystemInfoTool does not modify or register into default_registry."""
        self.assertFalse(default_registry.has("system_info"))
        self.assertIsNone(default_registry.get("system_info"))

    def test_explicit_registration_with_default_registry(self):
        """Verify tool can be explicitly registered into default_registry without side effects."""
        try:
            self.assertFalse(default_registry.has("system_info"))
            default_registry.register(system_info_tool)
            self.assertTrue(default_registry.has("system_info"))
            self.assertIs(default_registry.get("system_info"), system_info_tool)
        finally:
            if default_registry.has("system_info"):
                default_registry.unregister("system_info")

    def test_register_in_custom_registry(self):
        """Verify tool can be registered in a custom ToolRegistry instance."""
        custom_registry = ToolRegistry()
        tool = SystemInfoTool()
        custom_registry.register(tool)

        self.assertTrue(custom_registry.has("system_info"))
        self.assertIs(custom_registry.get("system_info"), tool)
        self.assertEqual(len(custom_registry.list_tools()), 1)

    def test_linux_meminfo_parsing(self):
        """Verify /proc/meminfo parsing helper correctly extracts values."""
        sample_meminfo = (
            "MemTotal:       16384000 kB\n"
            "MemFree:         4096000 kB\n"
            "MemAvailable:    8192000 kB\n"
            "Buffers:          500000 kB\n"
        )
        with patch("builtins.open", mock_open(read_data=sample_meminfo)):
            total, available = _get_ram_linux()
            self.assertEqual(total, 16384000 * 1024)
            self.assertEqual(available, 8192000 * 1024)

    def test_format_bytes_helper(self):
        """Verify byte formatting helper function."""
        self.assertEqual(_format_bytes(0), "Unknown")
        self.assertEqual(_format_bytes(-100), "Unknown")
        self.assertEqual(_format_bytes(1024 ** 3), "1.00 GB")
        self.assertEqual(_format_bytes(int(7.5 * (1024 ** 3))), "7.50 GB")


if __name__ == "__main__":
    unittest.main()
