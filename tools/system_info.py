"""System information tool for TOM."""
import os
import platform
import sys
from typing import Any, Dict, Tuple

from tools.base import RiskLevel, Tool


def _get_ram_windows() -> Tuple[int, int]:
    """Retrieve total and available physical RAM on Windows in bytes using ctypes."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
    except Exception:
        pass
    return 0, 0


def _get_ram_linux() -> Tuple[int, int]:
    """Retrieve total and available physical RAM on Linux from /proc/meminfo."""
    total_bytes = 0
    available_bytes = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                val_str = parts[1].strip().split()[0]
                val_bytes = int(val_str) * 1024
                if key == "MemTotal":
                    total_bytes = val_bytes
                elif key == "MemAvailable":
                    available_bytes = val_bytes
    except Exception:
        pass
    return total_bytes, available_bytes


def _get_ram_generic_unix() -> Tuple[int, int]:
    """Retrieve physical RAM on generic POSIX/Unix systems using os.sysconf."""
    total_bytes = 0
    available_bytes = 0
    try:
        if hasattr(os, "sysconf"):
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            total_bytes = int(page_size * phys_pages)
            if "SC_AVPHYS_PAGES" in os.sysconf_names:
                av_pages = os.sysconf("SC_AVPHYS_PAGES")
                available_bytes = int(page_size * av_pages)
    except Exception:
        pass
    return total_bytes, available_bytes


def _format_bytes(num_bytes: int) -> str:
    """Format byte count into a human-readable gigabyte string."""
    if num_bytes <= 0:
        return "Unknown"
    gb = num_bytes / (1024 ** 3)
    return f"{gb:.2f} GB"


def _get_ram_info() -> Dict[str, Any]:
    """Gather structured RAM information."""
    if sys.platform == "win32":
        total, available = _get_ram_windows()
    elif sys.platform.startswith("linux"):
        total, available = _get_ram_linux()
    else:
        total, available = _get_ram_generic_unix()

    used = max(0, total - available) if total > 0 and available > 0 else 0
    percent_used = round((used / total) * 100, 1) if total > 0 else 0.0

    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": used,
        "total": _format_bytes(total),
        "available": _format_bytes(available),
        "percent_used": percent_used,
    }


class SystemInfoTool(Tool):
    """Read-only tool that gathers host system specifications."""

    name: str = "system_info"
    description: str = (
        "Retrieves read-only structured host system information including "
        "operating system, CPU specifications, total and available RAM, and Python version."
    )
    risk_level: str = RiskLevel.LOW.value

    def execute(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Collect and return host system specifications.

        Returns:
            Dict[str, Any]: Structured system information.
        """
        ram_info = _get_ram_info()
        cpu_count = os.cpu_count() or 1

        return {
            "operating_system": {
                "name": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "architecture": platform.machine(),
                "platform": platform.platform(),
            },
            "cpu": {
                "processor": platform.processor() or platform.machine(),
                "cores": cpu_count,
                "architecture": platform.machine(),
            },
            "ram": ram_info,
            "total_ram": ram_info["total"],
            "available_ram": ram_info["available"],
            "python_version": platform.python_version(),
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "compiler": platform.python_compiler(),
                "executable": sys.executable,
            },
        }


# Explicit pre-instantiated tool instance (not registered automatically)
system_info_tool = SystemInfoTool()
