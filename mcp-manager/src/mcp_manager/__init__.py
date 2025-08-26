"""
Dynamic MCP Management System

A comprehensive orchestrator for Model Context Protocol (MCP) servers providing:
- Dynamic server discovery and lifecycle management
- Health monitoring with auto-recovery
- Unified interface for all MCP servers
- Configuration management and validation
- Performance monitoring and metrics
"""

__version__ = "1.0.0"
__author__ = "SutazAI System"

from .manager import MCPManager
from .connection import ConnectionManager
from .discovery import ServerDiscoveryEngine
from .health import HealthMonitor
from .interface import UnifiedMCPInterface

__all__ = [
    "MCPManager",
    "ConnectionManager", 
    "ServerDiscoveryEngine",
    "HealthMonitor",
    "UnifiedMCPInterface",
]