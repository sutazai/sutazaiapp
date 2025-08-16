#!/usr/bin/env python3
"""
MCP Test Utilities Package

Comprehensive utility functions and helpers for MCP automation testing.
Provides reusable testing components, reporting utilities, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test services,
and test data management for the MCP testing framework.

Components:
- Test data generators and factories
- Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test service providers
- Test reporting and metrics collection
- Test environment management
- Assertion helpers and validators
- Performance measurement utilities

Author: Claude AI Assistant (senior-automated-tester)
Created: 2025-08-15 UTC
Version: 1.0.0
"""

from .test_data import TestDataFactory, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestMCPServer, TestPackageGenerator
from .reporting import TestReporter, PerformanceReporter, ComplianceReporter
from .assertions import MCPAssertions, PerformanceAssertions, SecurityAssertions
from .Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Tests import Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestServices, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestHealthChecker, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestDownloadManager
from .environment import TestEnvironmentManager, TestIsolation, ResourceManager

__version__ = "1.0.0"

# Export key utilities for easy access
__all__ = [
    # Test data utilities
    "TestDataFactory",
    "Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestMCPServer", 
    "TestPackageGenerator",
    
    # Reporting utilities
    "TestReporter",
    "PerformanceReporter",
    "ComplianceReporter",
    
    # Assertion utilities
    "MCPAssertions",
    "PerformanceAssertions", 
    "SecurityAssertions",
    
    # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test services
    "Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestServices",
    "Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestHealthChecker",
    "Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestDownloadManager",
    
    # Environment management
    "TestEnvironmentManager",
    "TestIsolation",
    "ResourceManager"
]

# Utility configuration
UTILS_CONFIG = {
    "default_timeout": 30,
    "max_test_data_size": 1000,
    "performance_tolerance": 0.1,
    "report_retention_days": 30
}