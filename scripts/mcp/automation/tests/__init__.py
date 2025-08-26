#!/usr/bin/env python3
"""
MCP Automation Testing Framework

Comprehensive testing suite for MCP server automation system.
Provides integration testing, health validation, performance benchmarks,
security testing, compatibility checks, and rollback validation.

Test Categories:
- Integration: End-to-end MCP server deployment and operation testing
- Health: Service health validation and monitoring verification
- Performance: Load testing and resource utilization benchmarks
- Security: Vulnerability scanning and security compliance validation
- Compatibility: Version compatibility and system integration testing
- Rollback: Disaster recovery and rollback procedure validation

Author: Claude AI Assistant (senior-automated-tester)
Created: 2025-08-15 UTC
Version: 1.0.0

Usage:
    # Run all tests
    pytest scripts/mcp/automation/tests/

    # Run specific test category
    pytest scripts/mcp/automation/tests/ -m integration
    pytest scripts/mcp/automation/tests/ -m performance
    
    # Run with coverage
    pytest scripts/mcp/automation/tests/ --cov=scripts.mcp.automation
"""

import sys
import os
from pathlib import Path

# Add automation module to path for testing
automation_root = Path(__file__).parent.parent
if str(automation_root) not in sys.path:
    sys.path.insert(0, str(automation_root))

# Test framework version
__version__ = "1.0.0"

# Test categories for pytest markers
TEST_CATEGORIES = [
    "integration",
    "health", 
    "performance",
    "security",
    "compatibility",
    "rollback"
]

# Test configuration
TEST_CONFIG = {
    "timeout_short": 30,        # Short operations (health checks)
    "timeout_medium": 300,      # Medium operations (downloads)
    "timeout_long": 900,        # Long operations (full deployments)
    "retry_attempts": 3,
    "test_data_retention_days": 7
}

# Export key modules for test imports
__all__ = [
    "TEST_CATEGORIES",
    "TEST_CONFIG",
    "__version__"
]