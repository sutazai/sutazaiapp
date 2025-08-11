"""
SutazAI Scripts Package
=======================

Consolidated Python scripts for the SutazAI system.

This package contains:
- automation: Automation and scheduling scripts
- analysis: System analysis and auditing tools  
- utils: Common utilities and helpers
- deployment: Deployment and orchestration tools
- monitoring: System monitoring and health checks
- testing: Test runners and validation tools
- maintenance: System maintenance and cleanup
- security: Security validation and hardening
"""

__version__ = "1.0.0"
__author__ = "SutazAI Team"

# Import commonly used utilities
from .utils.common_utils import (
    setup_logging, 
    load_config,
    validate_ports,
    get_system_info
)

# Import core modules - avoid circular imports
try:
    from .monitoring.system_monitor import SystemMonitor
except ImportError:
    SystemMonitor = None

try:
    from .deployment.deployment_manager import DeploymentManager
except ImportError:
    DeploymentManager = None

try:
    from .testing.test_runner import TestRunner  
except ImportError:
    TestRunner = None

__all__ = [
    'setup_logging',
    'load_config', 
    'validate_ports',
    'get_system_info'
]

# Add available modules to __all__
if SystemMonitor:
    __all__.append('SystemMonitor')
if DeploymentManager:
    __all__.append('DeploymentManager') 
if TestRunner:
    __all__.append('TestRunner')