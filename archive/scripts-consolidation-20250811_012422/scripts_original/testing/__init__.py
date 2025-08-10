"""
Testing Module
==============

Consolidated testing and validation utilities.
Replaces 284+ testing scripts with a unified module.
"""

from .test_runner import (
    TestRunner,
    TestConfig,
    TestSuite,
    TestResult
)

from .validators import (
    SystemValidator,
    ServiceValidator,
    SecurityValidator,
    PerformanceValidator
)

from .load_testing import (
    LoadTester,
    run_performance_tests,
    stress_test_services
)

__all__ = [
    'TestRunner', 'TestConfig', 'TestSuite', 'TestResult',
    'SystemValidator', 'ServiceValidator', 'SecurityValidator', 'PerformanceValidator',
    'LoadTester', 'run_performance_tests', 'stress_test_services'
]