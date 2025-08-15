"""
AI Testing Framework for SutazAI
Enterprise-grade AI system testing and validation
"""

__version__ = "1.0.0"
__author__ = "SutazAI AI QA Team"

# Core testing modules - Import only if dependencies are available
try:
    from .model_validation import AIModelTestSuite
    _model_validation_available = True
except ImportError:
    _model_validation_available = False

try:
    from .data_quality import DataQualityTestSuite
    _data_quality_available = True
except ImportError:
    _data_quality_available = False

try:
    from .performance_validation import AIPerformanceTestSuite
    _performance_validation_available = True
except ImportError:
    _performance_validation_available = False

try:
    from .security_testing import AISecurityTestSuite
    _security_testing_available = True
except ImportError:
    _security_testing_available = False

# Export available modules
__all__ = []
if _model_validation_available:
    __all__.append("AIModelTestSuite")
if _data_quality_available:
    __all__.append("DataQualityTestSuite")
if _performance_validation_available:
    __all__.append("AIPerformanceTestSuite")
if _security_testing_available:
    __all__.append("AISecurityTestSuite")