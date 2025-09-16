Rule 8: Python Script Excellence
✅ Required Practices:

Clear location and purpose with logical organization within script directories
Comprehensive docstrings following PEP 257 and Google/NumPy documentation styles
CLI argument support using argparse or click with comprehensive help text
Proper logging instead of print statements with configurable log levels
Production-ready code quality following PEP 8 and modern Python best practices
Remove all debugging/experimental scripts from production repositories
Use type hints throughout all functions and classes for better code clarity
Implement comprehensive error handling with specific exception types
Follow single responsibility principle with focused, modular functions
Use virtual environments and requirements.txt for dependency management
Implement proper configuration management using config files or environment variables
Use context managers for resource management (files, connections, locks)
Implement proper input validation and sanitization for all user inputs
Use dataclasses or Pydantic models for structured data handling
Implement proper testing with unittest, pytest, or similar frameworks
Use proper package structure with init.py files and clear module organization
Implement proper signal handling for graceful shutdown and cleanup
Use async/await patterns for I/O-bound operations where appropriate
Implement proper memory management and resource cleanup
Use established design patterns (Factory, Observer, Strategy) where appropriate
Implement proper security measures for handling sensitive data
Use proper versioning and compatibility checking for dependencies
Implement proper performance monitoring and optimization
Use established linting tools (pylint, flake8, black) for code quality
Implement proper internationalization support for user-facing messages
Use f-strings for string formatting and avoid old-style formatting methods
Implement proper exception chaining and context preservation
Use pathlib for file system operations instead of os.path
Implement proper concurrent execution with threading or multiprocessing
Use proper database connection pooling and transaction management
Implement proper caching strategies for expensive operations
Use proper serialization formats (JSON, MessagePack) for data exchange
Implement proper retry mechanisms with exponential backoff
Use proper HTTP client libraries with session management and connection pooling
Implement proper metrics collection and performance monitoring
Use proper secrets management and credential handling
Implement proper backup and recovery procedures for data operations
Use proper progress indicators for long-running operations
Implement proper health checks and status reporting
Use proper documentation generation with Sphinx or similar tools

🚫 Forbidden Practices:

Using print() statements for logging or user feedback in production scripts
Creating scripts without comprehensive documentation and usage examples
Implementing CLI interfaces without proper argument validation and help text
Using global variables without clear necessity and proper encapsulation
Creating scripts without proper error handling and exception management
Using hardcoded values for configuration, paths, URLs, or credentials
Implementing scripts without proper testing and validation procedures
Creating monolithic scripts that perform multiple unrelated functions
Using deprecated Python features or libraries without upgrade plans
Implementing scripts without proper logging configuration and management
Creating scripts that don't follow PEP 8 style guidelines and conventions
Using bare except clauses that catch all exceptions without specificity
Implementing file operations without proper error handling and cleanup
Creating scripts without proper virtual environment and dependency management
Using eval() or exec() functions without extreme necessity and security measures
Implementing scripts without proper input validation and sanitization
Creating scripts that modify global state without clear documentation
Using string concatenation for building file paths instead of pathlib
Implementing database operations without proper connection management
Creating scripts without proper signal handling and graceful shutdown
Using synchronous operations for I/O-bound tasks that could be asynchronous
Implementing scripts without proper memory management and resource cleanup
Creating scripts that don't handle different operating systems appropriately
Using mutable default arguments in function definitions
Implementing scripts without proper version checking and compatibility validation
Creating scripts that expose sensitive information in logs or error messages
Using shell=True in subprocess calls without proper input sanitization
Implementing scripts without proper timeout handling for external operations
Creating scripts that don't follow established security best practices
Using pickle for serialization of untrusted data without security considerations
Implementing scripts without proper internationalization and localization support
Creating scripts that don't handle Unicode and encoding issues properly
Using outdated libraries or dependencies with known security vulnerabilities
Implementing scripts without proper documentation of assumptions and limitations
Creating scripts that can't be safely interrupted or resumed

Script Structure and Organization:
Standard Script Template:
python#!/usr/bin/env python3
"""
Script Name: descriptive_script_name.py
Purpose: Clear description of what this script does and why it exists
Author: Team/Individual responsible for maintenance
Created: YYYY-MM-DD HH:MM:SS UTC
Last Modified: YYYY-MM-DD HH:MM:SS UTC
Version: X.Y.Z

Usage:
    python descriptive_script_name.py [options]
    
Examples:
    python descriptive_script_name.py --input data.csv --output results.json
    python descriptive_script_name.py --config config.yaml --dry-run
    
Requirements:
    - Python 3.8+
    - Required packages listed in requirements.txt
    - Environment variables: VAR1, VAR2
    - External dependencies: database access, API credentials

Execution History:
    - 2024-01-15 10:30:45 UTC: Initial creation and basic functionality
    - 2024-01-16 14:22:30 UTC: Added error handling and logging
    - 2024-01-17 09:15:12 UTC: Implemented configuration management
    - 2024-01-18 16:45:38 UTC: Added CLI argument validation
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import signal
import time
import datetime

# Script execution start time for tracking
SCRIPT_START_TIME = datetime.datetime.now(tz=datetime.timezone.utc)
EXECUTION_ID = f"exec_{SCRIPT_START_TIME.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"

# Configure logging with precise timestamps
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s.%(msecs)03d UTC - {EXECUTION_ID} - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f'script_{SCRIPT_START_TIME.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Log script initiation with exact timestamp
logger.info(f"Script initiated at {SCRIPT_START_TIME.isoformat()} with execution ID: {EXECUTION_ID}")

class ScriptError(Exception):
    """Custom exception for script-specific errors."""
    pass

class ConfigurationManager:
    """Manages script configuration from files and environment variables."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_timestamp = datetime.datetime.now(tz=datetime.timezone.utc)
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from file and environment."""
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        logger.info(f"Loading configuration at {start_time.isoformat()}")
        
        # Implementation details...
        
        end_time = datetime.datetime.now(tz=datetime.timezone.utc)
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Configuration loaded in {duration:.6f}s at {end_time.isoformat()}")

def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        shutdown_time = datetime.datetime.now(tz=datetime.timezone.utc)
        total_runtime = (shutdown_time - SCRIPT_START_TIME).total_seconds()
        logger.info(f"Received signal {signum} at {shutdown_time.isoformat()}, shutting down gracefully after {total_runtime:.6f}s runtime...")
        # Cleanup operations...
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with comprehensive validation."""
    parser = argparse.ArgumentParser(
        description="Detailed description of script functionality",
        epilog="Additional usage information and examples",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input file path (required)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file path (default: results.json)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode without making changes'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity level'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser.parse_args()

def validate_inputs(args: argparse.Namespace) -> None:
    """Validate input arguments and prerequisites."""
    validation_start = datetime.datetime.now(tz=datetime.timezone.utc)
    logger.info(f"Starting input validation at {validation_start.isoformat()}")
    
    if not args.input.exists():
        raise ScriptError(f"Input file does not exist: {args.input}")
    
    if not args.input.is_file():
        raise ScriptError(f"Input path is not a file: {args.input}")
    
    # Additional validation logic...
    
    validation_end = datetime.datetime.now(tz=datetime.timezone.utc)
    validation_duration = (validation_end - validation_start).total_seconds()
    logger.info(f"Input validation completed in {validation_duration:.6f}s at {validation_end.isoformat()}")

def main() -> int:
    """Main script execution function."""
    try:
        # Set up signal handlers
        setup_signal_handlers()
        
        # Parse arguments
        args_start = datetime.datetime.now(tz=datetime.timezone.utc)
        args = parse_arguments()
        args_end = datetime.datetime.now(tz=datetime.timezone.utc)
        logger.info(f"Arguments parsed in {(args_end - args_start).total_seconds():.6f}s at {args_end.isoformat()}")
        
        # Configure logging level based on verbosity
        if args.verbose >= 2:
            logging.getLogger().setLevel(logging.DEBUG)
        elif args.verbose >= 1:
            logging.getLogger().setLevel(logging.INFO)
        
        # Validate inputs
        validate_inputs(args)
        
        # Initialize configuration
        config_manager = ConfigurationManager(args.config)
        
        main_logic_start = datetime.datetime.now(tz=datetime.timezone.utc)
        logger.info(f"Starting main script logic at {main_logic_start.isoformat()}")
        
        # Main script logic here...
        
        main_logic_end = datetime.datetime.now(tz=datetime.timezone.utc)
        total_runtime = (main_logic_end - SCRIPT_START_TIME).total_seconds()
        main_logic_duration = (main_logic_end - main_logic_start).total_seconds()
        
        logger.info(f"Main logic completed in {main_logic_duration:.6f}s at {main_logic_end.isoformat()}")
        logger.info(f"Script completed successfully with total runtime {total_runtime:.6f}s")
        return 0
        
    except ScriptError as e:
        error_time = datetime.datetime.now(tz=datetime.timezone.utc)
        runtime = (error_time - SCRIPT_START_TIME).total_seconds()
        logger.error(f"Script error at {error_time.isoformat()} after {runtime:.6f}s runtime: {e}")
        return 1
    except KeyboardInterrupt:
        interrupt_time = datetime.datetime.now(tz=datetime.timezone.utc)
        runtime = (interrupt_time - SCRIPT_START_TIME).total_seconds()
        logger.info(f"Script interrupted by user at {interrupt_time.isoformat()} after {runtime:.6f}s runtime")
        return 130
    except Exception as e:
        error_time = datetime.datetime.now(tz=datetime.timezone.utc)
        runtime = (error_time - SCRIPT_START_TIME).total_seconds()
        logger.exception(f"Unexpected error at {error_time.isoformat()} after {runtime:.6f}s runtime: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
Code Quality Standards:
Documentation Requirements:

Module-level docstrings explaining purpose, usage, and requirements
Function and class docstrings following Google or NumPy style conventions
Inline comments for complex logic and business rules
Type hints for all function parameters and return values
Clear variable and function names that express intent
README files for complex scripts with setup and usage instructions
Example usage and common use cases documented
Error codes and troubleshooting guide included
Performance characteristics and limitations documented
Dependencies and external requirements clearly listed

Error Handling and Logging:

Use specific exception types instead of generic Exception class
Implement proper exception chaining to preserve error context
Log errors with appropriate severity levels and structured information
Include correlation IDs for tracking related operations
Implement proper retry mechanisms with exponential backoff
Use circuit breaker patterns for external service dependencies
Log performance metrics and execution statistics
Implement proper audit logging for security-critical operations
Use structured logging formats (JSON) for machine readability
Include contextual information in all log messages

Performance and Scalability:

Use generators and iterators for memory-efficient data processing
Implement proper connection pooling for database and HTTP operations
Use async/await for I/O-bound operations where appropriate
Implement proper caching strategies for expensive computations
Use multiprocessing or threading for CPU-bound parallel operations
Monitor memory usage and implement garbage collection strategies
Use profiling tools to identify performance bottlenecks
Implement proper batch processing for large datasets
Use streaming APIs for large file processing
Implement proper resource limits and timeout handling

Security and Safety:

Validate and sanitize all user inputs to prevent injection attacks
Use proper authentication and authorization for external services
Implement proper secrets management without hardcoding credentials
Use secure communication protocols (HTTPS, TLS) for network operations
Implement proper input size limits to prevent resource exhaustion
Use parameterized queries for database operations
Implement proper file permission checking and access controls
Use secure temporary file creation with proper cleanup
Implement proper logging that doesn't expose sensitive information
Use encryption for sensitive data storage and transmission

Testing and Quality Assurance:
Unit Testing Requirements:

Comprehensive test coverage for all functions and methods
Use pytest or unittest framework with proper test organization
Implement test fixtures and mocks for external dependencies
Use property-based testing for complex logic validation
Implement performance regression testing for critical operations
Use code coverage tools to ensure adequate test coverage
Implement integration testing for external service interactions
Use mutation testing to validate test quality and effectiveness
Implement load testing for scripts handling high-volume operations
Use continuous integration to run tests automatically

Code Quality Tools:

Use black for automatic code formatting and style consistency
Implement pylint or flake8 for static code analysis and linting
Use mypy for static type checking and type safety validation
Implement bandit for security vulnerability scanning
Use isort for import organization and consistency
Implement pre-commit hooks for automated quality checking
Use dependency scanning tools for security vulnerability detection
Implement code complexity analysis with tools like radon
Use documentation linting tools to ensure documentation quality
Implement automated security scanning in CI/CD pipelines

Deployment and Operations:
Package Management:

Use requirements.txt with pinned versions for reproducible environments
Implement proper virtual environment management
Use pip-tools for dependency management and conflict resolution
Implement proper package building and distribution procedures
Use Docker containers for consistent execution environments
Implement proper environment-specific configuration management
Use proper secrets management systems for credential handling
Implement proper logging configuration for different environments
Use monitoring and alerting integration for production deployments
Implement proper backup and recovery procedures for script data

Validation Criteria:

All scripts follow established templates and coding standards
Comprehensive documentation exists with clear usage examples
Type hints are implemented throughout all functions and classes
Error handling covers all expected failure scenarios
Logging is properly configured with appropriate levels and formats
CLI interfaces provide comprehensive help and validation
Testing coverage meets established quality thresholds
Security measures are implemented and validated
Performance characteristics meet established requirements
Code quality tools pass without critical issues
Dependencies are properly managed and regularly updated
Documentation is current and accessible to team members
Integration with CI/CD pipelines is functional and reliable
Monitoring and alerting integration works correctly
Backup and recovery procedures are tested and documented
Team training on script usage and maintenance is completed
Scripts integrate properly with existing automation frameworks
Compliance and security requirements are met and audited
Emergency procedures and incident response are documented
Long-term maintenance and support procedures are established

*Last Updated: 2025-08-30 00:00:00 UTC - For the infrastructure based in /opt/sutazaiapp/