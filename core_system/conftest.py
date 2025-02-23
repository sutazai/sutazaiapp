import os
import sys

import pytest
from loguru import logger


def pytest_configure(config):
    """
    Configure pytest for the document processor test suite
    
    Args:
        config (pytest.Config): Pytest configuration object
    """
    # Add project root to Python path
    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), 
            '..', '..', '..', '..'
        )
    )
    sys.path.insert(0, project_root)
    
    # Configure logging
    logger.add(
        os.path.join(project_root, 'logs', 'document_processor_tests.log'),
        rotation="10 MB",
        level="INFO"
    )
    
    # Add custom markers
    config.addinivalue_line(
        "markers", 
        "document_processor: mark test as a document processor agent test"
    )

def pytest_addoption(parser):
    """
    Add custom command-line options for pytest
    
    Args:
        parser (pytest.Parser): Pytest argument parser
    """
    parser.addoption(
        "--env", 
        action="store", 
        default="test", 
        help="Specify test environment"
    )
    parser.addoption(
        "--log-level", 
        action="store", 
        default="INFO", 
        help="Set logging level"
    )

@pytest.fixture(scope="session")
def test_environment(request):
    """
    Fixture to provide test environment configuration
    
    Args:
        request (pytest.FixtureRequest): Pytest fixture request
    
    Returns:
        Dict: Test environment configuration
    """
    env = request.config.getoption("--env")
    log_level = request.config.getoption("--log-level")
    
    # Environment-specific configurations
    environments = {
        'test': {
            'debug': False,
            'log_level': log_level,
            'temp_dir': '/tmp/sutazai_test'
        },
        'dev': {
            'debug': True,
            'log_level': 'DEBUG',
            'temp_dir': '/tmp/sutazai_dev'
        }
    }
    
    # Create temp directory if not exists
    os.makedirs(environments[env]['temp_dir'], exist_ok=True)
    
    return environments[env]

@pytest.fixture(scope="function")
def temp_document_dir(test_environment):
    """
    Fixture to provide a temporary directory for document processing tests
    
    Args:
        test_environment (Dict): Test environment configuration
    
    Returns:
        str: Path to temporary document directory
    """
    import tempfile
    
    with tempfile.TemporaryDirectory(
        prefix='sutazai_doc_test_', 
        dir=test_environment['temp_dir']
    ) as temp_dir:
        yield temp_dir

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Custom terminal summary for test results
    
    Args:
        terminalreporter (TerminalReporter): Pytest terminal reporter
        exitstatus (int): Exit status of the test run
        config (pytest.Config): Pytest configuration object
    """
    # Log test summary
    logger.info("Test Run Summary:")
    logger.info(f"Total Tests: {terminalreporter.stats.get('call', [])}")
    logger.info(f"Passed: {len(terminalreporter.stats.get('passed', []))}")
    logger.info(f"Failed: {len(terminalreporter.stats.get('failed', []))}")
    logger.info(f"Skipped: {len(terminalreporter.stats.get('skipped', []))}")
    logger.info(f"Exit Status: {exitstatus}")

def main():
    """
    Run tests with custom configuration
    """
    pytest.main([
        '-v',
        '--tb=short',
        '--capture=no',
        '--doctest-modules',
        '--cov=ai_agents.document_processor',
        '--cov-report=html',
        __file__
    ])

if __name__ == "__main__":
    main()