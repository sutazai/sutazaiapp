import pytest
import os
import sys
from pathlib import Path

# Add the project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Environment setup for testing
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup the test environment with test configurations."""
    # Set environment variables for testing
    os.environ["TESTING"] = "1"
    os.environ["DEBUG"] = "1"

    # Use in-memory database for tests
    os.environ["POSTGRES_DB"] = "test_sutazaiapp"
    os.environ["DOCUMENT_STORE_PATH"] = str(
        Path(os.getcwd()) / "tests" / "test_data" / "documents"
    )

    # Create test directories if they don't exist
    Path(os.environ["DOCUMENT_STORE_PATH"]).mkdir(parents=True, exist_ok=True)

    yield

    # Cleanup after tests if needed
    # ...


# Add any shared fixtures here that multiple tests might need
