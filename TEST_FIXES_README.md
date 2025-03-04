# Test Suite Fixes

This directory contains scripts to fix various issues in the test suite. Due to shell execution problems, individual Python scripts were created instead of a single shell script.

## Available Fix Scripts

1. **run_all_fixes.py** - Master script that runs all the fixes in sequence
2. **fix_indentation_and_decorators.py** - Fixes indentation issues and duplicate decorators in test files
3. **fix_coroutine_warnings.py** - Fixes unawaited coroutine warnings in `agent_manager.py`
4. **fix_sync_exception_test.py** - Fixes the implementation of the `test_sync_exception` method
5. **setup_pytest_config.py** - Sets up proper pytest configuration
6. **verify_fixes.py** - Verifies that all fixes have been properly applied

## How to Use

Since there are shell execution issues, you should run the Python scripts directly:

```bash
# Run all fixes at once
python run_all_fixes.py

# Or run individual fix scripts
python fix_indentation_and_decorators.py
python fix_coroutine_warnings.py
python fix_sync_exception_test.py
python setup_pytest_config.py

# Verify fixes
python verify_fixes.py
```

## Issues Fixed

### 1. Indentation and Decorator Issues

- Fixed improper indentation of `@pytest.mark.asyncio` decorators
- Removed duplicate `@pytest.mark.asyncio` decorators

### 2. Unawaited Coroutine Warnings

Fixed the `agent_manager.py` file to properly handle potentially unawaited coroutines when cancelling tasks:

```python
if self.heartbeat_task is not None:
    try:
        self.heartbeat_task.cancel()
        # For test mocks that might return a coroutine
        if hasattr(self.heartbeat_task, "_is_coroutine") and self.heartbeat_task._is_coroutine:
            await self.heartbeat_task
    except Exception as e:
        logger.warning(f"Error cancelling heartbeat task: {e}")
```

### 3. Test Sync Exception Method

Fixed the implementation of the `test_sync_exception` method in `tests/test_sync_manager_complete_coverage.py`:

```python
@pytest.mark.asyncio
async def test_sync_exception(self, sync_manager):
    """Test the sync method with an exception."""
    with patch.object(sync_manager, "sync_with_server", side_effect=Exception("Test exception")):
        # This should not raise an exception
        await sync_manager.sync()
        assert True  # If we get here, no exception was raised
```

### 4. Pytest Configuration

Created proper pytest configuration to handle asyncio tests:

**pyproject.toml**:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "asyncio: mark test as an asyncio test",
]
asyncio_default_fixture_loop_scope = "function"
```

**tests/conftest.py**:
```python
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "asyncio: mark test as an asyncio test")
```

## Verification

Run the verification script to check if all fixes have been properly applied:

```bash
python verify_fixes.py
```

## Running Tests

After applying the fixes:

```bash
# Run all tests
python -m pytest

# Generate coverage report
python -m pytest --cov=core_system.orchestrator --cov-report=html:coverage
``` 