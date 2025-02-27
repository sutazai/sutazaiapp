# SutazAI Tests

This directory contains test files for the SutazAI project.

## Structure

- `test_main.py`: Tests for the main FastAPI application endpoints

## Running Tests

You can run all tests using pytest:

```bash
pytest
```

Or run a specific test file:

```bash
pytest tests/test_main.py
```

## Adding New Tests

When adding new tests:

1. Name test files with the prefix `test_` (e.g., `test_ai_agents.py`)
2. Name test functions with the prefix `test_` (e.g., `def test_agent_initialization():`)
3. Use appropriate fixtures for setup and teardown
4. Include docstrings explaining what each test is checking

## Test Coverage

To generate a test coverage report:

```bash
pytest --cov=. tests/
``` 