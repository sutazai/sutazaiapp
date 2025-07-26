# SutazAI Integration Tests

This directory contains integration tests for the SutazAI application, focusing on testing the interaction between components and API endpoints.

## Overview

Integration tests validate that different parts of the application work together correctly. These tests simulate real user interactions with the API endpoints and verify that the system behaves as expected.

## Test Files

- `test_api.py`: Tests for basic API functionality including health check and code generation endpoints
- `test_vector_search.py`: Tests for vector search and document storage functionality
- `test_orchestrator.py`: Tests for the AI orchestrator and performance metrics

## Running Integration Tests

### Prerequisites

- All application components should be correctly configured
- Test dependencies installed (`pip install -r requirements-test.txt`)

### Running Tests

```bash
# Run all integration tests
python -m pytest tests/integration/

# Run a specific integration test file
python -m pytest tests/integration/test_api.py

# Run with more detailed output
python -m pytest tests/integration/ -v
```

## Test Database

By default, integration tests will use a test database (`test_sutazaiapp`) to avoid affecting production data. The database configuration is set in `tests/conftest.py`.

## Mocking External Services

When appropriate, tests will mock external services like the vector database to ensure tests are reliable and don't depend on external systems. To see examples of mocking, check the `mock_vector_service` fixture in `test_vector_search.py`.

## Adding New Integration Tests

When adding new integration tests:

1. Create a new test file with a descriptive name
2. Use the TestClient from FastAPI for API testing
3. Create appropriate fixtures for any dependencies
4. Focus on testing the interaction between components
5. Use descriptive test names and docstrings 