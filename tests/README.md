# SutazAI Testing Suite

This directory contains the tests for the SutazAI application.

## Running Tests

### Prerequisites

- Python 3.8+
- pytest
- All dependencies installed (`pip install -r requirements.txt`)

### Running All Tests

```bash
python -m pytest
```

### Running Unit Tests Only

```bash
python -m pytest tests/unit/
```

### Running a Specific Test

```bash
python -m pytest tests/unit/test_document_processing.py::test_pdf_processor
```

### Running with Coverage

```bash
python -m pytest --cov=backend
```

## Test Structure

- `unit/`: Unit tests for individual components
- `integration/`: Tests for component interactions
- `test_data/`: Sample files and data for testing

## Adding New Tests

1. Create a new test file in the appropriate directory
2. Use descriptive test function names
3. Use fixtures for setup/teardown
4. Mock external dependencies when appropriate

## Test Conventions

- All test files should be named with the prefix `test_`
- Test functions should be prefixed with `test_`
- Use pytest fixtures for reusable setup
- Use parameterized tests for testing multiple cases
- Mock external services when possible 