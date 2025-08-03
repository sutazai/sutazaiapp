# Functionality Preservation Validator

A comprehensive agent that enforces **Rule 2: Do Not Break Existing Functionality** by analyzing code changes and preventing regressions before they reach production.

## Overview

The Functionality Preservation Validator is a production-ready tool that:

- **Tracks all file modifications** and analyzes their impact on existing functionality
- **Detects breaking changes** in functions, classes, APIs, and interfaces
- **Runs automated tests** before and after changes to catch regressions
- **Creates dependency graphs** to understand usage patterns
- **Integrates with git hooks** to prevent breaking commits
- **Provides detailed reports** with actionable suggestions

## Features

### 🔍 AST Analysis
- Parses Python code to extract function signatures, class definitions, and API endpoints
- Compares before/after states to identify breaking changes
- Tracks parameter changes, return types, and method removals

### 🌐 API Monitoring
- Detects Flask and FastAPI endpoint changes
- Validates HTTP method modifications
- Checks parameter and response schema changes

### 🧪 Test Integration
- Runs test suites on both previous and current code states
- Compares test results to identify new failures
- Provides detailed test regression analysis

### 📊 Dependency Analysis
- Maps import relationships between modules
- Identifies files that depend on modified code
- Warns about potential cascade effects

### 🚫 Git Integration
- Installs pre-commit hooks automatically
- Blocks commits with breaking changes
- Provides rollback suggestions

## Installation

1. **Place the validator in your project**:
   ```bash
   # The script should be at: scripts/agents/functionality-preservation-validator.py
   chmod +x scripts/agents/functionality-preservation-validator.py
   ```

2. **Install git hooks** (recommended):
   ```bash
   python scripts/agents/functionality-preservation-validator.py setup-hooks
   ```

3. **Install dependencies**:
   ```bash
   pip install ast-tools pytest networkx
   ```

## Usage

### Basic Validation

Validate all changes against the main branch:
```bash
python scripts/agents/functionality-preservation-validator.py validate
```

Validate against a specific branch:
```bash
python scripts/agents/functionality-preservation-validator.py validate --base-branch develop
```

### Output Formats

**Summary format** (for CI/CD):
```bash
python scripts/agents/functionality-preservation-validator.py validate --format summary
```

**Detailed format** (for developers):
```bash
python scripts/agents/functionality-preservation-validator.py validate --format detailed
```

**JSON format** (for tools integration):
```bash
python scripts/agents/functionality-preservation-validator.py validate --format json
```

### Analyze Specific Files

Analyze specific files without running full validation:
```bash
python scripts/agents/functionality-preservation-validator.py analyze --files src/api.py src/models.py
```

### Generate Reports

Create a comprehensive validation report:
```bash
python scripts/agents/functionality-preservation-validator.py report --output validation-report.json
```

### Test-Only Validation

Run only test comparison:
```bash
python scripts/agents/functionality-preservation-validator.py test
```

## Configuration

The validator can be configured using `functionality-preservation-config.yaml`:

```yaml
general:
  base_branch: "main"
  log_level: "INFO"

rules:
  functions:
    check_signature_changes: true
    check_parameter_removal: true
    allow_parameter_addition: true
  
  classes:
    check_method_removal: true
    check_inheritance_changes: true
  
  api:
    check_endpoint_removal: true
    check_parameter_changes: true

severity:
  function_removed: "error"
  method_removed: "error"
  api_endpoint_removed: "error"
  function_signature_changed: "error"
```

## Validation Categories

### ❌ Breaking Changes (Block Commit)
- **Function Removed**: A function that was previously available is no longer present
- **Class Removed**: A class definition has been deleted
- **Method Removed**: A method has been removed from a class
- **API Endpoint Removed**: An HTTP endpoint is no longer available
- **Function Signature Changed**: Function parameters or return types modified
- **Test Regression**: Previously passing tests now fail

### ⚠️ Warnings (Review Required)
- **Function Defaults Changed**: Default parameter values modified
- **Class Inheritance Changed**: Base classes modified
- **Import Removed**: Module imports removed
- **API Parameters Changed**: Endpoint parameters modified
- **Configuration Changed**: Config files modified

### ✅ Safe Changes (Informational)
- **Function Added**: New functions added
- **Method Added**: New methods added to classes
- **Test Improvement**: Previously failing tests now pass

## Integration Examples

### CI/CD Pipeline Integration

```yaml
# .github/workflows/validation.yml
name: Functionality Validation
on: [pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
        with:
          fetch-depth: 0  # Need full history for comparison
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run functionality validation
        run: |
          python scripts/agents/functionality-preservation-validator.py validate --format summary
          if [ $? -ne 0 ]; then
            echo "Breaking changes detected. Please review."
            exit 1
          fi
```

### Pre-commit Hook

The validator automatically creates a pre-commit hook when you run:
```bash
python scripts/agents/functionality-preservation-validator.py setup-hooks
```

This hook will:
1. Run validation on all staged changes
2. Block commits if breaking changes are detected
3. Show a summary of issues found
4. Provide commands for detailed analysis

### IDE Integration

For VS Code, add to your tasks.json:
```json
{
  "label": "Validate Functionality",
  "type": "shell",
  "command": "python",
  "args": [
    "scripts/agents/functionality-preservation-validator.py",
    "validate",
    "--format",
    "detailed"
  ],
  "group": "test",
  "presentation": {
    "echo": true,
    "reveal": "always",
    "focus": false,
    "panel": "shared"
  }
}
```

## Report Structure

The validator generates comprehensive reports in JSON format:

```json
{
  "timestamp": "2025-08-03T10:30:00Z",
  "summary": {
    "total_checks": 25,
    "passed": 20,
    "warnings": 3,
    "failures": 2,
    "breaking_changes": 1
  },
  "results": [
    {
      "status": "fail",
      "category": "function_removed",
      "message": "Function 'calculate_tax' was removed",
      "file_path": "src/billing.py",
      "line_number": 45,
      "breaking_change": true,
      "suggestion": "Consider deprecating instead of removing, or ensure no code depends on this function"
    }
  ],
  "recommendations": [
    "❌ CRITICAL: Breaking changes detected. Review all failures before committing.",
    "Consider using feature flags or deprecation warnings for gradual migration."
  ]
}
```

## Testing the Validator

Run the included test suite to verify the validator works correctly:

```bash
python scripts/agents/test-functionality-validator.py
```

This will:
1. Create a temporary git repository
2. Test various scenarios (breaking changes, safe changes, etc.)
3. Verify the validator correctly identifies issues
4. Clean up test files

## Best Practices

### 1. Run Early and Often
- Validate changes frequently during development
- Don't wait until commit time to discover breaking changes

### 2. Use Feature Flags
- For major changes, use feature flags to gradually roll out functionality
- This allows for safer deployments and easier rollbacks

### 3. Deprecation Strategy
- Instead of removing functions immediately, mark them as deprecated
- Provide migration paths for consumers

### 4. Test Coverage
- Ensure comprehensive test coverage before relying on test regression detection
- Write tests for both positive and negative scenarios

### 5. Documentation
- Update API documentation when making changes
- Include migration guides for breaking changes

## Troubleshooting

### Common Issues

**"Git diff failed"**
- Ensure you're in a git repository
- Check that the base branch exists
- Verify git is installed and accessible

**"Could not parse file"**
- Check for syntax errors in Python files
- Ensure file encoding is UTF-8
- Verify file permissions

**"Tests not found"**
- Ensure pytest is installed
- Check test directory configuration
- Verify test files follow naming conventions

**"Validation timeout"**
- Large codebases may take time to analyze
- Consider using `--files` to analyze specific files
- Increase timeout in configuration

### Debug Mode

Enable verbose logging:
```bash
export PYTHONPATH=.
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python scripts/agents/functionality-preservation-validator.py validate
```

## Advanced Features

### Custom Rules

Extend the validator by modifying the configuration:

```yaml
rules:
  custom:
    check_docstring_changes: true
    require_type_hints: true
    check_complexity_increase: true
```

### Machine Learning Integration

The validator supports ML-based suggestions (when enabled):

```yaml
advanced:
  ml_suggestions:
    enabled: true
    confidence_threshold: 0.8
```

### Dependency Graph Visualization

Generate dependency graphs:
```bash
python scripts/agents/functionality-preservation-validator.py graph --output dependency-graph.json
```

## Contributing

To extend the validator:

1. **Add new validation rules** in the `FunctionalityPreservationValidator` class
2. **Extend AST analysis** for additional language constructs
3. **Add new output formats** in the reporting section
4. **Create new integrations** for different CI/CD systems

## License

This functionality preservation validator is part of the SutazaiApp project and follows the same licensing terms.

---

**Remember**: The goal is not to prevent all changes, but to ensure that changes are intentional, documented, and don't break existing functionality. The validator helps maintain the stability and reliability that Rule 2 demands.