# Fantasy Elements Validator

## Overview

The Fantasy Elements Validator is a comprehensive enforcement agent that scans all code for forbidden terms, validates dependencies are real and verifiable, checks for speculative or placeholder code, and integrates with pre-commit hooks to enforce CLAUDE.md Rule 1: "No Fantasy Elements".

## Purpose

This tool ensures that all code follows production-ready, concrete implementations without:
- Fantasy terminology (specific implementation name (e.g., emailSender, dataProcessor), assistant, helper, processor, manager, transfer, send, transmit, copy, etc.)
- Speculative or placeholder code
- Unverifiable dependencies
- Hypothetical implementations

## Features

### 1. Forbidden Terms Detection
Scans for and categorizes forbidden terms across multiple categories:

**High Severity Terms:**
- `specific implementation name (e.g., emailSender, dataProcessor)`, `magical`, `magically`, `magicHandler`, `magicService`, `magicFunction`
- `assistant, helper, processor, manager`, `wizardService`, `wizardry`, `wizardHandler`
- `transfer, send, transmit, copy`, `teleportData`, `teleportation`, `teleporting`

**Medium Severity Terms:**
- `external service, third-party API, opaque system`, `blackbox`, `black_box`, `blackBox`
- `specific future version or roadmap item`, `maybe`, `could work`, `might work`, `theoretical`, `concrete implementation or real example`

### 2. Placeholder Code Detection
Identifies patterns that indicate speculative or incomplete implementations:
- TODO comments with fantasy references
- Stub implementations
- Mock data functions
- Temporary fixes
- Placeholder functions
- Hypothetical implementations

### 3. Dependency Validation
Validates that all dependencies are real and exist:
- **Python**: PyPI package validation
- **Node.js**: npm registry validation
- **Rust**: crates.io validation
- **Go**: Basic module name validation

### 4. Auto-Fix Capabilities
Provides intelligent suggestions and can automatically fix many violations:
- Replaces fantasy terms with concrete alternatives
- Suggests better naming conventions
- Provides context-aware fixes

### 5. Pre-commit Integration
Automatically blocks commits that contain fantasy elements:
- Creates Git pre-commit hooks
- Integrates with CI/CD pipelines
- Provides clear error messages and fix suggestions

## Installation & Setup

### Prerequisites
- Python 3.8+
- ripgrep (`rg` command)
- Required Python packages (from requirements.txt):
  - `requests`
  - `rich`
  - `pyyaml`
  - `tomli` (for TOML parsing)

### Installation
```bash
# Ensure ripgrep is installed
sudo apt install ripgrep  # Ubuntu/Debian
brew install ripgrep      # macOS

# The script is already executable
chmod +x /opt/sutazaiapp/scripts/agents/fantasy-elements-validator.py
```

## Usage

### Basic Scanning
```bash
# Scan entire codebase for fantasy elements
python3 scripts/agents/fantasy-elements-validator.py

# Scan specific directory
python3 scripts/agents/fantasy-elements-validator.py --root-path /path/to/scan

# Generate detailed report
python3 scripts/agents/fantasy-elements-validator.py --output my-report.json
```

### Auto-Fix Mode
```bash
# Automatically fix violations where possible
python3 scripts/agents/fantasy-elements-validator.py --fix

# Fix specific directory
python3 scripts/agents/fantasy-elements-validator.py --root-path /path/to/fix --fix
```

### Pre-commit Integration
```bash
# Create pre-commit hook
python3 scripts/agents/fantasy-elements-validator.py --create-hook

# Test pre-commit mode (used by hook)
python3 scripts/agents/fantasy-elements-validator.py --pre-commit
```

### CI/CD Integration
```bash
# Use in CI pipeline (exits with error code if violations found)
python3 scripts/agents/fantasy-elements-validator.py --pre-commit
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--fix` | Apply automatic fixes where possible |
| `--pre-commit` | Run in pre-commit mode (minimal output, exit codes for CI) |
| `--report-only` | Generate report without scanning (requires existing report.json) |
| `--output OUTPUT` | Output file for detailed report (default: fantasy-elements-report.json) |
| `--root-path PATH` | Root path to scan (default: /opt/sutazaiapp) |
| `--create-hook` | Create or update pre-commit hook |

## Output & Reports

### Console Output
The validator provides rich console output with:
- Color-coded severity levels
- Detailed violation descriptions
- Auto-fix suggestions
- Summary statistics
- Progress indicators

### JSON Reports
Detailed JSON reports include:
- Violation details with file paths and line numbers
- Categorized statistics
- Dependency validation results
- Recommendations for remediation
- Timestamp and metadata

### Example Report Structure
```json
{
  "timestamp": "2025-08-03T12:00:00",
  "total_files_scanned": 1543,
  "violations_found": 25,
  "violations_by_type": {
    "forbidden_term": 18,
    "placeholder_code": 7
  },
  "violations_by_severity": {
    "high": 12,
    "medium": 13
  },
  "violations": [...],
  "dependency_issues": [...],
  "recommendations": [...]
}
```

## Integration Examples

### Git Pre-commit Hook
The validator automatically creates a pre-commit hook that:
1. Runs validation on all staged files
2. Blocks commits if violations are found
3. Provides clear fix instructions
4. Suggests using `--fix` flag for automatic remediation

### CI/CD Pipeline Integration
```yaml
# GitHub Actions example
- name: Validate Fantasy Elements
  run: |
    python3 scripts/agents/fantasy-elements-validator.py --pre-commit
  continue-on-error: false
```

### IDE Integration
The validator can be integrated with IDEs through:
- External tools configuration
- Custom build steps
- File watcher integration

## Configuration

### File Patterns
The validator scans these file types by default:
- Python: `**/*.py`
- JavaScript/TypeScript: `**/*.js`, `**/*.ts`, `**/*.jsx`, `**/*.tsx`
- Go: `**/*.go`
- Rust: `**/*.rs`
- C/C++: `**/*.c`, `**/*.cpp`, `**/*.h`, `**/*.hpp`
- Java: `**/*.java`
- Configuration: `**/*.yml`, `**/*.yaml`, `**/*.json`, `**/*.toml`
- Documentation: `**/*.md`
- Docker: `**/Dockerfile*`
- Shell: `**/*.sh`, `**/*.bash`

### Exclusion Patterns
These directories are automatically excluded:
- `.git/*`, `node_modules/*`, `__pycache__/*`
- `venv/*`, `env/*`, `.venv/*`, `.env/*`
- `build/*`, `dist/*`, `.pytest_cache/*`
- `logs/*`, `data/*`, `archive/*`, `backup*`

## Customization

### Adding New Forbidden Terms
Edit the `FORBIDDEN_TERMS` dictionary in the script to add new categories or terms:

```python
FORBIDDEN_TERMS = {
    'custom_category': {
        'terms': ['custom_term1', 'custom_term2'],
        'severity': 'high',
        'suggestions': {
            'custom_term1': 'better_alternative'
        }
    }
}
```

### Custom Placeholder Patterns
Add new regex patterns to detect additional placeholder code:

```python
placeholder_patterns = [
    (r'your_pattern_here', 'Description of what it catches'),
    # ... existing patterns
]
```

## Best Practices

### 1. Regular Scanning
- Run validation before every commit
- Include in CI/CD pipeline
- Schedule periodic full codebase scans

### 2. Team Training
- Educate team on forbidden terms
- Provide concrete alternatives
- Share validation reports

### 3. Progressive Cleanup
- Use `--fix` for automatic improvements
- Manually review and improve suggested fixes
- Address high-severity violations first

### 4. Documentation Updates
- Update documentation when fixing violations
- Ensure examples use concrete terms
- Keep naming conventions consistent

## Troubleshooting

### Common Issues

**1. "ripgrep not found"**
```bash
sudo apt install ripgrep  # Ubuntu/Debian
brew install ripgrep      # macOS
```

**2. "Permission denied" on pre-commit hook**
```bash
chmod +x .git/hooks/pre-commit
```

**3. "Too many violations to fix automatically"**
- Run with `--fix` multiple times
- Manually review complex cases
- Use smaller scope (`--root-path`)

**4. "Dependency validation failed"**
- Check network connectivity
- Verify package names are correct
- Update package registry URLs if needed

### Performance Optimization

For large codebases:
1. Use specific `--root-path` to limit scope
2. Exclude unnecessary directories
3. Run in parallel on different subdirectories
4. Cache validation results for unchanged files

## Integration with CLAUDE.md Rules

This validator specifically enforces **Rule 1: No Fantasy Elements** by:

1. **Preventing speculative code**: Blocks terms like "specific future version or roadmap item", "maybe", "theoretical"
2. **Enforcing concrete implementations**: Replaces vague terms with specific alternatives
3. **Validating real dependencies**: Ensures all packages exist and are available
4. **Blocking hypothetical designs**: Prevents abstract or concrete implementation or real example system references
5. **Promoting production-ready code**: Encourages tested, validated implementations

## Contributing

To contribute improvements to the Fantasy Elements Validator:

1. Test changes on diverse codebases
2. Add new forbidden term categories as needed
3. Improve auto-fix suggestions
4. Enhance dependency validation coverage
5. Add support for new file types or languages

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review command line options with `--help`
3. Examine generated JSON reports for details
4. Test with minimal examples to isolate problems

---

*This validator is part of the SUTAZAI codebase hygiene enforcement system, ensuring production-ready, maintainable code across all projects.*