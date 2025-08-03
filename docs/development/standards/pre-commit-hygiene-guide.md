# Pre-commit Hygiene Enforcement Guide

## Overview

This guide explains how the pre-commit hooks enforce all CLAUDE.md rules automatically before code can be committed to the repository. The system provides comprehensive hygiene enforcement with clear error messages and bypass options for emergencies.

## Table of Contents

1. [Installation](#installation)
2. [How It Works](#how-it-works)
3. [Hook Phases](#hook-phases)
4. [Using the Hooks](#using-the-hooks)
5. [Troubleshooting](#troubleshooting)
6. [Emergency Bypass](#emergency-bypass)
7. [Hook Reference](#hook-reference)

## Installation

### Quick Install
```bash
cd /opt/sutazaiapp
./scripts/install-hygiene-hooks.sh
```

### Manual Install
```bash
pip3 install pre-commit
pre-commit install
```

### Verify Installation
```bash
pre-commit --version
pre-commit run --all-files  # Test run on all files
```

## How It Works

The pre-commit framework runs a series of checks before allowing commits:

1. **File Analysis**: Examines staged files for rule violations
2. **System Checks**: Validates overall codebase health
3. **Reporting**: Provides clear error messages with fix instructions
4. **Enforcement**: Blocks commits if critical violations are found

### Execution Flow
```
git commit
    ↓
Pre-commit activated
    ↓
Run hooks in order:
  - Standard checks (syntax, formatting)
  - Phase 1: Critical violations
  - Phase 2: Structural violations  
  - Phase 3: Organizational violations
  - Master orchestration
    ↓
All pass? → Commit proceeds
Any fail? → Commit blocked with instructions
```

## Hook Phases

### Phase 1: Critical Violations (Must Fix)
These hooks enforce rules that, if violated, could break the system:

- **Rule 13**: No garbage/backup files
- **Rule 12**: Single deployment script
- **Rule 9**: No duplicate directories

### Phase 2: Structural Violations (Important)
These hooks ensure code quality and maintainability:

- **Rule 11**: Docker structure validation
- **Rule 8**: Python documentation requirements
- **Rule 1**: No fantasy/placeholder code
- **Rule 2**: Breaking change detection
- **Rule 3**: System analysis

### Phase 3: Organizational Violations (Best Practices)
These hooks maintain project organization:

- **Rule 7**: Script organization
- **Rule 6**: Documentation structure
- **Rule 15**: Documentation deduplication
- **Rule 4**: Script reuse enforcement
- **Rule 10**: Safe deletion verification
- **Rule 14**: Correct agent usage
- **Rule 16**: Ollama/TinyLlama compliance

## Using the Hooks

### Normal Workflow
```bash
# Make changes
vim script.py

# Stage changes
git add script.py

# Commit - hooks run automatically
git commit -m "Add new feature"
```

### Manual Hook Execution
```bash
# Run on staged files only
pre-commit run

# Run on specific files
pre-commit run --files script.py

# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run python-documentation
```

### Understanding Output

#### Success Output
```
Rule 8: Python documentation.............................Passed
Rule 13: No garbage files................................Passed
✅ All hygiene checks passed
```

#### Failure Output
```
Rule 8: Python documentation.............................Failed
- hook id: python-documentation
- files were modified by this hook

❌ Rule 8 violations in scripts/example.py:
  Line 1: Missing module-level docstring
    Fix: Add docstring with Purpose, Usage, Requirements
```

## Troubleshooting

### Common Issues

#### 1. Hook Command Not Found
```bash
# Fix: Install dependencies
pip3 install pre-commit pyyaml gitpython detect-secrets
```

#### 2. Hooks Running Slowly
```bash
# Check which hooks are slow
time pre-commit run --verbose

# Run only on changed files (default behavior)
pre-commit run
```

#### 3. False Positives
If a hook incorrectly flags valid code:

1. Check if you're following the correct pattern
2. Review the specific rule in CLAUDE.md
3. Use targeted bypass if legitimate exception

#### 4. Hook Not Running
```bash
# Reinstall hooks
pre-commit install --force

# Check git hooks directory
ls -la .git/hooks/
```

### Debugging Hooks

```bash
# Verbose output
pre-commit run --verbose --show-diff-on-failure

# Test specific hook with debug info
python scripts/pre-commit/check-python-docs.py myfile.py
```

## Emergency Bypass

### When to Use Bypass

Only bypass hooks when:
- Production hotfix required
- False positive blocking critical work
- Temporary exception with follow-up ticket

### Bypass Methods

#### 1. Skip Specific Hook (Preferred)
```bash
# Skip one hook
SKIP=no-garbage-files git commit -m "Hotfix: keeping backup for rollback"

# Skip multiple hooks
SKIP=no-garbage-files,python-documentation git commit -m "Emergency fix"
```

#### 2. Skip All Hooks (Use Sparingly)
```bash
git commit --no-verify -m "Emergency: [detailed justification]"
```

#### 3. Bypass Logging
All bypasses are automatically logged to `.git/hooks/bypass.log`:
```
[2025-08-03 14:30:00] BYPASS: User=developer Skip=no-garbage-files Message='Hotfix: keeping backup'
```

### Post-Bypass Process

1. Create follow-up ticket to address skipped checks
2. Review bypass log in team meetings
3. Fix violations in next sprint

## Hook Reference

### Standard Hooks

| Hook ID | Description | Rule |
|---------|-------------|------|
| trailing-whitespace | Remove trailing whitespace | Hygiene |
| end-of-file-fixer | Ensure files end with newline | Hygiene |
| check-yaml | Validate YAML syntax | Config |
| check-json | Validate JSON syntax | Config |
| black | Format Python code | Rule 8 |
| ruff | Lint Python code | Rule 8 |
| shellcheck | Validate shell scripts | Rule 7 |
| detect-secrets | Find hardcoded secrets | Security |

### Custom Hygiene Hooks

| Hook ID | Description | Rule | Phase |
|---------|-------------|------|-------|
| no-garbage-files | Block backup/temp files | 13 | 1 |
| single-deployment-script | Enforce one deploy.sh | 12 | 1 |
| no-duplicate-directories | Prevent dir duplication | 9 | 1 |
| docker-structure | Validate Docker files | 11 | 2 |
| python-documentation | Require Python docs | 8 | 2 |
| no-fantasy-elements | Block placeholder code | 1 | 2 |
| no-breaking-changes | Detect breaking changes | 2 | 2 |
| system-analysis | Quick health check | 3 | 2 |
| script-organization | Validate script layout | 7 | 3 |
| documentation-structure | Check doc organization | 6 | 3 |
| safe-deletion | Verify safe deletions | 10 | 3 |

### Master Orchestration Hooks

| Hook ID | Description | Runs |
|---------|-------------|------|
| hygiene-coordinator | Phase-based enforcement | Always |
| agent-orchestrator | AI agent coordination | Always |
| enforce-all-claude-rules | Final comprehensive check | Always |

## Best Practices

1. **Run hooks before pushing**: `pre-commit run --all-files`
2. **Fix issues immediately**: Don't accumulate violations
3. **Update hooks regularly**: `pre-commit autoupdate`
4. **Document bypasses**: Always explain why in commit message
5. **Review hook failures**: They often catch real issues

## Configuration

### Customizing Hooks

Edit `.pre-commit-config.yaml` to:
- Add new hooks
- Modify hook arguments
- Change execution order
- Exclude certain files

Example:
```yaml
- id: python-documentation
  name: "Rule 8: Verify Python documentation"
  entry: python scripts/pre-commit/check-python-docs.py
  language: python
  types: [python]
  exclude: ^tests/  # Exclude test files
```

### Performance Optimization

```yaml
# Run only on changed files
pass_filenames: true

# Run always (slower)
always_run: true

# Fail fast on first error
fail_fast: true
```

## Integration with CI/CD

The same hooks can run in CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run pre-commit
  uses: pre-commit/action@v3.0.0
```

```bash
# GitLab CI example
hygiene-check:
  script:
    - pip install pre-commit
    - pre-commit run --all-files
```

## Support

- **Issues**: Create issue with hook output and file content
- **Updates**: Run `./scripts/install-hygiene-hooks.sh --force`
- **Disable temporarily**: `git config core.hooksPath /dev/null`
- **Re-enable**: `git config --unset core.hooksPath && pre-commit install`

Remember: These hooks exist to maintain code quality and prevent issues. Working with them, not against them, leads to a cleaner, more maintainable codebase.