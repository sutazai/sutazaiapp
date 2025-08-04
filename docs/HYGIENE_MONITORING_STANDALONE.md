# Standalone Hygiene Monitoring System

## Overview

The Standalone Hygiene Monitoring System is a lightweight, on-demand tool for checking codebase compliance with CLAUDE.md rules. It runs completely independently from the main Sutazai application with no port conflicts or interference.

## Key Features

- **Completely Standalone**: Uses separate network (172.25.0.0/16) and ports
- **On-Demand Execution**: Only runs when you need it
- **No Interference**: Won't conflict with main application services
- **Simple Web UI**: View reports at http://localhost:9080
- **Comprehensive Scanning**: Checks for forbidden patterns, structure violations, duplicates, and naming conventions

## Quick Start

```bash
# Run a hygiene scan and view results
./scripts/run-hygiene-check.sh

# Run full validation (scan + rule compliance check)
./scripts/run-hygiene-check.sh full

# Stop the report viewer
./scripts/run-hygiene-check.sh stop
```

## Architecture

The system consists of three lightweight containers:

1. **hygiene-scanner**: Scans codebase for violations
2. **hygiene-reporter**: Nginx server for viewing reports (port 9080)
3. **hygiene-validator**: Validates CLAUDE.md rule compliance

## Port Usage

- **9080**: Report viewer web interface (only port exposed)
- All other services run internally on the isolated network

## Commands

```bash
# Basic hygiene scan
./scripts/run-hygiene-check.sh scan

# Validate CLAUDE.md rules only
./scripts/run-hygiene-check.sh validate

# Full check (scan + validation)
./scripts/run-hygiene-check.sh full

# Just view existing reports
./scripts/run-hygiene-check.sh report-only

# Stop report viewer
./scripts/run-hygiene-check.sh stop

# Clean up everything
./scripts/run-hygiene-check.sh clean
```

## Reports

Reports are saved in `/opt/sutazaiapp/reports/` in both JSON and HTML formats:
- `hygiene-report.json`: Machine-readable results
- `hygiene-report.html`: Human-friendly web view
- `rule-validation.json`: CLAUDE.md compliance results

## What It Checks

### Forbidden Patterns
- Fantasy elements (magic, wizard, teleport, etc.)
- Garbage files (temp, old, backup, WIP files)
- Duplicate patterns (utils1, service2, etc.)

### Structure Compliance
- Required directories exist (scripts, docker, backend, frontend)
- Files are in correct locations
- No misplaced scripts or configs

### Naming Conventions
- Python files follow snake_case
- Scripts use lowercase with hyphens
- Dockerfiles follow standard naming

### Security
- No hardcoded passwords or API keys
- No exposed secrets in code

## Integration with CI/CD

The hygiene scanner returns appropriate exit codes:
- **0**: All checks passed
- **1**: Violations found

This makes it easy to integrate into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Hygiene Check
  run: |
    docker-compose -f docker-compose.hygiene-standalone.yml run --rm hygiene-scanner
```

## Customization

Edit `/opt/sutazaiapp/docker/hygiene-scanner/rule_definitions.yaml` to customize:
- Forbidden patterns
- Required directories
- Naming conventions
- File structure rules

## Troubleshooting

### Port 9080 already in use
```bash
# Find what's using port 9080
sudo lsof -i :9080

# Use a different port by editing docker-compose.hygiene-standalone.yml
```

### Can't access report viewer
- Check if container is running: `docker ps | grep hygiene-reporter`
- Ensure firewall allows port 9080
- Try `http://localhost:9080` directly

### Scanner finds too many false positives
- Update rule_definitions.yaml to exclude certain patterns
- Add directories to skip in hygiene_scanner.py

## Benefits

1. **Zero Impact**: Doesn't affect running services
2. **Fast**: Lightweight containers start quickly
3. **Portable**: Can run on any system with Docker
4. **Extensible**: Easy to add new rules and checks
5. **Clear Reports**: HTML reports are easy to share

## Future Enhancements

- [ ] Real-time monitoring mode
- [ ] Slack/email notifications
- [ ] Historical trend tracking
- [ ] Custom rule plugins
- [ ] IDE integration