# Playwright E2E Tests for MCP Servers

- Component: Testing (Playwright)
- Maintainer: DevOps Automation (Claude Code)
- Scope: E2E validation for MCP registrations & health via CLI/containers

## Overview

This suite validates MCP server integration using Playwright’s test runner executed in a Docker container. Tests are designed to:
- Exercise MCP registration via `claude mcp list` (non-destructive).
- Be configuration-driven via environment variables.
- Produce HTML reports under `reports/playwright-mcp/html` (mounted volume).

Project-specific config file: `tests/playwright/playwright.mcp.config.ts`

## Configuration

Environment variables:
- `PLAYWRIGHT_MCP_CONTEXTS` (default: `context7`) — comma-separated contexts to assert as present.

Examples:
- Host run: `PLAYWRIGHT_MCP_CONTEXTS=context7,sequentialthinking npx playwright test -c tests/playwright/playwright.mcp.config.ts`
- Docker run (compose service): `docker compose -f docker/docker-compose.mcp.yml run --rm playwright-mcp-tests`

Reports:
- HTML: `reports/playwright-mcp/html`

## Docker Image (Multi-stage)

- Dockerfile: `tests/playwright/Dockerfile`
- Base: `node:20.16.0-alpine3.19` (version-pinned)
- Contains only what’s required to run CLI-oriented tests (no browsers).

Build:
- `docker build -t sutazai/mcp-playwright-tests -f tests/playwright/Dockerfile .`

## Orchestration Script

- Script: `scripts/testing/run_playwright_tests.sh`
- Starts `mcp-sequentialthinking` (stdio MCP server) and runs Playwright tests.
- Options:
  - `--keep-up` — keep containers running after tests
  - `--compose <file>` — override compose file path
  - `--contexts <list>` — override `PLAYWRIGHT_MCP_CONTEXTS`
  - `--timeout <seconds>` — service startup wait timeout

Example:
```
./scripts/testing/run_playwright_tests.sh --contexts "context7,sequentialthinking"
```

## Troubleshooting

- `Claude CLI not available`: Tests automatically skip CLI assertions if `claude` is not present in the runtime container. Run tests on host or bake the CLI into a derivative image if needed.
- `sequentialthinking` image missing: Build it before running: `docker build -t mcp/sequentialthinking -f servers/src/sequentialthinking/Dockerfile .`
- Permission errors: Ensure the Docker user has access to the project workspace and the Claude config directory if you run tests on the host.

