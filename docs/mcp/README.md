# MCP Servers: Configuration and Usage

- Component: MCP Servers and Claude CLI
- Maintainer: DevOps Automation (Claude Code)
- Scope: Context registration, shell helpers, Docker orchestration

## Overview

This document centralizes the configuration and operation of all MCP servers used by SutazAI, including:
- `context7` (Upstash Context7 via `npx -y @upstash/context7-mcp`)
- `sequentialthinking` (local implementation under `servers/src/sequentialthinking`)

All steps are idempotent, follow COMPREHENSIVE_ENGINEERING_STANDARDS, and avoid duplication.

## Batch Registration (Idempotent)

- Script: `scripts/mcp/register_mcp_contexts.sh`
- Behavior: Adds missing MCP contexts using correct startup commands, no duplicates.
- Default scope: `local` (project-scoped). Override with `--scope user|project`.

Examples:
- Ensure contexts (project): `./scripts/mcp/register_mcp_contexts.sh`
- Dry run: `./scripts/mcp/register_mcp_contexts.sh --dry-run`
- User scope: `./scripts/mcp/register_mcp_contexts.sh --scope user`

Registered contexts:
- `context7`: `claude mcp add context7 npx -y @upstash/context7-mcp`
- `sequentialthinking`: `claude mcp add sequentialthinking docker run --rm -i mcp/sequentialthinking`

Note: For `sequentialthinking`, build the image first:
- `docker build -t mcp/sequentialthinking -f servers/src/sequentialthinking/Dockerfile .`

## Shell Helpers (Persistent)

- File: `scripts/shell/claude_mcp_aliases.sh`
- Provides helpers: `mcp:ls`, `mcp:get <name>`, `mcp:use <context7|seq>`
- Install persistently by sourcing from your shell profile:
  - Add to `/root/.bashrc`: `source /opt/sutazaiapp/scripts/shell/claude_mcp_aliases.sh`

Usage:
- `mcp:ls` → List configured MCP servers
- `mcp:get context7` → Show details
- `mcp:use context7` → Ensure `context7` is present (project scope)
- `mcp:use seq` → Ensure `sequentialthinking` is present (project scope)

## Docker Compose Orchestration

- File: `docker/docker-compose.mcp.yml`
- Network: `sutazai_mcp_net`
- Services:
  - `mcp-sequentialthinking` → builds `mcp/sequentialthinking` from `servers/src/sequentialthinking/Dockerfile`
  - `playwright-mcp-tests` → runs Playwright tests (`tests/playwright`)

Start services and run tests with the orchestration script (see testing doc), or:
- `docker compose -f docker/docker-compose.mcp.yml up -d mcp-sequentialthinking`
- `docker compose -f docker/docker-compose.mcp.yml run --rm playwright-mcp-tests`

## Troubleshooting

- `context7` missing: Re-run `./scripts/mcp/register_mcp_contexts.sh`; ensure internet when using `npx` at runtime.
- `sequentialthinking` missing: Build image locally before registering.
- Permission errors with `claude`: Ensure the CLI can access `~/.claude.json`. Run commands as the same user who uses Claude.
- Avoid duplicates: Use the provided registration script—never hand-edit MCP configs.

