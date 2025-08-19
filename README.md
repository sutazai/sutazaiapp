# SutazAI Application – Overview & Developer Guide

This repository hosts the SutazAI application: backend APIs, frontend UI, Dockerized infrastructure, MCP integrations, and monitoring tools. Use this README as a quick, accurate entry point; see `AGENTS.md` for contributor guidelines.

## Project Layout
- `backend/` (Python) – API and services
- `frontend/` (JS/React) – UI
- `docker/` – Dockerfiles and compose manifests
- `scripts/monitoring/` – Ops tools (e.g., `live_logs.sh`)
- `scripts/mcp/wrappers/` – MCP server wrappers; config in `.mcp.json`
- `tests/` – Jest/Cypress/API and shell tests
- `config/port-registry.yaml` – Canonical port assignments
- `IMPORTANT/INDEX.md` – Generated inventory (Docker/MCP/files)

## Quick Start
1) Prereqs: Docker, Docker Compose, Node 18+, Python 3.12+.  
2) Start core services: `npm run docker:up`  
3) Monitoring menu: `./scripts/monitoring/live_logs.sh` (Ctrl+C in log views returns to menu)  
4) Frontend: http://localhost:10011  
5) Health check: `npm run test:health`

## Testing
- API (Newman): `npm run test:api`  
- E2E (Cypress): `npm run test:e2e` (or `:open`)  
- Unit (Jest): `npm run test:unit`  
- Monitoring script tests:  
  - `LIVE_LOGS_NO_AUTORUN=true tests/monitoring/test_live_logs_basic.sh`  
  - `LIVE_LOGS_NONINTERACTIVE=true tests/monitoring/test_live_logs_handlers.sh`

## MCP & Mesh
- Define servers in `.mcp.json`; wrappers live in `scripts/mcp/wrappers/`.  
- Examples: `scripts/mcp/wrappers/mcp_ssh.sh --selfcheck`, `scripts/mcp/wrappers/playwright-mcp.sh --selfcheck`.  
- If registering with a mesh, align ports with `config/port-registry.yaml`.

## Troubleshooting
- Backend not responding: check container logs with `./scripts/monitoring/live_logs.sh` → option 2 or 10.  
- Docker issues: option 11 in `live_logs.sh` (Troubleshooting & Recovery).  
- Regenerate system index: `python3 scripts/tools/generate_index.py` → see `IMPORTANT/INDEX.md`.

## Security & Configuration
- Never commit secrets; use `.env` (see `.env.example`) and Docker secrets.  
- Keep ports/services synchronized between compose files, `.mcp.json`, and `config/port-registry.yaml`.

