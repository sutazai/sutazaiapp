# Repository Guidelines

## Project Structure & Module Organization
- Root services: `backend/`, `frontend/`, `docker/`, `scripts/`, `docs/`, `tests/`.
- MCP wrappers: `scripts/mcp/wrappers/*.sh` (each server supports `--selfcheck`).
- Python MCP module: `mcp_ssh/` (`src/mcp_ssh/`, tests in `mcp_ssh/tests/`).
- Data/config: `.mcp.json`, `.env`, `config/`, `database/`, `docker-compose*` under `docker/`.

## Build, Test, and Development Commands
- Services (Docker): `npm run docker:up` | `npm run docker:down` | `npm run docker:logs`
- API tests (Postman/Newman): `npm run test:api`
- E2E tests (Cypress): `npm run test:e2e` (headless) | `npm run test:e2e:open`
- Health check: `npm run test:health`
- Full test sweep: `npm run test:all`
- Python (mcp_ssh) tests: from `mcp_ssh/` run `uv run pytest -q` (or `pytest`)
- Verify MCP servers locally: `scripts/mcp/wrappers/<name>.sh --selfcheck`

## Coding Style & Naming Conventions
- Python: Black (88 cols), isort (profile=black), Ruff, MyPy (strict) configured in `mcp_ssh/pyproject.toml`.
  - Prefer `snake_case` for functions/vars, `PascalCase` for classes.
- JS/TS: Jest/Cypress tests present; linting is minimal here—follow existing patterns; format with Prettier if configured in editor.
- Shell: Wrapper scripts live in `scripts/mcp/wrappers/` and are executable; name as `<server>.sh`.

## Testing Guidelines
- Python: Pytest with markers (`unit`, `integration`, `slow`); tests under `mcp_ssh/tests/` (patterns `test_*.py`).
  - Coverage configured for `mcp_ssh` (`tool.coverage.*` in `pyproject.toml`).
- Node: Jest unit tests (`tests/**/*.test.js|spec.js`), Cypress for E2E, Newman for API suites.
- Add targeted tests alongside the module you change and ensure `npm run test:health` passes before CI runs.

## Commit & Pull Request Guidelines
- Commit style: short, imperative subject; common types seen: `chore:`, version sync notes. Prefer Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`) where possible.
- PRs should include: concise description, motivation/links to issues, test evidence (logs/screenshots for Cypress), and any ops notes (migrations, env keys).

## Security & Configuration Tips
- Never commit secrets. Use `.env` (see `.env.example`) and Docker secrets.
- Core services: Postgres/Redis/Ollama via Docker; ensure network `sutazai-network` exists.
- Quick MCP validation: e.g., `scripts/mcp/wrappers/postgres.sh --selfcheck` or `puppeteer-mcp.sh --selfcheck`.

## Architecture Overview
- Control Plane: MCP layer via `scripts/mcp/wrappers/*` orchestrates tools (e.g., postgres, ssh, memory-bank, playwright) and exposes `--selfcheck` for diagnostics.
- App Plane: `backend/` (API/services) and `frontend/` (UI) interact with MCP tools and data services.
- Data Plane: Dockerized Postgres/Redis/Ollama connected on `sutazai-network` (see `docker/` compose files).
- Config: `.mcp.json` (MCP servers), `.env` (secrets), `config/` (runtime tuning).

Flow (simplified):
`Client → MCP Wrapper (scripts/mcp/wrappers/<name>.sh) → Target Service (e.g., Postgres via docker) → Results to client`
