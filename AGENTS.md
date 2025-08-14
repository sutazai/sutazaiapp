# Repository Guidelines

## Mandatory Enforcement Rules (Read First)
Before any change or execution, you MUST read and follow: `/opt/sutazaiapp/IMPORTANT/Enforcement_Rules`. It supersedes this guide and all other docs.

## Project Structure & Module Organization
- `backend/` FastAPI API (port 10010), DB access, business logic.
- `frontend/` Streamlit UI (port 10011).
- `agents/` Agent services and orchestration.
- `src/` Shared Python utilities/libraries.
- `tests/` Unit, integration, e2e, performance, security.
- `scripts/` Dev/test/ops utils (MCP under `scripts/mcp/`).
- `config/`, `docker/`, `monitoring/` Config, compose files, observability.

## Build, Test, and Development Commands
- `make docker-up` / `make docker-down`: Start/stop the stack.
- `make health`: Quick checks (backend, frontend, ollama).
- `make test` / `make test-all`: Run core/all test suites.
- `make lint` / `make format`: Lint and auto-format (black, isort, flake8, mypy).
- Node tests: `npm test`, `npm run test:e2e`, `npm run docker:up`.
Example: `curl http://localhost:10010/health` (backend health).

## Coding Style & Naming Conventions
- Python 3.12, 4-space indents, type hints required in new/modified code.
- Tools: black (line length 88), isort, flake8, mypy (backend focus).
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `ALL_CAPS`.
- Reuse utilities in `src/`; avoid duplication (single source of truth).

## Testing Guidelines
- Framework: pytest with markers: `-m unit|integration|e2e|performance|security`.
- Location: `tests/` with files named `test_*.py`.
- Coverage: target ≥80%. Commands: `make coverage`, `make coverage-report`.
- Integration tests may require services: run `make docker-up` first.

## Commit & Pull Request Guidelines
- Commits: concise, imperative; version-tag style OK (e.g., `v89: Improve agent docs`).
- Link issues (e.g., `Fixes #123`) and note rationale for significant changes.
- PRs must include summary, screenshots for UI changes, test plan (commands + results), and any schema/API impact.
- CI gates: tests green (`make test`), linting clean (`make lint`).

## Security & Configuration Tips
- Never commit secrets. Start with `cp .env.example .env` and set local values.
- Prefer `make docker-up` on the isolated Docker network.
- MCP config: `.mcp.json`; validate with `scripts/mcp/selfcheck_all.sh`.
- For GitHub MCP, set `GITHUB_PERSONAL_ACCESS_TOKEN` in environment.

## Architecture Overview (Brief)
Microservices: FastAPI backend (10010), Streamlit frontend (10011), agents, and data stores: PostgreSQL (10000), Redis (10001), Neo4j (10002/10003), vector DBs (Qdrant/Chroma/FAISS). Ollama (10104) serves the default TinyLlama model.

