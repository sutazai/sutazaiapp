# Repository Guidelines

This guide is authoritative and derived only from IMPORTANT/*. Treat other files as provisional.

## Project Structure & Modules
- `backend/`: API and service layer (FastAPI). Tests belong in `backend/tests`.
- `frontend/`: Streamlit UI. Source under `frontend/src` when present.
- `agents/`: AI agent implementations and orchestrators.
- `config/`, `docker/`, `scripts/`: Configuration, container definitions, and ops scripts.
- `IMPORTANT/`: Single source of truth for standards and architecture.

## Build, Test, and Run
- Containers: Use Docker Compose as the canonical interface. Start the stack with the root compose file; verify via health checks below.
- Health checks (examples):
  - `curl -f http://localhost:10010/health` (Backend)
  - `curl -f http://localhost:10104/` (Ollama tinyllama)
  - `curl -f http://localhost:10006/v1/status/leader` (Consul)
- Python tests (backend): `pytest -v backend/tests --cov=backend --cov-fail-under=80`.
- Lint/format (Python): Black, Flake8, mypy. If a Makefile exists, prefer `make lint`, `make test`.

## Coding Style & Naming
- Indentation: 4 spaces (Python), 2 spaces (JS/TS). Max line length 120.
- Naming: snake_case for Python functions/modules; PascalCase for classes; UPPER_SNAKE_CASE constants.
- Enforce formatting and imports; no commented‑out or placeholder code in main branches.

## Testing Guidelines
- Frameworks: pytest (backend), Playwright/Cypress or Newman for E2E where applicable.
- Coverage: ≥ 80% required for merges.
- Test layout: files `test_*.py` (backend); keep unit/integration markers explicit.

## Commit & PR Guidelines
- Commits: Conventional Commits format, one logical change per commit, reference issue IDs.
- PR requirements: description of change, risk/rollback plan, tests added/updated, CHANGELOG entry. Peer review is mandatory.

## Security & Configuration
- Secrets: never commit. Use environment variables; production secrets must come from the runtime.
- Network/Ports: 10000–10999 core infra; 11000–11999 agents. Verify only Ollama `tinyllama` is loaded unless changed intentionally.
- Minimum controls: input validation, security headers, rate limiting at gateway, and regular dependency scans.

