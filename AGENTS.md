# Repository Guidelines

## Project Structure & Module Organization
- `backend/`: FastAPI service (`backend/app/main.py`), routers, models, `requirements.txt`.
- `frontend/`: Streamlit UI (`frontend/app.py`), `components/`, `pages/`, `utils/`.
- `src/`: JS integration and client stores (e.g., `src/store/voiceStore.js`).
- `tests/`: Pytest suites, integration/e2e, Playwright; shared config in `tests/pytest.ini`.
- `docker/`: Primary Compose files (use `docker/docker-compose.yml`).
- `IMPORTANT/docs/testing/`: Canonical Newman, Cypress, and k6 test assets.
- `scripts/`: Ops/deployment/testing helpers (see `scripts/testing/*`).

## Build, Test, and Development Commands
- Backend (Python): `python -m venv .venv && source .venv/bin/activate && pip install -r backend/requirements.txt`
- Run API locally: `uvicorn backend.app.main:app --host 0.0.0.0 --port 10010`
- Frontend (Streamlit): `streamlit run frontend/app.py --server.port 10011 --server.address 0.0.0.0`
- Docker (from repo root): `cd docker && docker network create sutazai-network || true && docker-compose up -d`
- JS tooling: `npm ci` then:
  - API tests (Newman): `node IMPORTANT/docs/testing/newman_ci_integration.js`
  - E2E: `npx cypress run --spec IMPORTANT/docs/testing/cypress_e2e_tests.js`
  - Unit (Jest): `npm run test:unit`

## Coding Style & Naming Conventions
- Python: PEP8, 4-space indent, snake_case files; prefer type hints. Recommended formatters: Black + isort (see `docs/pyproject.toml`).
- JavaScript: ES modules, camelCase for vars/functions, PascalCase for components; keep store names aligned with `src/store/*.js` patterns.
- Keep diffs minimal; run formatters locally before PRs (Ruff/Black for Python, Prettier for JS if configured).

## Testing Guidelines
- Pytest: use `pytest` from repo root (config in `tests/pytest.ini`); name tests `test_*.py`. Example: `pytest -m "not slow"`.
- Coverage: `.coveragerc` provided; recommended target ≥80%. Example: `coverage run -m pytest && coverage report`.
- JS tests: Jest uses `tests/**/*.test.js|*.spec.js`. Cypress base URL is `http://localhost:10011`.
- Health checks: Backend `GET /health` at `http://localhost:10010/health`.

## Commit & Pull Request Guidelines
- Commits: Prefer Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`). History commonly uses `chore: sync ...`; keep scopes clear.
- PRs: Provide summary, linked issues, test evidence (pytest/Jest/Cypress), and screenshots for UI. Ensure local tests pass and note coverage changes.

## Security & Configuration Tips
- Never commit secrets. Base local env on `.env.example`; production uses `.env.master` (symlinked by `.env.production`).
- Default ports (via Compose): Backend `10010` → `backend:8000`; Frontend `10011` → `frontend:8501`.
- Compose expects external network `sutazai-network`; create it once with `docker network create sutazai-network`.
