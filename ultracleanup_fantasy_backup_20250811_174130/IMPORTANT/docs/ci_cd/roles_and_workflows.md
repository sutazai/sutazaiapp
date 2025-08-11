# Team Roles, Branching Strategy, and CI/CD Workflows

This governance framework defines roles, responsibilities, branch naming, and workflow requirements for Perfect Jarvis. It reflects current repository reality and uses production‑ready tooling.

## Roles & Responsibilities

- Frontend: Maintain UI (Streamlit in this repo). Coordinate API contracts, accessibility, and user flows.
- Backend: Own Python microservices under `services/` and `backend/`. Implement endpoints, health/metrics, and integration logic.
- DevOps: Own `scripts/`, deployment automation, CI/CD integration, monitoring, and infra health verification.
- QA: Own automated testing plans and gates; integrate smoke tests and regression suites with CI.
- Documentation: Maintain `docs/` and CHANGELOG; ensure diagrams and READMEs are current and non‑speculative.

## Communication & Escalation

- Daily async status on CI, open failures, and blockers.
- Escalation path: Service Owner → Area Lead → Project Supervisor.

## Branching & PR Requirements

- Branch prefixes: `feature/`, `bugfix/`, `hotfix/`, `release/`.
- Conventional commits enforced for atomic changes (e.g., `feat:`, `fix:`, `docs:`).
- PRs require: at least 1 reviewer, CI green, and updated CHANGELOG entries.
- Feature flags: Where applicable, guard incomplete backend features via env flags; avoid shipping inactive, unreachable endpoints.

## CI/CD Workflows (Overview)

- Lint/Format: run configured linters/formatters (e.g., Flake8/Black if present) on changed Python files.
- Tests: run unit/integration tests (`pytest`) with reports.
- Security: run static analysis and dependency vulnerability scans (where configured in repo).
- Infra Health: run `scripts/devops/check_services_health.py` against target env during pre‑merge or pre‑deploy stages.

See pipeline snippets in this folder for GitLab and GitHub examples.

