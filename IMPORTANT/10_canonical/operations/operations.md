# Operations

- CI/CD: Lint, tests, build, compose deploy; canary where feasible
- Environments: dev (local), staging, prod (local cluster)
- Feature Flags: config-driven toggles
- Rollbacks: versioned compose; DB migrations reversible
- Config Mgmt: `.env` templates; no secrets in repo