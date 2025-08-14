Title: Secrets and Environment Configuration Policy

Overview
- Zero hardcoded secrets: never commit credentials or tokens.
- Use `.env.example` for placeholders only; do not populate with real values.
- Load secrets at runtime from your secrets manager or secure CI/CD variables.

Required Practices
- Local development: create a private `.env.local` (gitignored) based on `.env.example`.
- Containers: inject secrets via `docker-compose`/orchestrator env or secrets mounts.
- Production: use Vault/Key Management Service; rotate regularly; audit access.
- Remove generated artifacts (coverage, logs, caches) before committing.

Quick Start
1) Copy `.env.example` to `.env.local` and fill local-only values.
2) Do not add `.env*` files to Git. `.gitignore` already enforces this.
3) For CI, define required variables in the pipeline/runner secret store.

Notes
- If any secret is found in the repo, remove the file and rotate the secret.
- Use consistent variable names across services to simplify deployment.
