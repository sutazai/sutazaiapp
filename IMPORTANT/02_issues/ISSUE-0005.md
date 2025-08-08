# ISSUE-0005: Missing Authentication/Authorization

- Impacted: Backend API, Frontend, Security posture
- Options:
  - A: JWT Auth with RBAC and scopes (recommended)
  - B: API key per service only (weak)
  - C: Mutual TLS and service tokens (advanced)
- Recommendation: A
- Consequences: Introduce user/session tables, middleware, secrets management
- Sources: `/opt/sutazaiapp/IMPORTANT/SUTAZAI_PRD.md`
