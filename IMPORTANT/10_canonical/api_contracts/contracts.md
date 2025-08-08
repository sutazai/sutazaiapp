# API & Integration Contracts

- Versioning: `/api/v1` with backward-compatible changes; deprecations announced 2 releases ahead.
- Pagination: cursor-based with `limit`, `cursor`.
- Idempotency: `Idempotency-Key` header for POST endpoints.
- Errors: JSON `{code, message, details}`; 4xx client, 5xx server.
- Auth: Bearer JWT; service-to-service tokens; RBAC scopes.
- Endpoints (illustrative):
  - POST `/api/v1/generate`
  - POST `/api/v1/documents`
  - GET `/api/v1/query`
  - POST `/api/v1/agents/execute`
  - GET `/api/v1/agents/list`
- Integrations: Ollama `/api/generate`, `/api/embeddings`; ChromaDB collections CRUD.

Footnotes: [PRD `/workspace/IMPORTANT/SUTAZAI_PRD.md` 1–60], [Blueprint `/workspace/IMPORTANT/SUTAZAI_SYSTEM_ARCHITECTURE_BLUEPRINT.md` 1–60]