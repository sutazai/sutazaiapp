# Data Management

- Schemas: UUID PKs; FK indexed; timestamps default now.
- Migrations: Alembic-based, idempotent init; separate bootstrap for extensions.
- Retention: logs 30d; embeddings persisted; PII minimized.
- Backups: Postgres nightly (RPO 24h), restore tested (RTO 2h).
- PII: encrypted at rest (volume-level) and in transit (TLS within network where applicable).

Footnotes: [`DATABASE_SCHEMA.sql` 1–80], [`init_db.sql` 1–20]