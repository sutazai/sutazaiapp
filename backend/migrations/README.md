# Database Migrations (UUID-First)

This project adopts UUID primary keys for all tables. Until ORM models are consolidated, apply migrations via raw SQL.

- Prereqs
  - PostgreSQL extension: `pgcrypto` (for `gen_random_uuid()`)
  - Env: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`

- Initial schema
  - `uuid_init.sql` creates core tables using UUID PKs and indexed FKs

- Apply
  - `bash scripts/db/apply_uuid_schema.sh` (uses env vars)

- Next
  - Introduce Alembic when SQLAlchemy models are stabilized
  - Generate autogenerate diffs and remove raw SQL path
