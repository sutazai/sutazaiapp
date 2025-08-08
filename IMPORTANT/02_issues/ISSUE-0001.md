# ISSUE-0001: Database Schema Mismatch (UUID vs SERIAL)

- Impacted Components: Backend API, DB migrations, Services using DB
- Options:
  - A: Keep SERIAL ints (quick, violates standards)
  - B: Migrate to UUID PKs with `gen_random_uuid()`; add FK indexes (recommended)
  - C: Hybrid (new tables UUID, legacy SERIAL) with adapters
- Recommendation: B
- Consequences: Requires migration scripts, code updates to UUID handling
- Sources: `/workspace/IMPORTANT/DATABASE_SCHEMA.sql` (1–80), `/opt/sutazaiapp/IMPORTANT/sql/init_db.sql` (1–20, 65–74)