# ISSUE-0008: Hardcoded Secret in init_db.sql

- Impacted: Security, Compliance
- Options:
  - A: Replace with env-driven secrets via Docker/compose (recommended)
  - B: Use Vault-like KMS (future)
- Recommendation: A
- Consequences: Update compose/env and SQL bootstrap
- Sources: `/opt/sutazaiapp/IMPORTANT/init_db.sql` (line 1)
