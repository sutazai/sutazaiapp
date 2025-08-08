# Conflicts Summary

- Database schema rules vs current SQL
  - Conflict: `DATABASE_SCHEMA.sql` uses `SERIAL` and integer FKs; project rules mandate UUID PKs, FK indexes, env-driven secrets.
  - Sources: `/workspace/IMPORTANT/DATABASE_SCHEMA.sql` (lines 1–80), `/opt/sutazaiapp/IMPORTANT/sql/init_db.sql` (lines 1–20, 65–74)

- Claimed vs actual system capabilities
  - Conflict: Claims of 69 agents and production readiness vs verified 7 stubs and partial POC.
  - Sources: `/workspace/IMPORTANT/SUTAZAI_PRD.md` (Current State Assessment, ~lines 1–60), `/workspace/IMPORTANT/SUTAZAI_SYSTEM_ARCHITECTURE_BLUEPRINT.md` (Executive Summary, ~1–60)

- Model mismatch
  - Conflict: Backend expects `gpt-oss` while TinyLlama is loaded.
  - Sources: `/workspace/IMPORTANT/SUTAZAI_PRD.md` (~lines 20–40)

- Service mesh routes missing
  - Conflict: Kong running but no routes defined.
  - Sources: `/workspace/IMPORTANT/SUTAZAI_PRD.md` (~lines 20–40)

- AuthN/AuthZ missing
  - Conflict: No authentication/authorization described for current system.
  - Sources: `/workspace/IMPORTANT/SUTAZAI_PRD.md` (~lines 40–60)

- Vector DB integration
  - Conflict: Vector DBs running but not integrated to backend flows.
  - Sources: `/workspace/IMPORTANT/SUTAZAI_PRD.md` (~lines 20–40)

- Document drift/duplication
  - Conflict: Multiple versions of standards and roadmap docs.
  - Sources: `/workspace/IMPORTANT/COMPREHENSIVE_ENGINEERING_STANDARDS.md` and `..._FULL.md` (md5 differs), `/workspace/IMPORTANT/Archives/SYSTEM_ROADMAP_BIBLE*.md`

- Placeholder/empty doc
  - Conflict: `Sutazai_Strategy _Plan_IMPORTANT.md` is empty.
  - Sources: `/workspace/IMPORTANT/Sutazai_Strategy _Plan_IMPORTANT.md`