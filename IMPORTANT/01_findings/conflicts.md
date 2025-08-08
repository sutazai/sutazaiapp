# Conflicts Summary

- Database schema rules vs current SQL
  - Conflict: `DATABASE_SCHEMA.sql` uses `SERIAL` and integer FKs; standards mandate UUID PKs, FK indexes, env-driven secrets.
  - Sources: `/opt/sutazaiapp/IMPORTANT/DATABASE_SCHEMA.sql` (1–80), `/opt/sutazaiapp/IMPORTANT/init_db.sql` (1–40)

- Claimed vs actual system capabilities
  - Conflict: Claims of 69 agents and production readiness vs verified 7 stubs and partial POC.
  - Sources: `/opt/sutazaiapp/IMPORTANT/SUTAZAI_PRD.md` (~1–60), `/opt/sutazaiapp/IMPORTANT/SUTAZAI_SYSTEM_ARCHITECTURE_BLUEPRINT.md` (~1–60)

- Model mismatch
  - Conflict: Backend expects `gpt-oss` while TinyLlama is loaded.
  - Sources: `/opt/sutazaiapp/IMPORTANT/SUTAZAI_PRD.md` (~20–40)

- Service mesh routes missing
  - Conflict: Kong running but no routes defined.
  - Sources: `/opt/sutazaiapp/IMPORTANT/SUTAZAI_PRD.md` (~20–40)

- AuthN/AuthZ missing
  - Conflict: No authentication/authorization described for current system.
  - Sources: `/opt/sutazaiapp/IMPORTANT/SUTAZAI_PRD.md` (~40–60)

- Vector DB integration
  - Conflict: Vector DBs running but not integrated to backend flows.
  - Sources: `/opt/sutazaiapp/IMPORTANT/SUTAZAI_PRD.md` (~20–40)

- Document drift/duplication
  - Conflict: Multiple versions of standards and roadmap docs.
  - Sources: `/opt/sutazaiapp/IMPORTANT/COMPREHENSIVE_ENGINEERING_STANDARDS*.md`, `/opt/sutazaiapp/IMPORTANT/Archives/SYSTEM_ROADMAP_BIBLE*.md`

- Placeholder/empty doc
  - Conflict: `Sutazai_Strategy _Plan_IMPORTANT.md` is empty.
  - Sources: `/opt/sutazaiapp/IMPORTANT/Sutazai_Strategy _Plan_IMPORTANT.md`
