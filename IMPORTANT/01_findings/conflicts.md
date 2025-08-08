# Conflicts Summary

- Database schema rules vs current SQL  
  Conflict: `DATABASE_SCHEMA.sql` uses `SERIAL` and integer FKs; standards mandate UUID PKs, FK indexes, env-driven secrets.  
  [source] /opt/sutazaiapp/IMPORTANT/DATABASE_SCHEMA.sql#L1-L120; /opt/sutazaiapp/IMPORTANT/10_canonical/standards/ADR-0001.md#L1-L40

- Claimed vs actual system capabilities  
  Conflict: Claims of 60+ agents and production readiness vs verified handful of stubs and partial POC.  
  [source] /opt/sutazaiapp/IMPORTANT/SUTAZAI_PRD.md#L1-L120; /opt/sutazaiapp/IMPORTANT/Archives/SYSTEM_TRUTH_SUMMARY.md#L1-L80

- Model mismatch  
  Conflict: Backend docs reference `gpt-oss` while TinyLlama is the loaded/approved model.  
  [source] /opt/sutazaiapp/IMPORTANT/10_canonical/standards/ADR-0002.md#L1-L40; /opt/sutazaiapp/IMPORTANT/PHASE1_EXECUTIVE_SUMMARY.md#L20-L40

- Service mesh routes missing  
  Conflict: Kong mentioned in docs but route/service definitions are absent.  
  [source] /opt/sutazaiapp/IMPORTANT/20_plan/quick_wins.md#L1-L40; /opt/sutazaiapp/IMPORTANT/PHASE1_EXECUTIVE_SUMMARY.md#L60-L100

- AuthN/AuthZ missing  
  Conflict: No authentication/authorization in current state despite enterprise claims.  
  [source] /opt/sutazaiapp/IMPORTANT/PHASE1_EXECUTIVE_SUMMARY.md#L40-L70; /opt/sutazaiapp/IMPORTANT/10_canonical/security/security_privacy.md#L1-L100

- Vector DB integration  
  Conflict: Vector DBs referenced but not wired to backend flows.  
  [source] /opt/sutazaiapp/IMPORTANT/10_canonical/current_state/system_reality.md#L1-L120; /opt/sutazaiapp/IMPORTANT/20_plan/gap_analysis.md#L1-L60

- Documentation drift/duplication  
  Conflict: Multiple versions of standards and roadmap docs across root, nested, and Archives.  
  [source] /opt/sutazaiapp/IMPORTANT/00_inventory/inventory.md#L1-L999; /opt/sutazaiapp/IMPORTANT/00_inventory/deduplication_analysis.json#L1-L200

- Placeholder/empty doc  
  Conflict: `Sutazai_Strategy _Plan_IMPORTANT.md` is empty.  
  [source] /opt/sutazaiapp/IMPORTANT/Sutazai_Strategy _Plan_IMPORTANT.md#L1-L5
