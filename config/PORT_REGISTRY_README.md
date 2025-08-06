# Port Registry System for SUTAZAIAPP

## Quick Reference

This directory contains the comprehensive port management system for SUTAZAIAPP.

## Key Files

- **`port-registry.yaml`** - Central port allocation registry
- **`PORT_REGISTRY_README.md`** - This file

## Management Scripts

All scripts are located in `/opt/sutazaiapp/scripts/`:

### 1. Port Validation (`validate_ports.py`)
```bash
# Check for conflicts and compliance
python3 scripts/validate_ports.py

# Check if a specific port is available
python3 scripts/validate_ports.py --check-port 11075

# Suggest a port for a new service
python3 scripts/validate_ports.py --suggest "my-new-agent"

# Fix non-compliant services automatically
python3 scripts/validate_ports.py --fix
```

### 2. Port Migration (`migrate_agent_ports.py`)
```bash
# Dry run to see what would be migrated
python3 scripts/migrate_agent_ports.py

# Apply migrations
python3 scripts/migrate_agent_ports.py --apply

# Generate migration report
python3 scripts/migrate_agent_ports.py --report

# Rollback last migration
python3 scripts/migrate_agent_ports.py --rollback
```

### 3. Port Extraction (`extract_port_mappings.py`)
```bash
# Extract all current port mappings
python3 scripts/extract_port_mappings.py
```

## Port Ranges

| Range | Purpose | Status |
|-------|---------|--------|
| **10000-10199** | Infrastructure Services | 21.5% used |
| **10200-10299** | Monitoring Stack | 21.0% used |
| **10300-10499** | External Integrations | 25.0% used |
| **10500-10599** | AGI System | 13.0% used |
| **11000-11148** | AI Agents (STANDARD) | 46.3% used |
| **10104-11436** | Ollama LLM | 100% used |

## Current Status

As of 2025-08-05:
- ✅ **286** ports allocated across all services
- ❌ **26** port conflicts detected (needs resolution)
- ⚠️ **95** non-compliant agent services (needs migration)
- ✅ **80** available ports in agent range (11069-11148)

## Quick Actions

### Add a New Agent Service
1. Get next available port: `python3 scripts/validate_ports.py --suggest "agent-name"`
2. Update your docker-compose file with the allocated port
3. Update `port-registry.yaml` with the new allocation
4. Validate: `python3 scripts/validate_ports.py`

### Fix Port Conflicts
1. Run validation: `python3 scripts/validate_ports.py`
2. Review conflicts in the output
3. Either:
   - Auto-fix: `python3 scripts/validate_ports.py --fix`
   - Manual fix: Update docker-compose files with suggested ports

### Migrate Non-Compliant Services
1. Review needed migrations: `python3 scripts/migrate_agent_ports.py`
2. Apply migrations: `python3 scripts/migrate_agent_ports.py --apply`
3. Restart affected services
4. Update port registry

## Documentation

- Full strategy: `/opt/sutazaiapp/docs/infrastructure/PORT_ALLOCATION_STRATEGY.md`
- Port registry: `/opt/sutazaiapp/config/port-registry.yaml`
- Validation results: `port_validation_results.json` (generated after validation)
- Port mappings: `port_mappings.json` (generated after extraction)

## Best Practices

1. **Always validate** before deploying new services
2. **Use standard ranges** - agents MUST use 11000-11148
3. **Document changes** in port-registry.yaml
4. **No manual port selection** - use the suggest feature
5. **Resolve conflicts immediately** - don't deploy with conflicts

## Support

For issues or questions about port allocation:
1. Check validation output first
2. Review the port registry
3. Consult the full documentation
4. Contact System Architecture Team if needed

---
**Last Updated**: 2025-08-05
**Version**: 1.0.0