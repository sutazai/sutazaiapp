# CHANGELOG Policy Document
## Version 1.0 - 2025-08-20

## Purpose
This document establishes the official policy for CHANGELOG.md files in the SutazAI codebase to prevent proliferation and maintain single source of truth.

## Principles
1. **Minimal Coverage**: Only essential components need CHANGELOGs
2. **No Auto-Generation**: Never auto-generate template CHANGELOGs
3. **Actual History Only**: CHANGELOGs must contain real change history
4. **Single Source of Truth**: One authoritative CHANGELOG per component

## Required CHANGELOGs

### Level 1: Project Root
- `/opt/sutazaiapp/CHANGELOG.md` - Main project changelog (MANDATORY)

### Level 2: Major Modules
- `/opt/sutazaiapp/backend/CHANGELOG.md` - Backend service changes
- `/opt/sutazaiapp/frontend/CHANGELOG.md` - Frontend application changes
- `/opt/sutazaiapp/tests/CHANGELOG.md` - Test framework changes

### Level 3: Published Packages
Only directories containing:
- `package.json` (Node.js packages)
- `pyproject.toml` or `setup.py` (Python packages)
- Published or versioned components

### Level 4: MCP Components
- Active MCP server directories with actual implementations
- Must contain real server code, not just configuration

## Prohibited CHANGELOGs

### Never Create CHANGELOGs For:
1. **Empty directories** - No content to track
2. **Single-file directories** - Track in parent CHANGELOG
3. **Auto-generated directories** - (node_modules, .venv, build, dist)
4. **Configuration directories** - Unless versioned separately
5. **Utility script directories** - Track in parent module
6. **Test subdirectories** - Use parent test CHANGELOG
7. **Documentation subdirectories** - Use docs/CHANGELOG.md

## CHANGELOG Content Requirements

### Minimum Content
- At least one actual change entry
- Real timestamps (not placeholder dates)
- Meaningful change descriptions
- Author attribution for changes

### Format
Follow [Keep a Changelog](https://keepachangelog.com/) format:
- **Added** - New features
- **Changed** - Changes in existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Vulnerability fixes

## Enforcement

### Automated Checks
1. Pre-commit hooks to prevent empty CHANGELOG creation
2. CI/CD validation for CHANGELOG format
3. Regular cleanup of auto-generated CHANGELOGs

### Manual Review
1. PR reviews must verify CHANGELOG necessity
2. Quarterly audits to remove unnecessary CHANGELOGs
3. Documentation team approval for new CHANGELOGs

## Migration Plan

### Phase 1: Cleanup (Immediate)
- Remove 542 auto-generated template CHANGELOGs
- Consolidate duplicate change tracking

### Phase 2: Standardization (Week 1)
- Update remaining CHANGELOGs to standard format
- Add real content or remove if unnecessary

### Phase 3: Automation (Week 2)
- Implement pre-commit hooks
- Add CI/CD validation
- Create CHANGELOG generation tools for releases

## Exceptions
Exceptions to this policy require:
1. Technical justification
2. Architecture team approval
3. Documentation in this policy

## References
- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning](https://semver.org/)
- Rule 18: Change Tracking Requirements

## Policy Maintenance
- Review: Quarterly
- Owner: Architecture Team
- Last Updated: 2025-08-20
- Next Review: 2025-11-20